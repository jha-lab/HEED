"""HEED student in SGLang: SGLang's Qwen3-VL with Mamba-2 blocks in place of the attention of `heed_mamba_layers`.

Selected by SGLANG_EXTERNAL_MODEL_PACKAGE=heed.serving for a checkpoint written by scripts/export.py
(served with --trust-remote-code). The class keeps SGLang's architecture name, so its Qwen3-VL processor, chat template
and server settings apply unchanged, and a config without `heed_mamba_layers` (the teacher) gives SGLang's Qwen3-VL.
The blocks run SGLang's MambaMixer2 (state pool, continuous batching, chunked prefill, decode CUDA graphs) on the
weights of heed.mamba2.Mamba2Mixer in the layout of heed.mamba2.to_mamba_mixer2.
"""
import re
from collections import defaultdict

import torch
from torch import nn

from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend, Mamba2AttnBackend
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models import qwen3_vl
from sglang.srt.runtime_context import get_parallel

from heed.mamba2 import repeat_kv_heads, to_mamba_mixer2


class KVWidthInProj(nn.Module):
    """in_proj of MambaMixer2 that projects x and B at KV-head width and repeats them per query head afterwards."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, prefix=""):
        super().__init__()
        self.hkv, self.rep = num_kv_heads, num_heads // num_kv_heads
        d_inner, d_xb = num_heads * head_dim, num_kv_heads * head_dim
        self.sizes = [d_inner, d_xb, d_xb, d_inner, num_heads]  # z, x, B, C, dt
        self.proj = ReplicatedLinear(hidden_size, sum(self.sizes), bias=False, prefix=prefix)

    def forward(self, hidden_states):
        z, x, B, C, dt = self.proj(hidden_states)[0].split(self.sizes, dim=-1)
        x, B = (repeat_kv_heads(v, self.hkv, self.rep, dim=-1) for v in (x, B))
        return torch.cat([z, x, B, C, dt], dim=-1), None


class HeedMamba2(nn.Module):
    """A student Mamba-2 block in the place, and with the call signature, of Qwen3Attention."""

    def __init__(self, config, layer_id, prefix=""):
        super().__init__()
        assert get_parallel().tp_size == 1, "HEED student: tensor parallelism is not implemented"
        tc = config.text_config
        self.layer_id = layer_id
        self.heads = (tc.num_attention_heads, tc.num_key_value_heads)
        self.mixer = MambaMixer2(cache_params=config.mamba2_cache_params, hidden_size=tc.hidden_size,
                                 use_conv_bias=True, use_bias=False, n_groups=tc.num_attention_heads,
                                 rms_norm_eps=tc.rms_norm_eps, prefix=f"{prefix}.mixer")
        self.mixer.in_proj = KVWidthInProj(tc.hidden_size, *self.heads, tc.head_dim, prefix=f"{prefix}.mixer.in_proj")

    def forward(self, positions, hidden_states, forward_batch):
        backend = get_attn_backend()
        assert isinstance(backend, HybridLinearAttnBackend)
        assert isinstance(backend.linear_attn_backend, Mamba2AttnBackend)
        return backend.linear_attn_backend.forward(mixer=self.mixer, layer_id=self.layer_id,
                                                   hidden_states=hidden_states, output=None,
                                                   forward_batch=forward_batch, use_triton_causal_conv=True)

    @torch.no_grad()
    def load_heed_weights(self, w):
        """`w`: this block's Mamba2Mixer tensors as heed.hybrid.save_student writes them."""
        params = dict(self.mixer.named_parameters())
        values = {name.replace("in_proj.", "in_proj.proj."): v for name, v in to_mamba_mixer2(w, *self.heads).items()}
        assert values.keys() == params.keys(), sorted(values.keys() ^ params.keys())
        for name, v in values.items():
            assert params[name].shape == v.shape, (name, params[name].shape, v.shape)
            params[name].copy_(v)


class Qwen3VLForConditionalGeneration(qwen3_vl.Qwen3VLForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.heed_mamba_layers = list(getattr(config, "heed_mamba_layers", None) or [])
        for i in self.heed_mamba_layers:  # replaces the attention built above
            self.model.layers[i].self_attn = HeedMamba2(
                config, i, prefix=f"{prefix}model.language_model.layers.{i}.self_attn")

    def load_weights(self, weights):
        mamba = defaultdict(dict)

        def others():
            for name, w in weights:
                m = re.search(r"layers\.(\d+)\.self_attn\.mamba\.(.+)", name)
                if m:
                    mamba[int(m[1])][m[2]] = w
                else:  # the teacher attention of the Mamba-2 layers matches no parameter and is skipped
                    yield name, w

        super().load_weights(others())
        assert sorted(mamba) == self.heed_mamba_layers, "checkpoint and config disagree on the Mamba-2 layers"
        for i, w in mamba.items():
            self.model.layers[i].self_attn.load_heed_weights(w)


EntryClass = Qwen3VLForConditionalGeneration
