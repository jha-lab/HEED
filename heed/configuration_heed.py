"""Config of a HEED student checkpoint for SGLang; scripts/export.py copies this file next to its config.json.

SGLang loads it with trust_remote_code in every process, before it checks whether the model is a hybrid, so the
Mamba-2 state cache of the student is registered here. Without SGLang it is a plain Qwen3VLConfig.
"""
from transformers import Qwen3VLConfig

try:
    from sglang.srt.configs.linear_attn_model_registry import LinearAttnModelSpec, register_linear_attn_model
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype
    from sglang.srt.runtime_context import get_parallel
except ImportError:
    register_linear_attn_model = None


class HeedQwen3VLConfig(Qwen3VLConfig):
    """`heed_mamba_layers`: decoder layers whose attention is a Mamba-2 block; `heed_d_conv`: its conv width."""

    @property
    def full_attention_layer_ids(self):
        return [i for i in range(self.text_config.num_hidden_layers) if i not in self.heed_mamba_layers]

    @property
    def mamba2_cache_params(self):
        # x and B are repeated per query head before the conv (heed/serving): one B/C group per head
        tc = self.text_config
        h, p = tc.num_attention_heads, tc.head_dim
        shape = Mamba2StateShape.create(tp_world_size=get_parallel().attn_tp_size, intermediate_size=h * p, n_groups=h,
                                        num_heads=h, head_dim=p, state_size=p, conv_kernel=self.heed_d_conv)
        return Mamba2CacheParams(shape=shape, layers=self.heed_mamba_layers, dtype=mamba2_state_dtype(self))


if register_linear_attn_model is not None:
    register_linear_attn_model(LinearAttnModelSpec(
        config_class=HeedQwen3VLConfig,
        backend_class_name="sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend",
        config_predicate=lambda config: bool(getattr(config, "heed_mamba_layers", None)),
    ))
