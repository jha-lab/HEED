"""Teacher and 3:1 Mamba-2/attention student in one Qwen3-VL model.

Everything except the replaced attention blocks (vision encoder, projector, embeddings, RMSNorms, MLPs, retained
attention layers, LM head) is frozen and identical for teacher and student, so both share a single copy. Each
replaced layer holds the teacher attention and its Mamba-2 replacement; `set_mode` picks which one runs.
"""
import json
import os

import torch
import torch.nn as nn

from .mamba2 import Mamba2Mixer


class HybridMixer(nn.Module):
    def __init__(self, attn, mamba):
        super().__init__()
        self.attn, self.mamba = attn, mamba
        self.mode = "teacher"
        self.hsa_pair = None

    def forward(self, hidden_states, **kwargs):
        if self.mode == "student":
            return self.mamba(hidden_states, past_key_values=kwargs.get("past_key_values"),
                              attention_mask=kwargs.get("attention_mask")), None
        out, weights = self.attn(hidden_states, **kwargs)
        if self.mode == "hsa":
            # C2 (mmMamba-style): the student block sees the teacher's input; the teacher stream continues
            self.hsa_pair = (out.detach(), self.mamba(hidden_states.detach()))
        return out, weights


def uniform_mamba_layers(num_layers, attn_every=4):
    """3:1 hybrid: keep attention in every 4th layer (3, 7, ..., 35 for 36 layers) and replace the rest."""
    return [i for i in range(num_layers) if i % attn_every != attn_every - 1]


def decoder_layers(model):
    return model.model.language_model.layers


def to_hybrid(model, mamba_layers):
    """Replace the attention of `mamba_layers` with WT-initialised Mamba-2 blocks (fp32 master weights)."""
    tc = model.config.text_config
    layers = decoder_layers(model)
    for i in mamba_layers:
        attn = layers[i].self_attn
        with torch.device(attn.q_proj.weight.device):
            mamba = Mamba2Mixer(tc.hidden_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim,
                                eps=tc.rms_norm_eps, layer_idx=i)
        mamba.transfer_from_attention(attn)
        layers[i].self_attn = HybridMixer(attn, mamba)
    model.heed_mamba_layers = list(mamba_layers)
    return model


def mixers(model):
    return [m for m in model.modules() if isinstance(m, HybridMixer)]


def set_mode(model, mode):
    assert mode in ("teacher", "student", "hsa")
    for m in mixers(model):
        m.mode = mode
        m.hsa_pair = None
    # generate() sizes its cache from layer_types: the student's Mamba-2 layers keep a recurrent state, not KV
    tc = model.config.text_config
    if mode == "student":
        tc.layer_types = ["linear_attention" if i in model.heed_mamba_layers else "full_attention"
                          for i in range(tc.num_hidden_layers)]
    elif hasattr(tc, "layer_types"):
        del tc.layer_types


def mamba_parameters(model):
    return [p for m in mixers(model) for p in m.mamba.parameters()]


def set_stage_trainable(model, stage):
    """Stage 1 keeps W_{Q,K,V,O} at their WT values; Stages 2 and 3 train the full Mamba-2 blocks."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mixers(model):
        for name, p in m.mamba.named_parameters():
            p.requires_grad_(stage > 1 or name.split(".")[0] not in Mamba2Mixer.TRANSFERRED)


def save_student(model, path, base_model, dtype=torch.bfloat16, post_trained=False, **meta):
    """Distillation trains only the Mamba-2 blocks; SFT and DPO (`post_trained`) train all but the vision encoder."""
    os.makedirs(path, exist_ok=True)
    state = {k: v.to(dtype) for k, v in model.state_dict().items()
             if (".visual." not in k if post_trained else ".mamba." in k)}
    torch.save(state, os.path.join(path, "student.pt"))
    with open(os.path.join(path, "heed_student.json"), "w") as f:
        json.dump({"base_model": base_model, "mamba_layers": model.heed_mamba_layers, **meta}, f, indent=2)


def for_inference(model):
    """Student-only model for generation: drops the replaced teacher attention and casts Mamba-2 to the model dtype."""
    dtype = model.get_input_embeddings().weight.dtype
    set_mode(model, "student")
    for m in mixers(model):
        m.attn = None
        m.mamba.to(dtype)
    return model


def load_student(model, path):
    """Turn a freshly loaded teacher into the student saved at `path`."""
    with open(os.path.join(path, "heed_student.json")) as f:
        cfg = json.load(f)
    to_hybrid(model, cfg["mamba_layers"])
    state = torch.load(os.path.join(path, "student.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not [k for k in missing if ".mamba." in k], "student checkpoint does not match"
    return for_inference(model)
