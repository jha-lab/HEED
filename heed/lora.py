"""LoRA adapters for DPO: a low-rank update trained on top of frozen linear layers and merged into them afterwards."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return y


def add_lora(model, r=32, alpha=64, dropout=0.05):
    """Adapters on every linear layer outside the vision encoder."""
    for name, parent in list(model.named_modules()):
        if not name.startswith("model.visual"):
            for child_name, child in parent.named_children():
                if isinstance(child, nn.Linear):
                    setattr(parent, child_name, LoraLinear(child, r, alpha, dropout))


def set_lora(model, enabled):
    """With the adapters off the model is the frozen DPO reference."""
    for m in model.modules():
        if isinstance(m, LoraLinear):
            m.enabled = enabled


@torch.no_grad()
def merge_lora(model):
    for parent in list(model.modules()):
        for name, child in parent.named_children():
            if isinstance(child, LoraLinear):
                child.base.weight += (child.scale * child.lora_b @ child.lora_a).to(child.base.weight.dtype)
                setattr(parent, name, child.base)
