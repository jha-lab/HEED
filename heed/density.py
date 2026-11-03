"""Patch self-dissimilarity density (Eq. 2), HEED position weights (Eq. 3 / Alg. 1) and the 4-bit density cache."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def patch_density(vit_feats, grid_thw, merge_size=2):
    """rho(p) = 1 - mean cosine similarity to the 3x3 neighbourhood, for every LLM visual token.

    vit_feats: final pre-merger (pre-projector) ViT features, (sum_i t_i*h_i*w_i, C), in the Qwen-VL processor's
    merge-window order (t, h/m, w/m, m, m). Density is computed on the patch grid with reflected padding and averaged
    over each m x m merge window, so the result follows the order of the image's LLM visual tokens (t, h/m, w/m).
    Returns one 1-D fp32 tensor per image.
    """
    out, start, m = [], 0, merge_size
    for t, h, w in grid_thw.tolist():
        f = vit_feats[start:start + t * h * w].float()
        start += t * h * w
        f = f.reshape(t, h // m, w // m, m, m, -1).permute(0, 5, 1, 3, 2, 4).reshape(t, -1, h, w)
        f = F.normalize(f, dim=1)
        fp = F.pad(f, (1, 1, 1, 1), mode="reflect")
        sim = sum((fp[:, :, dy:dy + h, dx:dx + w] * f).sum(1)
                  for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1))
        rho = 1 - sim / 8
        out.append(rho.reshape(t, h // m, m, w // m, m).mean((2, 4)).reshape(-1))
    return out


def heed_weights(rho_visual, visual_mask, valid_mask, tau=0.5, beta=2.0):
    """w(p) for a batch (Alg. 1).

    rho_visual: per sample, the raw densities of its visual tokens in sequence order (None for text-only samples).
    Text positions get rho_text = beta * mean visual density; densities are min-max normalised per sample,
    mapped through exp(rho / tau) and rescaled so that sum_p w(p) = T over the valid positions. Padding gets 0 and
    text-only samples get uniform weights.
    """
    w = torch.zeros(valid_mask.shape, dtype=torch.float32, device=valid_mask.device)
    for i, rv in enumerate(rho_visual):
        valid = valid_mask[i]
        if rv is None or rv.numel() == 0:
            w[i, valid] = 1.0
            continue
        vis = visual_mask[i] & valid
        assert int(vis.sum()) == rv.numel(), "visual tokens and density entries do not line up"
        rho = torch.full_like(w[i], beta * float(rv.mean()))
        rho[vis] = rv.to(rho)
        r = rho[valid]
        r = (r - r.min()) / (r.max() - r.min()).clamp_min(1e-6)
        e = torch.exp(r / tau)
        w[i, valid] = e * valid.sum() / e.sum()
    return w


def quantize_density(rho):
    """Raw visual densities -> (header [min, max] fp32, 4-bit codes packed two per byte)."""
    rho = rho.float().cpu()
    lo, hi = rho.min(), rho.max()
    q = torch.round((rho - lo) / (hi - lo).clamp_min(1e-6) * 15).to(torch.uint8)
    if q.numel() % 2:
        q = F.pad(q, (0, 1))
    header = np.array([lo, hi], dtype=np.float32)
    return header, (q[0::2] | (q[1::2] << 4)).numpy()


def dequantize_density(header, packed, n):
    lo, hi = header.tolist()
    q = np.stack([packed & 15, packed >> 4], axis=1).reshape(-1)[:n]
    return torch.from_numpy(lo + q.astype(np.float32) / 15 * (hi - lo))


class DensityCacheWriter:
    """Writes `<prefix>.bin` (64-byte aligned records) and `<prefix>.idx.json` (sample id -> [offset, n])."""

    def __init__(self, prefix):
        self.prefix = prefix
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        self.f = open(prefix + ".bin", "wb")
        self.index = {}

    def add(self, key, rho):
        header, packed = quantize_density(rho)
        self.index[key] = [self.f.tell(), int(rho.numel())]
        self.f.write(header.tobytes())
        self.f.write(packed.tobytes())
        self.f.write(b"\0" * ((-self.f.tell()) % 64))

    def close(self):
        self.f.close()
        with open(self.prefix + ".idx.json", "w") as f:
            json.dump(self.index, f)


class DensityCache:
    def __init__(self, prefix):
        self.data = np.memmap(prefix + ".bin", dtype=np.uint8, mode="r")
        with open(prefix + ".idx.json") as f:
            self.index = json.load(f)

    def __contains__(self, key):
        return key in self.index

    def get(self, key):
        """Dequantised visual densities of one sample (text positions are derived from them at runtime)."""
        off, n = self.index[key]
        header = np.frombuffer(self.data[off:off + 8].tobytes(), dtype=np.float32)
        return dequantize_density(header, np.asarray(self.data[off + 8:off + 8 + (n + 1) // 2]), n)
