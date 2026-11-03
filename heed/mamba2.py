"""Mamba-2 mixer that replaces a Qwen3-VL attention block, initialised by SSD weight transfer (Sect. 4.1)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.simple_gla import chunk_simple_gla
except ImportError:  # CPU-only installs fall back to the reference scan below
    chunk_simple_gla = None


def _segsum(x):
    """out[..., i, j] = sum(x[..., j+1:i+1]) for j <= i and -inf above the diagonal."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    x = x.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1), 0)
    x = torch.cumsum(x, dim=-2)
    return x.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0), -torch.inf)


def ssd_torch(x, dt, A, B, C, chunk_size=64, initial_state=None):
    """Reference chunked SSD scan (Mamba-2 paper, Listing 1).

    x: (b, t, h, p), dt: (b, t, h), A: (h,), B and C: (b, t, h, n), initial_state: (b, h, n, p) or None.
    Returns y: (b, t, h, p) and the final state (b, h, n, p), both in fp32.
    """
    b, t, h, p = x.shape
    f = torch.float64 if x.dtype == torch.float64 else torch.float32
    X = x.to(f) * dt.to(f)[..., None]
    a = dt.to(f) * A.to(f)
    B, C = B.to(f), C.to(f)
    pad = (-t) % chunk_size
    if pad:
        X, B, C = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (X, B, C))
        a = F.pad(a, (0, 0, 0, pad))
    c = X.shape[1] // chunk_size
    X, B, C = (z.reshape(b, c, chunk_size, *z.shape[2:]) for z in (X, B, C))
    a = a.reshape(b, c, chunk_size, h).permute(0, 3, 1, 2)  # (b, h, c, l)
    a_cumsum = torch.cumsum(a, dim=-1)

    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, torch.exp(_segsum(a)), X)
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    h0 = torch.zeros_like(states[:, :1]) if initial_state is None else initial_state.to(f).transpose(-1, -2)[:, None]
    states = torch.cat([h0, states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = states[:, :-1], states[:, -1]
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(a_cumsum))
    return (y_diag + y_off).reshape(b, c * chunk_size, h, p)[:, :t], final_state.transpose(-1, -2)


def ssd(x, dt, A, B, C, initial_state=None, output_final_state=False):
    """Mamba-2 SSD scan. With q=C, k=B, v=dt*x and log-decay g=dt*A it is exactly simple GLA, so we use fla's kernel.

    Returns y and, if `output_final_state`, the state after the last token (b, h, n, p) in fp32 (else None).
    """
    if chunk_simple_gla is None or not x.is_cuda:
        y, state = ssd_torch(x, dt, A, B, C, initial_state=initial_state)
        return y, state if output_final_state else None
    v = (x * dt[..., None].to(x.dtype)).contiguous()
    y, state = chunk_simple_gla(q=C.contiguous(), k=B.contiguous(), v=v, g=(dt * A).contiguous(), scale=1.0,
                                initial_state=initial_state, output_final_state=output_final_state)
    return y, None if state is None else state.float()


def ssd_step(x, dt, A, B, C, state):
    """One decoding step of the recurrence S <- exp(dt A) S + B (dt x)^T, y = S^T C, in fp32 (like ssd_torch).

    x: (b, h, p), dt: (b, h), A: (h,), B and C: (b, h, n), state: (b, h, n, p). Returns y: (b, h, p) and the new state.
    """
    f = torch.float64 if x.dtype == torch.float64 else torch.float32
    dt = dt.to(f)
    decay = torch.exp(dt * A.to(f))[..., None, None]
    state = decay * state.to(f) + B.to(f)[..., None] * (dt[..., None] * x.to(f))[..., None, :]
    return torch.einsum("bhn,bhnp->bhp", C.to(f), state), state


def _keep_mask(attention_mask, t):
    """(b, t) mask of real tokens at prefill, from the 2-D or 4-D attention mask the decoder layers pass down.

    Generation left-pads batches, so the last query is a real token and its mask row marks every real key.
    """
    if not isinstance(attention_mask, torch.Tensor):
        return None
    if attention_mask.dim() == 4:
        row = attention_mask[:, 0, -1, -t:]
        return row if row.dtype == torch.bool else row == 0
    return attention_mask[:, -t:].bool()


class Mamba2Mixer(nn.Module):
    """Mamba-2 block shaped like a GQA attention block.

    C <- W_Q, B <- W_K, x <- W_V and the output projection <- W_O (Mamba-in-Llama / mmMamba weight transfer).
    The gate W_G (z_proj), Delta-projector W_gamma (dt_proj), A and the 1-D conv W_conv are new.
    x and B have one head per KV head and are shared across the query heads of their group, as in GQA.
    For generation, `layer_idx` is this block's slot in a transformers Cache (a linear-attention layer there).
    """

    TRANSFERRED = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, d_conv=4, eps=1e-6, layer_idx=None):
        super().__init__()
        self.h, self.hkv, self.p = num_heads, num_kv_heads, head_dim
        self.d_conv, self.layer_idx = d_conv, layer_idx
        self.n = head_dim  # SSM state size = attention head dim, so W_Q / W_K map onto C / B
        d_inner, d_xb = num_heads * head_dim, num_kv_heads * head_dim
        self.d_xb = d_xb

        self.q_proj = nn.Linear(hidden_size, d_inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, d_xb, bias=False)
        self.v_proj = nn.Linear(hidden_size, d_xb, bias=False)
        self.o_proj = nn.Linear(d_inner, hidden_size, bias=False)

        self.z_proj = nn.Linear(hidden_size, d_inner, bias=False)
        self.dt_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(num_heads))
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.D = nn.Parameter(torch.empty(num_heads))
        conv_dim = 2 * d_xb + d_inner
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        self.norm_weight = nn.Parameter(torch.empty(d_inner))  # gated RMSNorm, one group per head
        self.eps = eps
        self.reset_new_parameters()

    @torch.no_grad()
    def reset_new_parameters(self, std=0.02, dt_min=1e-3, dt_max=1e-1):
        # App. C.3: W_G, W_gamma, A and W_conv are truncated-normal with sigma = 0.02
        for w in (self.z_proj.weight, self.dt_proj.weight, self.conv1d.weight, self.A_log):
            nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.conv1d.bias)
        dt = torch.exp(torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus, standard Mamba-2 dt init
        self.D.fill_(1.0)
        self.norm_weight.fill_(1.0)

    @torch.no_grad()
    def transfer_from_attention(self, attn):
        for name in self.TRANSFERRED:
            getattr(self, name).weight.copy_(getattr(attn, name).weight)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None):
        """Without a cache: one parallel scan over the sequence (training). With a cache: prefill or decode, keeping
        the last d_conv conv inputs and the SSM state, so a new token costs O(1) here at any context length."""
        b, t, _ = hidden_states.shape
        cache, state, keep = past_key_values, None, None
        if cache is not None:
            state = cache.get_recurrent_state(self.layer_idx)
            if state is None:
                keep = _keep_mask(attention_mask, t)
        if keep is not None:  # left padding must reach neither the conv window nor the state
            hidden_states = hidden_states * keep[..., None].to(hidden_states.dtype)

        xBC = torch.cat([self.v_proj(hidden_states), self.k_proj(hidden_states), self.q_proj(hidden_states)], dim=-1)
        xBC = xBC.transpose(1, 2)
        if cache is not None:  # prepends the cached conv inputs
            xBC = cache.update_conv_state(xBC, self.layer_idx, conv_kernel_size=self.d_conv)
        xBC = F.silu(self.conv1d(xBC)[..., :xBC.shape[-1]][..., -t:].transpose(1, 2))
        if keep is not None:
            xBC = xBC * keep[..., None].to(xBC.dtype)
        x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.h * self.n], dim=-1)
        rep = self.h // self.hkv
        x = x.reshape(b, t, self.hkv, self.p).repeat_interleave(rep, dim=2)
        B = B.reshape(b, t, self.hkv, self.n).repeat_interleave(rep, dim=2)
        C = C.reshape(b, t, self.h, self.n)

        dt = F.softplus(self.dt_proj(hidden_states).float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        if state is not None and t == 1:
            y, state = ssd_step(x[:, 0], dt[:, 0], A, B[:, 0], C[:, 0], state)
            y = y[:, None]
        else:
            y, state = ssd(x, dt, A, B, C, initial_state=state, output_final_state=cache is not None)
        if cache is not None:
            cache.update_recurrent_state(state, self.layer_idx)
        y = y.float() + x.float() * self.D.float()[:, None]

        y = y * F.silu(self.z_proj(hidden_states).float().reshape(b, t, self.h, self.p))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = (y * self.norm_weight.float().reshape(self.h, self.p)).reshape(b, t, -1)
        return self.o_proj(y.to(hidden_states.dtype))


def repeat_kv_heads(t, num_kv_heads, rep, dim=0):
    """Repeat each of the `num_kv_heads` blocks of `t` along `dim` `rep` times (block i of the result is i // rep)."""
    dim = dim % t.dim()
    t = t.unflatten(dim, (num_kv_heads, 1, -1))
    return t.expand(*t.shape[:dim + 1], rep, *t.shape[dim + 2:]).flatten(dim, dim + 2)


def to_mamba_mixer2(w, num_heads, num_kv_heads):
    """A Mamba2Mixer state dict `w` in the layout of SGLang's MambaMixer2: standard Mamba-2, one B/C group per head.

    in_proj = [W_G; W_V; W_K; W_Q; W_gamma] keeps x and B at KV-head width; heed/serving repeats them per head
    after the projection, so only the depthwise conv weights of x and B are repeated here.
    """
    rep, d_xb = num_heads // num_kv_heads, w["k_proj.weight"].shape[0]

    def conv(c):
        x, B, C = c.split([d_xb, d_xb, c.shape[0] - 2 * d_xb])
        return torch.cat([repeat_kv_heads(x, num_kv_heads, rep), repeat_kv_heads(B, num_kv_heads, rep), C])

    return {
        "in_proj.weight": torch.cat([w[f"{n}.weight"] for n in ("z_proj", "v_proj", "k_proj", "q_proj", "dt_proj")]),
        "conv1d.weight": conv(w["conv1d.weight"]),
        "conv1d.bias": conv(w["conv1d.bias"]),
        "A": -torch.exp(w["A_log"].float()),
        "D": w["D"],
        "dt_bias": w["dt_bias"],
        "norm.weight": w["norm_weight"],
        "out_proj.weight": w["o_proj.weight"],
    }
