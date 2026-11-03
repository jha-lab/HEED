"""Stage 1/2 alignment losses (C2 HSA, C3 RSA, C4 HEED), the Stage 3 KD loss (Eq. 1) and the DPO loss."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def residual_alignment_loss(student_res, teacher_res, valid_mask, weights=None):
    """Eq. 4 (weights=None, C3 RSA) or Eq. 5 (density weights w(p), C4 HEED).

    student_res / teacher_res: residual streams r_{l,p} at the readout layers S* (one (b, T, d) tensor per layer).
    Each sample is normalised by its own T, then averaged over layers and samples.
    """
    T = valid_mask.sum(1).clamp_min(1)
    w = valid_mask.float() if weights is None else weights
    total = 0.0
    for s, t in zip(student_res, teacher_res):
        d = (s.float() - t.float()).pow(2).sum(-1)
        total = total + ((w * d).sum(1) / T).mean()
    return total / len(student_res)


def hsa_loss(pairs, valid_mask):
    """C2 (mmMamba-style): per-layer mixer-output MSE, sum_l ||O_hat^l - O^l||^2, with teacher inputs at every layer."""
    m = valid_mask.float()
    T = m.sum(1).clamp_min(1)
    total = 0.0
    for t, s in pairs:
        total = total + (((s.float() - t.float()).pow(2).sum(-1) * m).sum(1) / T).mean()
    return total / len(pairs)


def _kd_chunk(weight, hs, ht, tgt):
    s = F.linear(hs, weight).float().log_softmax(-1)
    with torch.no_grad():
        t = F.linear(ht, weight).float().log_softmax(-1)
    return F.kl_div(s, t, log_target=True, reduction="sum"), F.nll_loss(s, tgt, reduction="sum")


def kd_loss(lm_head, s_hidden, t_hidden, input_ids, loss_mask, lambda_kl=1.0, lambda_ce=0.1, chunk=2048):
    """lambda_KL * KL(teacher || student) + lambda_CE * CE(student; D) over the supervised next-token positions.

    Logits are produced chunk by chunk under activation checkpointing so the (T, vocab) tensors are never stored.
    """
    m = loss_mask[:, 1:]
    hs, ht, tgt = s_hidden[:, :-1][m], t_hidden[:, :-1][m], input_ids[:, 1:][m]
    n = tgt.numel()
    if n == 0:
        zero = s_hidden.sum() * 0
        return zero, zero.detach(), zero.detach()
    kl = ce = 0.0
    for i in range(0, n, chunk):
        k, c = checkpoint(_kd_chunk, lm_head.weight, hs[i:i + chunk], ht[i:i + chunk], tgt[i:i + chunk],
                          use_reentrant=False)
        kl, ce = kl + k, ce + c
    kl, ce = kl / n, ce / n
    return lambda_kl * kl + lambda_ce * ce, kl.detach(), ce.detach()


def dpo_loss(logp, ref_logp, beta=0.1):
    """DPO on answer log-probabilities under the model and the reference: chosen answers first, then rejected ones."""
    chosen, rejected = (logp - ref_logp).chunk(2)
    return -F.logsigmoid(beta * (chosen - rejected)).mean()
