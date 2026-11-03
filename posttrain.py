"""HEED Phases 2 and 3: SFT and DPO of a distilled student (App. C.4, C.5).

    Phase 2 (SFT): answer-token CE; everything but the vision encoder trains
    Phase 3 (DPO): DPO loss against the SFT checkpoint, which is the model with its adapters off; LoRA (r = 32,
                   alpha = 64, dropout 0.05) on the linear layers outside the vision encoder, merged at the end

    torchrun --nproc_per_node 8 posttrain.py --phase sft --student <dir> --train_data <sft.jsonl> --output_dir <sft>
    torchrun --nproc_per_node 8 posttrain.py --phase dpo --student <sft>/final --train_data <dpo.jsonl> --output_dir <d>

Each phase runs one epoch and writes a student checkpoint to <output_dir>/final, which SGLang serves like any other
(scripts/export.py, eval/run_lmms_eval.sh).
"""
import argparse
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from heed.data import DistillDataset, PairDataset, collate, collate_pairs
from heed.hybrid import load_student, save_student
from heed.lora import add_lora, merge_lora, set_lora
from heed.losses import dpo_loss
from train import all_sum, infinite, init_dist, lm_forward, lr_at

PHASES = {"sft": {"lr": 5e-6, "weight_decay": 0.01, "batch_tokens": 1e6, "max_len": 8192},
          "dpo": {"lr": 1e-6, "weight_decay": 0.0, "batch_tokens": 5e5, "max_len": 4096}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--student", required=True, help="student checkpoint dir")
    p.add_argument("--train_data", required=True)
    p.add_argument("--image_root", default=None, help="default: the directory of --train_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_tokens", type=float, default=None, help="default: 1M for SFT, 0.5M for DPO")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--max_visual_tokens", type=int, default=2500)
    p.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for k, v in PHASES[args.phase].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    args.min_lr_ratio = 0.0
    return args


class MasterAdamW:
    """AdamW on fp32 master weights in the CPU memory of rank 0; every GPU holds bf16 weights and receives the update.

    At these learning rates an update is far below the rounding step of a bf16 weight, so it has to accumulate in
    fp32, and fp32 weights with their Adam state do not fit on one GPU.
    """

    def __init__(self, params, rank, world, weight_decay):
        self.params, self.rank, self.world = params, rank, world
        if rank == 0:
            self.masters = [p.detach().float().cpu() for p in params]
            self.opt = torch.optim.AdamW([{"params": [m for m in self.masters if m.ndim >= 2]},
                                          {"params": [m for m in self.masters if m.ndim < 2], "weight_decay": 0.0}],
                                         betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    def step(self, lr):
        if self.rank == 0:
            for g in self.opt.param_groups:
                g["lr"] = lr
            for p, m in zip(self.params, self.masters):
                m.grad = p.grad.float().cpu()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            for p, m in zip(self.params, self.masters):
                p.data.copy_(m)
        for p in self.params:
            p.grad = None
            if self.world > 1:
                dist.broadcast(p.data, 0)


def answer_logits(model, batch):
    """Logits at the supervised next-token positions, their targets, and the row each position belongs to."""
    vis = None
    if batch["pixel_values"] is not None:
        with torch.no_grad():  # frozen vision encoder
            vis = model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
    hidden = lm_forward(model, batch, {"image": vis} if vis is not None else None)
    m = batch["loss_mask"][:, 1:]
    return model.lm_head(hidden[:, :-1][m]).float(), batch["input_ids"][:, 1:][m], m.nonzero()[:, 0]


def answer_logp(model, batch):
    """Log-probability of each row's answer."""
    logits, tgt, rows = answer_logits(model, batch)
    logp = -F.cross_entropy(logits, tgt, reduction="none")
    return torch.zeros(len(batch["ids"]), device=logp.device).index_add(0, rows, logp)


def micro_step(model, batch, args):
    if args.phase == "sft":
        logits, tgt, _ = answer_logits(model, batch)
        return F.cross_entropy(logits, tgt, reduction="sum") / max(len(tgt), 1)
    set_lora(model, False)
    with torch.no_grad():
        ref_logp = answer_logp(model, batch)
    set_lora(model, True)
    return dpo_loss(answer_logp(model, batch), ref_logp, args.dpo_beta)


def main():
    args = parse_args()
    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    log = print if rank == 0 else (lambda *a, **k: None)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.student, "heed_student.json")) as f:
        base = json.load(f)["base_model"]
    processor = AutoProcessor.from_pretrained(base)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                            device_map={"": device})
    load_student(model, args.student)
    for name, p in model.named_parameters():
        p.requires_grad_(args.phase == "sft" and ".visual." not in name)
    if args.phase == "dpo":
        add_lora(model)
    model.eval()
    model.model.language_model.train()  # no dropout but the adapters'; lets the layers use activation checkpointing
    model.lm_head.train()
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = MasterAdamW(trainable, rank, world, args.weight_decay)
    log(f"[heed] {args.phase}: {sum(p.numel() for p in trainable) / 1e9:.3f}B trainable params, world={world}")

    pairs = args.phase == "dpo"
    ds = (PairDataset if pairs else DistillDataset)(args.train_data, processor, args.image_root, args.max_len,
                                                    args.max_len, args.max_visual_tokens)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    collate_fn = partial(collate_pairs if pairs else collate, pad_id=processor.tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=args.micro_batch_size, sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=True, persistent_workers=args.num_workers > 0)
    batches = infinite(loader, sampler)

    samples, step = 0, 0
    logf = open(os.path.join(args.output_dir, "train_log.jsonl"), "a") if rank == 0 else None
    while samples < len(ds) and (args.max_steps is None or step < args.max_steps):
        lr = lr_at(samples / len(ds), args)
        t0 = time.time()
        step_tokens, loss_sum, n_micro = 0, 0.0, 0
        while step_tokens < args.batch_tokens:
            batch = next(batches)
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            ntok = batch["attention_mask"].sum()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = micro_step(model, batch, args)
            (loss * ntok).backward()
            loss_sum += loss.detach() * ntok
            n = len(batch["ids"]) // (2 if pairs else 1)
            counts = all_sum(torch.stack([ntok, torch.tensor(n, device=device)]), world)
            step_tokens += int(counts[0])
            samples += int(counts[1])
            n_micro += 1

        # token-weighted mean over every micro-batch on every rank
        for p in trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            all_sum(p.grad, world).div_(step_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step(lr)
        step += 1

        rec = {"step": step, "samples": samples, "loss": float(all_sum(loss_sum.float(), world)) / step_tokens,
               "lr": lr, "grad_norm": float(grad_norm), "micro_batches": n_micro,
               "tok_per_s": step_tokens / (time.time() - t0), "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log(json.dumps(rec))
        if logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

    if rank == 0:
        if pairs:
            merge_lora(model)
        save_student(model, os.path.join(args.output_dir, "final"), base, post_trained=True, phase=args.phase,
                     step=step)
        log("[heed] saved final")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
