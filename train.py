"""HEED Phase 1: three-stage distillation of Qwen3-VL into a 3:1 Mamba-2/attention hybrid (Sect. 4.1, App. C.3).

    Stage 1 (warm-up,   first 10% of tokens): Mamba-2 blocks train except W_{Q,K,V,O}; loss L_align
    Stage 2 (full block, next 30%):           full Mamba-2 blocks;                    loss L_align
    Stage 3 (end-to-end, last 60%):           full Mamba-2 blocks;                    loss L_KD (Eq. 1)

L_align is the condition's alignment loss: hsa (C2), rsa (C3) or heed (C4, default). kd (C1) skips Stages 1/2
and spends the whole budget on Stage 3. Only the Mamba-2 blocks ever receive gradients.

    torchrun --nproc_per_node 8 train.py --train_data <data.jsonl> --output_dir <out>
"""
import argparse
import contextlib
import json
import math
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from heed.data import DistillDataset, collate
from heed.density import DensityCache, heed_weights, patch_density
from heed.hybrid import (decoder_layers, mamba_parameters, mixers, save_student, set_mode, set_stage_trainable,
                         to_hybrid, uniform_mamba_layers)
from heed.losses import hsa_loss, kd_loss, residual_alignment_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_data", required=True)
    p.add_argument("--image_root", default=None, help="default: the directory of --train_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--condition", default="heed", choices=["kd", "hsa", "rsa", "heed"])
    p.add_argument("--mamba_layers", default=None, help="comma-separated layer ids; default: uniform 3:1 hybrid")
    p.add_argument("--total_tokens", type=float, default=2e9)
    p.add_argument("--batch_tokens", type=float, default=2e6)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--stage_fracs", default="0.1,0.3,0.6")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_frac", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--density_cache", default=None,
                   help="prefix from scripts/build_density_cache.py; default: compute density from the teacher ViT pass")
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--max_text_len", type=int, default=4096)
    p.add_argument("--max_visual_tokens", type=int, default=2500)
    p.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps")
    p.add_argument("--save_every", type=int, default=0, help="optimizer steps between checkpoints; 0 = stage ends only")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.stage_fracs = [float(x) for x in args.stage_fracs.split(",")]
    return args


def init_dist():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def stage_at(tokens, args):
    if args.condition == "kd":
        return 3
    s1, s2, _ = args.stage_fracs
    f = tokens / args.total_tokens
    return 1 if f < s1 else 2 if f < s1 + s2 else 3


def lr_at(f, args):
    """At fraction `f` of training: linear warmup, then cosine decay to `min_lr_ratio` of the peak."""
    if f < args.warmup_frac:
        return args.lr * max(f, 1e-3) / args.warmup_frac
    prog = min((f - args.warmup_frac) / (1 - args.warmup_frac), 1.0)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class ResidualTap:
    """Records the residual stream leaving each replaced layer l, i.e. r_{l+1} for the readout set S*."""

    def __init__(self, model, layer_ids):
        self.layer_ids, self.out, self.active = layer_ids, {}, False
        for i in layer_ids:
            decoder_layers(model)[i].register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, inputs, output):
            if self.active:  # not during activation-checkpoint recomputation in backward
                self.out[i] = output[0] if isinstance(output, tuple) else output
        return fn

    @contextlib.contextmanager
    def recording(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def take(self):
        res, self.out = [self.out[i] for i in self.layer_ids], {}
        return res


def infinite(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            if batch is not None:
                yield batch
        epoch += 1


def lm_forward(model, batch, mm):
    return model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       image_grid_thw=batch["image_grid_thw"], mm_token_type_ids=batch["mm_token_type_ids"],
                       mm_encoder_outputs=mm, use_cache=False).last_hidden_state


def batch_density(batch, vis, cache, merge_size):
    """Raw visual densities per sample, in sequence order (None for text-only samples).

    Cached values are used when present; samples missing from the cache use the teacher's ViT features directly.
    """
    if vis is None:
        return [None] * len(batch["ids"])
    online = None
    out, k = [], 0
    for key, n in zip(batch["ids"], batch["num_images"]):
        if n == 0:
            out.append(None)
        elif cache is not None and key in cache:
            out.append(cache.get(key).to(batch["input_ids"].device))
        else:
            if online is None:
                online = patch_density(vis.last_hidden_state, batch["image_grid_thw"], merge_size)
            out.append(torch.cat(online[k:k + n]))
        k += n
    return out


def micro_step(model, batch, stage, args, tap, cache):
    valid = batch["attention_mask"].bool()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vis = None
        if batch["pixel_values"] is not None:
            with torch.no_grad():  # one frozen-ViT pass, shared by teacher and student
                vis = model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
        mm = {"image": vis} if vis is not None else None

        if stage < 3 and args.condition == "hsa":
            set_mode(model, "hsa")
            lm_forward(model, batch, mm)
            return hsa_loss([m.hsa_pair for m in mixers(model)], valid), {}

        align = stage < 3
        set_mode(model, "teacher")
        with torch.no_grad(), (tap.recording() if align else contextlib.nullcontext()):
            t_hidden = lm_forward(model, batch, mm)
        t_res = tap.take() if align else None
        set_mode(model, "student")
        with tap.recording() if align else contextlib.nullcontext():
            s_hidden = lm_forward(model, batch, mm)

        if align:
            w = None
            if args.condition == "heed":
                rho = batch_density(batch, vis, cache, model.config.vision_config.spatial_merge_size)
                w = heed_weights(rho, batch["input_ids"] == model.config.image_token_id, valid, args.tau, args.beta)
            return residual_alignment_loss(tap.take(), t_res, valid, w), {}
        loss, kl, ce = kd_loss(model.lm_head, s_hidden, t_hidden, batch["input_ids"], batch["loss_mask"],
                               args.lambda_kl, args.lambda_ce)
        return loss, {"kl": kl, "ce": ce}


def all_sum(x, world):
    if world > 1:
        dist.all_reduce(x)
    return x


def main():
    args = parse_args()
    rank, world, local = init_dist()
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    log = print if rank == 0 else (lambda *a, **k: None)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": device})
    n_layers = model.config.text_config.num_hidden_layers
    layers = ([int(x) for x in args.mamba_layers.split(",")] if args.mamba_layers
              else uniform_mamba_layers(n_layers))
    to_hybrid(model, layers)
    model.eval()
    model.model.language_model.train()  # dropout-free; lets the decoder layers use activation checkpointing
    tap = ResidualTap(model, layers)
    cache = DensityCache(args.density_cache) if args.density_cache else None

    params = mamba_parameters(model)
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    log(f"[heed] condition={args.condition} mamba layers ({len(layers)}/{n_layers}): {layers}")
    log(f"[heed] Mamba-2 params: {sum(p.numel() for p in params) / 1e9:.3f}B, world={world}")

    ds = DistillDataset(args.train_data, processor, args.image_root, args.max_len, args.max_text_len,
                        args.max_visual_tokens)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(ds, batch_size=args.micro_batch_size, sampler=sampler, num_workers=args.num_workers,
                        collate_fn=partial(collate, pad_id=processor.tokenizer.pad_token_id), pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    batches = infinite(loader, sampler)

    tokens_seen, step, stage = 0, 0, None
    logf = open(os.path.join(args.output_dir, "train_log.jsonl"), "a") if rank == 0 else None

    def save(tag):
        if rank == 0:
            save_student(model, os.path.join(args.output_dir, tag), args.model, condition=args.condition,
                         stage=stage, step=step, tokens=tokens_seen)
            log(f"[heed] saved {tag}")

    while tokens_seen < args.total_tokens and (args.max_steps is None or step < args.max_steps):
        new_stage = stage_at(tokens_seen, args)
        if new_stage != stage:
            if stage is not None:
                save(f"stage{stage}")
            stage = new_stage
            set_stage_trainable(model, stage)
            # HSA reads per-layer mixer outputs as a side channel, so it runs without activation checkpointing
            if args.no_grad_ckpt or (args.condition == "hsa" and stage < 3):
                model.gradient_checkpointing_disable()
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            trainable = [p for p in params if p.requires_grad]
            log(f"[heed] stage {stage} at {tokens_seen:.0f} tokens: "
                f"{sum(p.numel() for p in trainable) / 1e9:.3f}B trainable params")

        lr = lr_at(tokens_seen / args.total_tokens, args)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        step_tokens, loss_sum, extra_sum, n_micro = 0, 0.0, {}, 0
        while step_tokens < args.batch_tokens:
            batch = next(batches)
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            ntok = batch["attention_mask"].sum()
            loss, extra = micro_step(model, batch, stage, args, tap, cache)
            (loss * ntok).backward()
            loss_sum += loss.detach() * ntok
            for k, v in extra.items():
                extra_sum[k] = extra_sum.get(k, 0.0) + v * ntok
            step_tokens += int(all_sum(ntok.clone(), world))
            n_micro += 1

        # token-weighted mean over every micro-batch on every rank
        for p in trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            all_sum(p.grad, world).div_(step_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += step_tokens
        step += 1

        stats = torch.stack([loss_sum, *extra_sum.values()]) if extra_sum else loss_sum[None]
        stats = (all_sum(stats.float(), world) / step_tokens).tolist()
        rec = {"step": step, "stage": stage, "tokens": tokens_seen, "loss": stats[0],
               **dict(zip(extra_sum, stats[1:])), "lr": lr, "grad_norm": float(grad_norm), "micro_batches": n_micro,
               "tok_per_s": step_tokens / (time.time() - t0),
               "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log(json.dumps(rec))
        if logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if args.save_every and step % args.save_every == 0:
            save(f"step{step}")

    save("final")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
