"""One-time density cache (Alg. 1): frozen-ViT forward per sample -> rho(p) for visual tokens, 4-bit, 64-byte aligned.

    python scripts/build_density_cache.py --train_data <data.jsonl> --out <prefix>
Training then reads it with `train.py --density_cache <prefix>`. Text positions are derived at runtime.
"""
import argparse
import os
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heed.data import DistillDataset  # noqa: E402
from heed.density import DensityCacheWriter, patch_density  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_data", required=True)
    p.add_argument("--image_root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max_visual_tokens", type=int, default=2500)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    visual = model.model.visual.eval()
    ds = DistillDataset(args.train_data, processor, args.image_root, max_visual_tokens=args.max_visual_tokens)
    writer = DensityCacheWriter(args.out)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    for i in range(n):
        s = ds[i]
        if s is None or s["pixel_values"] is None:
            continue
        grid = s["image_grid_thw"].cuda()
        feats = visual(s["pixel_values"].cuda().to(visual.dtype), grid_thw=grid, return_dict=True).last_hidden_state
        writer.add(s["id"], torch.cat(patch_density(feats, grid, visual.spatial_merge_size)))
        if i % 1000 == 0:
            print(f"{i}/{n}", flush=True)
    writer.close()
    print(f"wrote {len(writer.index)} samples to {args.out}.bin")


if __name__ == "__main__":
    main()
