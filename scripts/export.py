"""Export a HEED student for SGLang as <student dir>/export: a Qwen3-VL checkpoint dir whose config lists the
Mamba-2 layers (heed/configuration_heed.py) and whose weights are the base model's plus the student's blocks, or
after SFT or DPO, which train all the rest, the base model's vision encoder plus the student.

    python scripts/export.py <student dir>               # written by heed.hybrid.save_student
    python scripts/export.py <new dir> --wt_init         # WT-initialised 3:1 student, e.g. for profiling

The base model's other files are symlinked. SGLang serves the result with SGLANG_EXTERNAL_MODEL_PACKAGE=heed.serving,
this repo on PYTHONPATH and --trust-remote-code, as eval/run_lmms_eval.sh and scripts/bench.py do. The export runs in
either environment; --wt_init needs the training one.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def wt_init_student(path, base_model, seed=0):
    from transformers import Qwen3VLForConditionalGeneration

    from heed.hybrid import save_student, to_hybrid, uniform_mamba_layers
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype=torch.bfloat16)
    to_hybrid(model, uniform_mamba_layers(model.config.text_config.num_hidden_layers))
    save_student(model, path, base_model, init="wt", seed=seed)


def export(student):
    with open(os.path.join(student, "heed_student.json")) as f:
        meta = json.load(f)
    base = meta["base_model"]
    base = base if os.path.isdir(base) else snapshot_download(base)
    state = torch.load(os.path.join(student, "student.pt"), map_location="cpu")
    if os.path.exists(os.path.join(base, "model.safetensors.index.json")):
        with open(os.path.join(base, "model.safetensors.index.json")) as f:
            index = json.load(f)
    else:  # a single-file checkpoint
        with safe_open(os.path.join(base, "model.safetensors"), "pt") as f:
            index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
    skip = {"config.json", "model.safetensors.index.json"}
    if state.keys() & index["weight_map"].keys():  # SGLang reads every shard of a checkpoint, stale tensors included
        skip |= set(index["weight_map"].values())
        for shard in set(index["weight_map"].values()):
            with safe_open(os.path.join(base, shard), "pt") as f:
                state.update({k: f.get_tensor(k) for k in f.keys() if ".visual." in k})
        index["weight_map"] = {}
    index["weight_map"].update({k: "heed_student.safetensors" for k in state})  # base attention of the blocks: unused

    out = os.path.join(student, "export")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(base):
        dst = os.path.join(out, name)
        if name not in skip and not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(base, name)), dst)
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "heed_student.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    d_conv = next(v for k, v in state.items() if k.endswith("conv1d.weight")).shape[-1]
    config.update(auto_map={"AutoConfig": "configuration_heed.HeedQwen3VLConfig"},
                  heed_mamba_layers=meta["mamba_layers"], heed_d_conv=d_conv)
    config["text_config"]["mamba_chunk_size"] = 256
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.join(ROOT, "heed", "configuration_heed.py"), out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("student")
    p.add_argument("--wt_init", action="store_true", help="first write a WT-initialised student to the dir")
    p.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = p.parse_args()
    if args.wt_init and not os.path.exists(os.path.join(args.student, "heed_student.json")):
        wt_init_student(args.student, args.base_model)
    print(export(args.student))


if __name__ == "__main__":
    main()
