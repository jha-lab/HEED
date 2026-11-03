"""Inference profile of a model under SGLang on one GPU, at each context length (random text prompts):
time-to-first-token and decode speed of one streamed request (medians of --repeats), and throughput of
--num_prompts concurrent ones.

    python scripts/bench.py <HF id, or a student's export dir> --out <model.json>
    python scripts/bench.py --compare <teacher.json> <student.json>

The radix cache is off, so no prompt reuses another's prefix. cache_gb is the KV cache (plus Mamba-2 state) one
request needs at that context, from the model config; kv_pool_tokens is what SGLang could allocate on this GPU, and
contexts that do not fit in it are skipped.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heed.serving import serve_students  # noqa: E402


def cache_gb(config, context):
    tc = config.text_config
    mamba = list(getattr(config, "heed_mamba_layers", None) or [])
    kv = 2 * tc.num_key_value_heads * tc.head_dim * 2 * (tc.num_hidden_layers - len(mamba)) * context  # bf16 K and V
    h, p = tc.num_attention_heads, tc.head_dim
    conv = 3 * h * p * (getattr(config, "heed_d_conv", 4) - 1) * 2  # bf16 [x, B, C] window
    ssm = h * p * p * 4  # fp32 state
    return (kv + len(mamba) * (conv + ssm)) / 2 ** 30


def profile(args):
    serve_students(args.model)
    import sglang
    from transformers import AutoConfig

    contexts = [int(c) for c in args.contexts.split(",")]
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    engine = sglang.Engine(model_path=args.model, trust_remote_code=True, disable_radix_cache=True,
                           sampling_backend="pytorch", cuda_graph_backend_prefill="disabled",
                           mem_fraction_static=args.mem_fraction, context_length=max(contexts) + args.new_tokens + 8)
    info = engine.get_server_info()
    rng = random.Random(0)

    def prompts(n, context):  # fresh random prompts: ordinary text tokens
        return [[rng.randrange(1000, 100000) for _ in range(context)] for _ in range(n)]

    def sampling(new_tokens):
        return {"temperature": 0, "max_new_tokens": new_tokens, "ignore_eos": True}

    def stream(context, new_tokens):
        """Time to the first token, and tokens/s after it, from the token counts of the streamed chunks."""
        t0, first = time.perf_counter(), None
        chunks = engine.generate(input_ids=prompts(1, context)[0], sampling_params=sampling(new_tokens), stream=True)
        for chunk in chunks:
            last = (time.perf_counter(), chunk["meta_info"]["completion_tokens"])
            first = first or last
        return first[0] - t0, (last[1] - first[1]) / (last[0] - first[0])

    stream(512, 8)  # warm-up
    results = {"model": args.model, "kv_pool_tokens": info.get("max_total_num_tokens"), "contexts": []}
    for c in contexts:
        if c + args.new_tokens > (results["kv_pool_tokens"] or float("inf")):
            print(f"context {c}: skipped, the KV cache on this GPU holds {results['kv_pool_tokens']} tokens", flush=True)
            continue
        ttft, decode = (statistics.median(v) for v in zip(*[stream(c, args.new_tokens) for _ in range(args.repeats)]))
        t0 = time.perf_counter()
        engine.generate(input_ids=prompts(args.num_prompts, c), sampling_params=sampling(args.new_tokens))
        batch = time.perf_counter() - t0
        results["contexts"].append({
            "context": c, "ttft_s": ttft, "decode_tok_s": decode,
            "throughput_tok_s": args.num_prompts * args.new_tokens / batch, "num_prompts": args.num_prompts,
            "cache_gb": cache_gb(config, c)})
        print(json.dumps(results["contexts"][-1]), flush=True)
    engine.shutdown()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


def compare(teacher_json, student_json):
    with open(teacher_json) as f:
        teacher = json.load(f)
    with open(student_json) as f:
        student = json.load(f)
    print(f"KV pool on this GPU: {teacher['kv_pool_tokens']} -> {student['kv_pool_tokens']} tokens")
    student_at = {s["context"]: s for s in student["contexts"]}
    for t in teacher["contexts"]:
        if t["context"] not in student_at:
            continue
        s = student_at[t["context"]]
        print(f"context {t['context']:>7}: TTFT {t['ttft_s']:.2f}s -> {s['ttft_s']:.2f}s, "
              f"decode {s['decode_tok_s'] / t['decode_tok_s']:.2f}x, "
              f"throughput ({t['num_prompts']} requests) {s['throughput_tok_s'] / t['throughput_tok_s']:.2f}x, "
              f"cache per request {t['cache_gb']:.2f} -> {s['cache_gb']:.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("TEACHER_JSON", "STUDENT_JSON"))
    p.add_argument("--contexts", default="4096,32768,131072")
    p.add_argument("--new_tokens", type=int, default=128)
    p.add_argument("--num_prompts", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mem_fraction", type=float, default=0.85)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        profile(args)


if __name__ == "__main__":
    main()
