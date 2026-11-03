#!/bin/bash
# Benchmark a Qwen3-VL checkpoint with lmms-eval on SGLang: greedy decoding, fixed prompt templates, per-sample logs
# (App. C). Runs in the serving environment (sglang, lmms-eval; see README).
#
#   bash eval/run_lmms_eval.sh <HF id, or a student dir> <comma-separated tasks | "all"> <output dir> [extra args]
#
# A student dir (heed_student.json) is served from its export, <dir>/export, written by scripts/export.py on first use.
# Images are capped at 2,500 visual tokens (32x32 px each), as in training. "all" is the 10-benchmark suite of
# Tab. 1. Extra args are passed through, e.g. --limit 100. BATCH (default 16) requests go to SGLang at a time.
set -euo pipefail

MODEL=${1:-Qwen/Qwen3-VL-8B-Instruct}
TASKS=${2:-chartqa}
OUT=${3:-outputs/eval}
shift $(( $# < 3 ? $# : 3 ))
REPO=$(cd "$(dirname "$0")/.." && pwd)

if [[ "$TASKS" == all ]]; then
  # fine-grained perception (6) + reasoning (4)
  TASKS=ocrbench_v2,docvqa_val,chartqa,textvqa_val,infovqa_val,ai2d,mmmu_pro,mmstar,mmbench_en_dev,mathvista_testmini
fi

if [[ -f "$MODEL/heed_student.json" ]]; then
  [[ -f "$MODEL/export/configuration_heed.py" ]] || python "$REPO/scripts/export.py" "$MODEL"
  MODEL=$MODEL/export
fi
if [[ -f "$MODEL/configuration_heed.py" ]]; then
  export SGLANG_EXTERNAL_MODEL_PACKAGE=heed.serving
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
fi

python -m lmms_eval \
  --model sglang \
  --model_args "model=${MODEL},gpu_memory_utilization=0.85,sampling_backend=pytorch,cuda_graph_backend_prefill=disabled,mm_process_config={\"image\":{\"max_pixels\":$((2500 * 32 * 32))}}" \
  --tasks "$TASKS" \
  --batch_size "${BATCH:-16}" \
  --gen_kwargs temperature=0 \
  --log_samples \
  --output_path "$OUT" \
  "$@"
