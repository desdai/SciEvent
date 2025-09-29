#!/bin/bash
# run_eval_all.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="baselines/LLM/segmentation_eval/segmentation_eval.py"

# Ensure output dir exists
mkdir -p baselines/LLM/LLM_results/Event_Segmentation

echo "distill-llama"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_distill_llama \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/distill_llama.txt

echo "llama"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_llama \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/llama.txt

echo "qwen"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_qwen \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/qwen.txt

echo "gpt"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_gpt \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/gpt.txt

echo "All done."
