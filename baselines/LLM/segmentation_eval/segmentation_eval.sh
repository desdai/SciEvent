#!/bin/bash
# run_eval_all.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="baselines/LLM/segmentation_eval/segmentation_eval.py"

# Ensure output dir exists
mkdir -p baselines/LLM/LLM_results/Event_Segmentation

echo "[1/5] distill-llama"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_distill_llama \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/distill_llama.txt

echo "[2/5] llama"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_llama \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/llama.txt

echo "[3/5] qwen"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_qwen \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/qwen.txt

echo "[4/5] gpt"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/pred_gpt \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/gpt.txt

echo "[5/5] human_subset"
$PYTHON_BIN "$PY_SCRIPT" \
  --pred-folder SciEvent_data/LLM/Event_Segmentation/human_subset \
  --true-folder SciEvent_data/LLM/Event_Segmentation/true \
  --out baselines/LLM/LLM_results/Event_Segmentation/human_subset.txt

echo "All done."
