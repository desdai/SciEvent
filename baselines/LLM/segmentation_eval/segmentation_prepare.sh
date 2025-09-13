#!/bin/bash
# run_all.sh
# Execute extraction script on multiple input/output folder pairs

PY_SCRIPT="baselines/LLM/segmentation_eval/segmentation_prepare.py"   # change to your python filename

# Qwen
python "$PY_SCRIPT" \
  --input-folder SciEvent_data/LLM/Event_Segmentation/chunked_qwen \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_qwen

# LLaMA
python "$PY_SCRIPT" \
  --input-folder SciEvent_data/LLM/Event_Segmentation/chunked_llama \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_llama

# Distill-LLaMA
python "$PY_SCRIPT" \
  --input-folder SciEvent_data/LLM/Event_Segmentation/chunked_distill_llama \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_distill_llama

# GPT
python "$PY_SCRIPT" \
  --input-folder SciEvent_data/LLM/Event_Segmentation/chunked_gpt \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_gpt
