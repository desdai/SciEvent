#!/bin/bash

PY_SCRIPT="baselines/LLM/segmentation_eval/segmentation_prepare.py"

# Qwen
python "$PY_SCRIPT" \
  --input-folder baselines/LLM/output/Event_Segmentation/Qwen2.5-7B-Instruct/Zero-Shot_Event_Segmentation \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_qwen

# LLaMA
python "$PY_SCRIPT" \
  --input-folder baselines/LLM/output/Event_Segmentation/Meta-Llama-3.1-8B-Instruct/Zero-Shot_Event_Segmentation \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_llama

# Distill-LLaMA
python "$PY_SCRIPT" \
  --input-folder baselines/LLM/output/Event_Segmentation/DeepSeek-R1-Distill-Llama-8B/Zero-Shot_Event_Segmentation \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_distill_llama

# GPT
python "$PY_SCRIPT" \
  --input-folder baselines/LLM/output/Event_Segmentation/gpt-4.1/Zero-Shot_Event_Segmentation \
  --output-folder SciEvent_data/LLM/Event_Segmentation/pred_gpt

# prepare the ground truth
python baselines/LLM/segmentation_eval/prepare_true.py/ \
  --in_folder SciEvent_data/to_be_annotated \
  --out_folder SciEvent_data/LLM/Event_Segmentation/true