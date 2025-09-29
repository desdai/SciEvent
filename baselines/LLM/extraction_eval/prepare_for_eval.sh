#!/bin/bash
set -euo pipefail

echo "=== Running prepare_for_eval jobs ==="

# ======================= qwen =======================

# Improve zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/Zero-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_zeroshot/pred_event_level.json

# Improve oneshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/One-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_oneshot/pred_event_level.json

# Improve 2shot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/Few-shot-2_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_2shot/pred_event_level.json

# Improve 5shot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/Few-shot-5_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_5shot/pred_event_level.json

# true-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/Zero-Shot_True_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_true_event/pred_event_level.json

# pred-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/Zero-Shot_Pred_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/Qwen2.5-7B-Instruct/eval_pred_event/pred_event_level.json

# ======================= llama =======================

# Improve zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/Zero-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_zeroshot/pred_event_level.json

# Improve oneshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/One-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_oneshot/pred_event_level.json

# Improve 2shot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/Few-shot-2_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_2shot/pred_event_level.json

# Improve 5shot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/Few-shot-5_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_5shot/pred_event_level.json

# true-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/Zero-Shot_True_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_true_event/pred_event_level.json

# pred-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/Zero-Shot_Pred_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/Meta-Llama-3.1-8B-Instruct/eval_pred_event/pred_event_level.json

# ======================= Distilled llama =======================

# Improve zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/Zero-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_zeroshot/pred_event_level.json

# Improve oneshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/One-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_oneshot/pred_event_level.json

# Improve 2shot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/Few-shot-2_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_2shot/pred_event_level.json

# Improve 5shot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/Few-shot-5_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_5shot/pred_event_level.json

# true-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/Zero-Shot_True_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_true_event/pred_event_level.json

# pred-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/Zero-Shot_Pred_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_pred_event/pred_event_level.json

# ======================= GPT-4.1 =======================

# zeroshot-GPT-4.1
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt-4.1/Zero-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_zeroshot/pred_event_level.json

# oneshot-GPT-4.1
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/One-Shot_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_oneshot/pred_event_level.json

# 2shot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/Few-shot-2_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_2shot/pred_event_level.json

# 5shot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/Few-shot-5_Event_Extraction \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_5shot/pred_event_level.json

# true-event-type-zeroshot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/Zero-Shot_True_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_true_event/pred_event_level.json

# pred-event-type-zeroshot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/Zero-Shot_Pred_Event_Type \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_pred_event/pred_event_level.json
