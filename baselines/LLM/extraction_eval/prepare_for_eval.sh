#!/bin/bash
set -euo pipefail

echo "=== Running prepare_for_eval jobs ==="

# ======================= qwen =======================

# Improve zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_zeroshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_zeroshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_zeroshot/gold_event_level.json

# Improve oneshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_oneshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_oneshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_oneshot/gold_event_level.json

# Improve 2shot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_2shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_2shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_2shot/gold_event_level.json

# Improve 5shot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_5shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_5shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_5shot/gold_event_level.json

# true-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_true_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_true_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_true_event/gold_event_level.json

# pred-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/predicted_pred_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_pred_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_pred_event/gold_event_level.json

# ======================= llama =======================

# Improve zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_zeroshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_zeroshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_zeroshot/gold_event_level.json

# Improve oneshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_oneshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_oneshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_oneshot/gold_event_level.json

# Improve 2shot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_2shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_2shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_2shot/gold_event_level.json

# Improve 5shot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_5shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_5shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_5shot/gold_event_level.json

# true-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_true_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_true_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_true_event/gold_event_level.json

# pred-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/predicted_pred_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_pred_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_pred_event/gold_event_level.json

# ======================= Distilled llama =======================

# Improve zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_zeroshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_zeroshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_zeroshot/gold_event_level.json

# Improve oneshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_oneshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_oneshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_oneshot/gold_event_level.json

# Improve 2shot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_2shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_2shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_2shot/gold_event_level.json

# Improve 5shot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_5shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_5shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_5shot/gold_event_level.json

# true-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_true_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_true_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_true_event/gold_event_level.json

# pred-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/predicted_pred_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_pred_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_pred_event/gold_event_level.json

# ======================= GPT-4.1 =======================

# zeroshot-GPT-4.1
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_zeroshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_zeroshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_zeroshot/gold_event_level.json

# oneshot-GPT-4.1
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_oneshot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_oneshot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_oneshot/gold_event_level.json

# 2shot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_2shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_2shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_2shot/gold_event_level.json

# 5shot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_5shot \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_5shot/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_5shot/gold_event_level.json

# true-event-type-zeroshot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_true_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_true_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_true_event/gold_event_level.json

# pred-event-type-zeroshot-gpt
python baselines/LLM/extraction_eval/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/Event_Extraction/gpt_4.1/predicted_pred_event \
    --gold_folder SciEvent_data/LLM/Event_Extraction/true \
    --pred_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_pred_event/pred_event_level.json \
    --gold_out SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_pred_event/gold_event_level.json
