#!/bin/bash
set -euo pipefail

echo "=== Running EM_overlap_eval jobs ==="

# ======================= qwen =======================

# Improve zeroshot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_zeroshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_zeroshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/zeroshot-qwen.txt

# Improve oneshot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_oneshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_oneshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/oneshot-qwen.txt

# Improve 2shot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_2shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_2shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/2shot-qwen.txt

# Improve 5shot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_5shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_5shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/5shot-qwen.txt

# true-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_true_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_true_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/true-event-type-zeroshot-qwen.txt

# pred-event-type-zeroshot-qwen
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_pred_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/qwen_2.5_7b/eval_pred_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/pred-event-type-zeroshot-qwen.txt

# ======================= llama =======================

# Improve zeroshot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_zeroshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_zeroshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/zeroshot-llama.txt

# Improve oneshot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_oneshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_oneshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/oneshot-llama.txt

# Improve 2shot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_2shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_2shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/2shot-llama.txt

# Improve 5shot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_5shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_5shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/5shot-llama.txt

# true-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_true_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_true_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/true-event-type-zeroshot-llama.txt

# pred-event-type-zeroshot-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_pred_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/llama_3.1_8b/eval_pred_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/pred-event-type-zeroshot-llama.txt

# ======================= Distilled llama =======================

# Improve zeroshot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_zeroshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_zeroshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/zeroshot-distill-llama.txt

# Improve oneshot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_oneshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_oneshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/oneshot-distill-llama.txt

# Improve 2shot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_2shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_2shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/2shot-distill-llama.txt

# Improve 5shot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_5shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_5shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/5shot-distill-llama.txt

# true-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_true_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_true_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/true-event-type-zeroshot-distill-llama.txt

# pred-event-type-zeroshot-distill-llama
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_pred_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/DeepSeek-R1-Distill-Llama-8B/eval_pred_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/pred-event-type-zeroshot-distill-llama.txt

# ======================= GPT-4.1 =======================

# zeroshot-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_zeroshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_zeroshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/zeroshot-GPT-4.1.txt

# oneshot-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_oneshot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_oneshot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/oneshot-GPT-4.1.txt

# 2shot-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_2shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_2shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/2shot-GPT-4.1.txt

# 5shot-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_5shot/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_5shot/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/5shot-GPT-4.1.txt

# true-evetype-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_true_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_true_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/true-evetype-GPT-4.1.txt

# pred-evetype-GPT-4.1
python baselines/LLM/extraction_eval/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_pred_event/pred_event_level.json \
    --gold SciEvent_data/LLM/Event_Extraction/gpt_4.1/eval_pred_event/gold_event_level.json \
    --out baselines/LLM/LLM_results/Event_Extraction/pred-evetype-GPT-4.1.txt