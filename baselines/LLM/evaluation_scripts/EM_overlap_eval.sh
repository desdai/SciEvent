#!/bin/bash
set -euo pipefail

echo "=== Running EM_overlap_eval jobs ==="

# Improve zeroshot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-qwen.txt

# Improve oneshot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-qwen.txt

# Improve 2shot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_2shot-qwen.txt

# Improve 5shot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_5shot-qwen.txt

# true-event-type-zeroshot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/true-event-type-zeroshot-qwen.txt

# pred-event-type-zeroshot-qwen
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/pred-event-type-zeroshot-qwen.txt

# Improve zeroshot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-llama.txt

# Improve oneshot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-llama.txt

# Improve 2shot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_2shot-llama.txt

# Improve 5shot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_5shot-llama.txt

# true-event-type-zeroshot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/true-event-type-zeroshot-llama.txt

# pred-event-type-zeroshot-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/pred-event-type-zeroshot-llama.txt

# ======================= Distilled =======================

# Improve zeroshot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-distill-llama.txt

# Improve oneshot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-distill-llama.txt

# Improve 2shot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_2shot-distill-llama.txt

# Improve 5shot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_5shot-distill-llama.txt

# true-event-type-zeroshot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/true-event-type-zeroshot-distill-llama.txt

# pred-event-type-zeroshot-distill-llama
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/pred-event-type-zeroshot-distill-llama.txt

# ======================= GPT-4.1 =======================

# zeroshot-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/zeroshot-GPT-4.1.txt

# oneshot-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/oneshot-GPT-4.1.txt

# 2shot-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/2shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/2shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/2shot-GPT-4.1.txt

# 5shot-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/5shot_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/5shot_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/5shot-GPT-4.1.txt

# true-evetype-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/true-evetype-GPT-4.1.txt

# pred-evetype-GPT-4.1
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/pred_event_level.json \
    --gold SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/gold_event_level.json \
    --out baselines/LLM/LLM_results/pred-evetype-GPT-4.1.txt

# ======================= Human Subsets =======================

# human
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/human/human_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/human.txt

# zeroshot-GPT-4.1 subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/zeroshot-GPT-4.1-subset.txt

# oneshot-GPT-4.1 subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/oneshot-GPT-4.1-subset.txt

# Improve zeroshot-distill-llama subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-distill-llama-subset.txt

# Improve oneshot-distill-llama subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-distill-llama-subset.txt

# Improve zeroshot-qwen subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-qwen-subset.txt

# Improve oneshot-qwen subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-qwen-subset.txt

# Improve zeroshot-llama subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_zeroshot-llama-subset.txt

# Improve oneshot-llama subset
python baselines/LLM/evaluation_scripts/EM_overlap_eval.py \
    --pred SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/filtered_pred_event_level.json \
    --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json \
    --out baselines/LLM/LLM_results/Improve_oneshot-llama-subset.txt
