#!/bin/bash
set -euo pipefail

echo "=== Running prepare_for_eval jobs ==="

# """improved prompt and checked data:"""

# Improve zeroshot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/pred_qwen_improve_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/pred_qwen_improve_oneshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/pred_qwen_improve_2shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/gold_event_level.json

# Improve 5shot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/pred_qwen_improve_5shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/true_evetype_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/gold_event_level.json

# pred-event-type-zeroshot-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/qwen_2.5_7b/pred_evetype_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/gold_event_level.json

# Improve zeroshot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_oneshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_2shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/gold_event_level.json

# Improve 5shot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_5shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_true \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/gold_event_level.json

# pred-event-type-zeroshot-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/llama_3_8b/pred_llama_improve_pred \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/gold_event_level.json


# """Distilled ones follow:"""
# Improve zeroshot-distill-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/pred_distill_qwen_improve_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-distill-qwen
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/pred_distill_qwen_improve_oneshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/gold_event_level.json

# Improve zeroshot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_oneshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_2shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/gold_event_level.json

# Improve 5shot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_5shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_true \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/gold_event_level.json

# pred-event-type-zeroshot-distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/pred_distill_llama_improve_pred \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/gold_event_level.json

# old distill-llama
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/oneshot_merge_modifier_old \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/gold_event_level.json

# """GPT-4.1""" 
# zeroshot-GPT-4.1
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/pred_gpt_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/gold_event_level.json

# oneshot-GPT-4.1
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/pred_gpt_oneshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/gold_event_level.json

# 2shot-gpt
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/pred_gpt_2shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/2shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/2shot_eval/gold_event_level.json

# 5shot-gpt
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/pred_gpt_5shot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/5shot_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/5shot_eval/gold_event_level.json

# true-event-type-zeroshot-gpt
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/true_event_type_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/gold_event_level.json

# pred-event-type-zeroshot-gpt
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/gpt_4.1/pred_event_type_zeroshot \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/gold_event_level.json

# """human""" 
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/human/human_raw \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/human/human_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/human/human_eval/gold_event_level.json

# """rebuttal_final_round annot"""
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/human/human_final_round \
    --gold_folder SciEvent_data/LLM/data/true \
    --pred_out SciEvent_data/LLM/data/human/human_eval_finalround/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/human/human_eval_finalround/gold_event_level.json

# """subset"""
python baselines/LLM/evaluation_code/prepare_for_eval.py \
    --pred_folder SciEvent_data/LLM/data/human/human_subset \
    --gold_folder SciEvent_data/LLM/data/true_subset \
    --pred_out SciEvent_data/LLM/data/human/human_eval/pred_event_level.json \
    --gold_out SciEvent_data/LLM/data/human/human_eval/gold_event_level.json
