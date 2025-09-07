#!/bin/sh
# ✅ Reuse best model dir directly
export CUDA_VISIBLE_DEVICES=1
start_time=$(date +%s)

export BEST_TRIGGER_DIR=baselines/EEQA/scievent_trigger_qa_output/no_cscw/best

export ARG_QUERY_FILE=baselines/EEQA/question_templates/arg_queries.csv
export DES_QUERY_FILE=baselines/EEQA/question_templates/description_queries.csv

# comparing. old is asking naively, SCIEVENT_simple is more human way, with example is ICL
export SCIEVENT_SIMPLE_FILE=baselines/EEQA/question_templates/scievent_simple_template.csv
# export SCIEVENT_SIMPLE_FILE=./question_templates/simple_temp.csv
# export SCIEVENT_SIMPLE_FILE=./question_templates/scievent_with_example_templat.csv

export SCIEVENT_DES_FILE=baselines/EEQA/question_templates/scievent_des_template.csv

echo "**************************"
echo "        do 2 and 3"
echo "**************************"

echo "=========================================================================================="
echo "     arg_qa: using des_query + trigger verb, trigger dir = $BEST_TRIGGER_DIR"
echo "=========================================================================================="
echo "check my_example_template to choose nth_query"

python baselines/EEQA/code/run_args_qa_thresh.py \
  --train_file SciEvent_data/EEQA/ablation/no_cscw/train_without_cscw.eeqa.json \
  --dev_file SciEvent_data/EEQA/all_splits/dev.eeqa.json  \
  --test_file baselines/EEQA/scievent_trigger_qa_output/no_cscw/best/trigger_predictions.json \
  --gold_file SciEvent_data/EEQA/all_splits/test.eeqa.json \
  --train_batch_size 16 \
  --eval_batch_size 16  \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --output_dir baselines/EEQA/scievent_args_qa_thresh_output/no_cscw \
  --model_dir baselines/EEQA/scievent_args_qa_thresh_output/no_cscw/best_args \
  --nth_query 2 \
  --normal_file $ARG_QUERY_FILE \
  --des_file $DES_QUERY_FILE \
  --scievent_template $SCIEVENT_SIMPLE_FILE \
  --scievent_des_template $SCIEVENT_DES_FILE \
  --eval_per_epoch 5 \
  --max_seq_length 512 \
  --n_best_size 5 \
  --max_answer_length 20 \
  --larger_than_cls \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --eval_test

# Save the time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time: ${elapsed}s" | tee -a time_for_trig.txt