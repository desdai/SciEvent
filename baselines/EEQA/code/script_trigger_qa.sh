#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
start_time=$(date +%s)

echo "=========================================================================================="
echo "                                          my query 2                                         "
echo "=========================================================================================="

python baselines/EEQA/code/run_trigger_qa.py \
  --train_file SciEvent_data/EEQA/all_splits/train.eeqa.json  \
  --dev_file SciEvent_data/EEQA/all_splits/dev.eeqa.json  \
  --test_file SciEvent_data/EEQA/all_splits/test.eeqa.json \
  --train_batch_size 8 \
  --eval_batch_size 8  \
  --eval_per_epoch 20 \
  --num_train_epochs 12 \
  --output_dir baselines/EEQA/scievent_trigger_qa_output/full_data \
  --model_dir baselines/EEQA/scievent_trigger_qa_output/full_data/best \
  --learning_rate 4e-5 \
  --nth_query 2 \
  --warmup_proportion 0.1 \
  --do_train \
  --do_eval \
  --model bert-base-uncased \
  --eval_test

# Save the time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time: ${elapsed}s" | tee -a time_for_trig.txt