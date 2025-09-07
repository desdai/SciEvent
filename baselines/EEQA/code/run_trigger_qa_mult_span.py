"""Run QA model to detect triggers."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
from collections import Counter
import copy
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTriggerClassification, BertLSTMForTriggerClassification
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering

from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

candidate_queries = [
['what', 'is', 'the', 'trigger', 'in', 'the', 'event', '?'], # 0 what is the trigger in the event?
['what', 'happened', 'in', 'the', 'event', '?'], # 1 what happened in the event?
['trigger'], # 2 trigger
['t'], # 3 t
['action'], # 4 action
['verb'], # 5 verb
['null'], # 6 null
]


class trigger_category_vocab(object):
    """docstring for trigger_category_vocab"""
    def __init__(self):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.counter = Counter()
        self.max_sent_length = 0

    def create_vocab(self, files_list):
        self.category_to_index["None"] = 0
        self.index_to_category[0] = "None"
        for file in files_list:
            with open(file) as f:
                for line in f:
                    example = json.loads(line)
                    events, sentence = example["event"], example["sentence"] 
                    if len(sentence) > self.max_sent_length: self.max_sent_length = len(sentence)
                    for event in events:
                        event_type = event[0][1]
                        self.counter[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.category_to_index)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type

        # add [CLS]
        self.max_sent_length += 12
                    

class InputFeatures(object):
    def __init__(self,
                 sentence_id,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 in_sentence,
                 start_position,
                 end_position,
                 offset_mapping):
        self.sentence_id = sentence_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.in_sentence = in_sentence
        self.start_position = start_position
        self.end_position = end_position
        self.offset_mapping = offset_mapping


def read_ace_examples(nth_query, input_file, tokenizer, category_vocab, is_training):
    """Read an ACE json file, transform to features (for QA model)"""
    features = []
    examples = []
    sentence_id = 0

    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            sentence, events, s_start = example["sentence"], example["event"], example["s_start"]

            tokens = []
            segment_ids = []
            in_sentence = []

            # [CLS]
            tokens.append("[CLS]")
            segment_ids.append(0)
            in_sentence.append(0)

            # Query
            query = candidate_queries[nth_query]
            for token in query:
                sub_tokens = tokenizer.tokenize(token)
                tokens.append(sub_tokens[0])  # use first sub_token
                segment_ids.append(0)
                in_sentence.append(0)

            # [SEP]
            tokens.append("[SEP]")
            segment_ids.append(0)
            in_sentence.append(0)

            # Sentence tokens
            offset_mapping = []  # map from sentence word idx -> token idx
            for i, token in enumerate(sentence):
                sub_tokens = tokenizer.tokenize(token)
                offset_mapping.append(len(tokens))  # record where this word starts in tokenized seq
                tokens.append(sub_tokens[0])  # assume basic tokenizer (one subword per word)
                segment_ids.append(1)
                in_sentence.append(1)

            # [SEP]
            tokens.append("[SEP]")
            segment_ids.append(1)
            in_sentence.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < category_vocab.max_sent_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                in_sentence.append(0)

            assert len(input_ids) == category_vocab.max_sent_length
            assert len(segment_ids) == category_vocab.max_sent_length
            assert len(in_sentence) == category_vocab.max_sent_length
            assert len(input_mask) == category_vocab.max_sent_length

            # Find trigger start and end positions
            start_position = 0
            end_position = 0
            if len(events) > 0:
                trigger = events[0][0]  # take the first event's trigger
                trigger_start_word = trigger[0] - s_start
                trigger_end_word = trigger[1] - 1 - s_start  # inclusive

                if 0 <= trigger_start_word < len(offset_mapping) and 0 <= trigger_end_word < len(offset_mapping):
                    start_position = offset_mapping[trigger_start_word]
                    end_position = offset_mapping[trigger_end_word]

            features.append(
                InputFeatures(
                    sentence_id=sentence_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    in_sentence=in_sentence,
                    start_position=start_position,
                    end_position=end_position,
                    offset_mapping=offset_mapping  # <- add this to InputFeatures
                )
            )

            examples.append(example)
            sentence_id += 1

    return examples, features



def evaluate(args, eval_examples, eval_features, category_vocab, model, device, eval_dataloader, pred_only=False):
    model.eval()
    pred_triggers = dict()

    for idx, (sentence_id, input_ids, segment_ids, in_sentence, input_mask, _, _) in enumerate(eval_dataloader):
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))

        sentence_id = sentence_id.tolist()
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            start_logits, end_logits = model(
                input_ids, token_type_ids=segment_ids, attention_mask=input_mask
            )

        for i, in_sent in enumerate(in_sentence):
            start_idx = torch.argmax(start_logits[i]).item()
            end_idx = torch.argmax(end_logits[i]).item()

            # Map back to word offset using offset_mapping
            offset_map = eval_features[i].offset_mapping

            word_start = None
            for word_idx, token_pos in enumerate(offset_map):
                if token_pos >= start_idx:
                    word_start = word_idx
                    break

            if word_start is not None and start_idx <= end_idx:
                pred_triggers[sentence_id[i]] = [[word_start, "PRED"]]
            else:
                pred_triggers[sentence_id[i]] = []

    # Build gold triggers
    gold_triggers = []
    for eval_example in eval_examples:
        events = eval_example["event"]
        s_start = eval_example["s_start"]
        gold_sentence_triggers = []
        for event in events:
            trigger_start, trigger_end, trigger_type = event[0]
            offset_new = trigger_start - s_start
            gold_sentence_triggers.append([offset_new, trigger_type])
        gold_triggers.append(gold_sentence_triggers)

    # [Optional] Log some mismatches
    for sent_id in pred_triggers:
        logger.debug(f"Sentence {sent_id} - PRED: {pred_triggers[sent_id]} | GOLD: {gold_triggers[sent_id]}")

    # Trigger Classification (C)
    gold_trigger_n, pred_trigger_n, true_positive_n = 0, 0, 0
    for sentence_id in pred_triggers:
        gold_sentence_triggers = gold_triggers[sentence_id]
        pred_sentence_triggers = pred_triggers[sentence_id]
        gold_trigger_n += len(gold_sentence_triggers)
        pred_trigger_n += len(pred_sentence_triggers)

        for pred in pred_sentence_triggers:
            if pred in gold_sentence_triggers:
                true_positive_n += 1

    prec_c = 100.0 * true_positive_n / pred_trigger_n if pred_trigger_n else 0
    recall_c = 100.0 * true_positive_n / gold_trigger_n if gold_trigger_n else 0
    f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if (prec_c + recall_c) else 0

    # Trigger Identification (I)
    gold_triggers_offset = [[x[0] for x in g] for g in gold_triggers]
    pred_triggers_offset = [[x[0] for x in pred_triggers.get(i, [])] for i in range(len(gold_triggers))]

    gold_trigger_n, pred_trigger_n, true_positive_n = 0, 0, 0
    for sentence_id, _ in enumerate(pred_triggers_offset):
        gold_sentence_triggers = gold_triggers_offset[sentence_id]
        pred_sentence_triggers = pred_triggers_offset[sentence_id]
        gold_trigger_n += len(gold_sentence_triggers)
        pred_trigger_n += len(pred_sentence_triggers)

        for pred in pred_sentence_triggers:
            if pred in gold_sentence_triggers:
                true_positive_n += 1

    prec_i = 100.0 * true_positive_n / pred_trigger_n if pred_trigger_n else 0
    recall_i = 100.0 * true_positive_n / gold_trigger_n if gold_trigger_n else 0
    f1_i = 2 * prec_i * recall_i / (prec_i + recall_i) if (prec_i + recall_i) else 0

    result = collections.OrderedDict([
        ('prec_c', prec_c), ('recall_c', recall_c), ('f1_c', f1_c),
        ('prec_i', prec_i), ('recall_i', recall_i), ('f1_i', f1_i)
    ])

    # Attach predictions back to examples
    preds = copy.deepcopy(eval_examples)
    for sentence_id, pred in enumerate(preds):
        s_start = pred['s_start']
        pred['event'] = []
        pred_sentence_triggers = pred_triggers.get(sentence_id, [])
        for trigger in pred_sentence_triggers:
            offset = s_start + trigger[0]
            category = trigger[1]
            pred['event'].append([[offset, category]])

    return result, preds




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    category_vocab = trigger_category_vocab()
    category_vocab.create_vocab([args.train_file, args.dev_file, args.test_file])

    # --------- prepare dev/test ----------
    if args.do_train or (not args.eval_test):
        eval_examples, eval_features = read_ace_examples(args.nth_query, args.dev_file, tokenizer, category_vocab, is_training=False)
        if args.add_lstm:
            eval_features = sorted(eval_features, key=lambda f: np.sum(f.input_mask), reverse=True)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_sentence_id = torch.tensor([f.sentence_id for f in eval_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_in_sentence = torch.tensor([f.in_sentence for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_sentence_id, all_input_ids, all_segment_ids, all_in_sentence, all_input_mask, all_start_positions, all_end_positions)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    # --------- prepare train ----------
    if args.do_train:
        _, train_features = read_ace_examples(args.nth_query, args.train_file, tokenizer, category_vocab, is_training=True)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        if args.add_lstm:
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask), reverse=True)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]

        for lr in lrs:
            model = BertForQuestionAnswering.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = model.to(device)

            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()

            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, segment_ids, input_mask, start_positions, end_positions = batch

                    loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                 start_positions=start_positions, end_positions=end_positions)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if step % 10 == 0:
                        logger.info(f"Step {step} - Loss: {loss.item():.4f}")

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0 or step == 0 or (step + 1) == len(train_batches):
                        save_model = False
                        if args.do_eval:
                            result, _ = evaluate(args, eval_examples, eval_features, category_vocab, model, device, eval_dataloader)

                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                    epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, p_i: %.2f, r_i: %.2f, f1_i: %.2f" %
                                            (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"], result["f1_c"], result["prec_i"], result["recall_i"], result["f1_i"]))
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            subdir = os.path.join(args.output_dir, "epoch{epoch}-step{step}".format(epoch=epoch, step=step))
                            if not os.path.exists(subdir):
                                os.makedirs(subdir)
                            output_model_file = os.path.join(subdir, WEIGHTS_NAME)
                            output_config_file = os.path.join(subdir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(subdir)
                            if best_result:
                                with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                    for key in best_result:
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

            del model

    if args.do_eval:
        if args.eval_test:
            eval_examples, eval_features = read_ace_examples(args.nth_query, args.test_file, tokenizer, category_vocab, is_training=False)
            if args.add_lstm:
                eval_features = sorted(eval_features, key=lambda f: np.sum(f.input_mask), reverse=True)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)

            all_sentence_id = torch.tensor([f.sentence_id for f in eval_features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_in_sentence = torch.tensor([f.in_sentence for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_sentence_id, all_input_ids, all_segment_ids, all_in_sentence, all_input_mask, all_start_positions, all_end_positions)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        model = BertForQuestionAnswering.from_pretrained(args.model_dir)
        if args.fp16:
            model.half()
        model.to(device)

        result, preds = evaluate(args, eval_examples, category_vocab, model, device, eval_dataloader)

        with open(os.path.join(args.model_dir, "test_results.txt"), "w") as writer:
            for key in result:
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(os.path.join(args.model_dir, "trigger_predictions.json"), "w") as writer:
            for line in preds:
                writer.write(json.dumps(line, default=int) + "\n")


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--model_dir", default="trigger_qa_output/epoch0-step0", type=str, required=True, help="eval/test model")
        parser.add_argument("--train_file", default=None, type=str)
        parser.add_argument("--dev_file", default=None, type=str)
        parser.add_argument("--test_file", default=None, type=str)
        parser.add_argument("--eval_per_epoch", default=10, type=int,
                            help="How many times it evaluates on dev set per epoch")
        parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
        parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
        parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
        parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--eval_metric", default='f1_c', type=str)
        parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
        parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                                 "of training.")
        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--loss_scale', type=float, default=0,
                            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                 "0 (default value): dynamic loss scaling.\n"
                                 "Positive power of 2: static loss scaling value.\n")
        parser.add_argument("--add_lstm", action='store_true', help="Whether to add LSTM on top of BERT.")
        parser.add_argument("--lstm_lr", default=None, type=float, help="The initial learning rate for lstm Adam.")        
        parser.add_argument("--nth_query", default=0, type=int, help="use n-th candidate query")
        args = parser.parse_args()

        main(args)
