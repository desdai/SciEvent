import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import GenerativeModel
from data import GenDataset, EEDataset
from utils import compute_f1
from argparse import ArgumentParser, Namespace
import ipdb
from collections import defaultdict
from rouge_score import rouge_scorer
from typing import List, Tuple, Callable, Any

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--e2e_config', required=True)
parser.add_argument('-e', '--e2e_model', required=True)
parser.add_argument('--no_dev', action='store_true', default=False)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--write_file', type=str)
args = parser.parse_args()
with open(args.e2e_config) as fp:
    config = json.load(fp)
config = Namespace(**config)

if config.dataset == "ace05e" or config.dataset == "ace05ep":
    from template_generate_ace import eve_template_generator
    template_file = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import eve_template_generator
    template_file = "template_generate_ere"

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

def spans_overlap(pred_span, gold_span):
    return max(pred_span[0], gold_span[0]) < min(pred_span[1], gold_span[1])

def scirex_overlap(pred_span, gold_span):
    inter_start = max(pred_span[0], gold_span[0])
    inter_end = min(pred_span[1], gold_span[1])
    if inter_start >= inter_end:
        return False
    intersection = inter_end - inter_start
    pred_len = pred_span[1] - pred_span[0]
    return intersection / pred_len > 0.5

def iou_overlap(pred_span, gold_span):
    inter_start = max(pred_span[0], gold_span[0])
    inter_end = min(pred_span[1], gold_span[1])
    if inter_start >= inter_end:
        return False
    intersection = inter_end - inter_start
    union = max(pred_span[1], gold_span[1]) - min(pred_span[0], gold_span[0])
    return intersection / union > 0.5

""" test_pred_roles is a list of tuples:
(
  (trigger_start_idx, trigger_end_idx, event_type),  # trigger info
  (arg_start_idx, arg_end_idx, role_type)            # argument span + role
)
"""

"""Role wise func:"""
def compute_rolewise_f1_dual(gold_roles_all, pred_roles_all, overlap_fn=None):
    from collections import defaultdict
    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    used_pred_global_C = set()
    
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for idx, (gold_roles, pred_roles) in enumerate(zip(gold_roles_all, pred_roles_all)):
        gold_by_role = defaultdict(list)
        pred_by_role = defaultdict(list)

        for (tri_s, tri_e, etype), (arg_s, arg_e, role) in gold_roles:
            if role not in exclude_roles:
                gold_by_role[role].append((arg_s, arg_e, role, etype))

        for (tri_s, tri_e, etype), (arg_s, arg_e, role) in pred_roles:
            if role not in exclude_roles:
                pred_by_role[role].append((arg_s, arg_e, role, etype))

        for role in set(gold_by_role.keys()).union(pred_by_role.keys()):
            g_spans = gold_by_role.get(role, [])
            p_spans = pred_by_role.get(role, [])
            stats_C[role]["gold_total"] += len(g_spans)
            stats_C[role]["pred_total"] += len(p_spans)
            matched = set()
            for g in g_spans:
                for i, p in enumerate(p_spans):
                    global_key = (idx, role, "C", i)
                    if i not in matched and global_key not in used_pred_global_C and \
                       p[2] == g[2] and p[3] == g[3] and \
                       (overlap_fn(p[:2], g[:2]) if overlap_fn else p[:2] == g[:2]):
                        stats_C[role]["matched"] += 1
                        matched.add(i)
                        used_pred_global_C.add(global_key)
                        break

    return stats_C

def print_rolewise_stats(name, stats_C):
    print(f"\n[ROLE-WISE ARGUMENT CLASSIFICATION - {name}] ------------------------------------------")
    print(f"{'Role':20s} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    print("-" * 50)
    for role in sorted(stats_C.keys()):
        m = stats_C[role]["matched"]
        p = stats_C[role]["pred_total"]
        g = stats_C[role]["gold_total"]
        prec = m / p if p > 0 else 0
        rec = m / g if g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{role:20s} | {prec*100:6.2f} | {rec*100:6.2f} | {f1*100:6.2f}")
    print("-" * 50)


# Event type wise func:
def compute_eventtype_f1_extended(gold_triggers_all, pred_triggers_all, gold_roles_all, pred_roles_all, overlap_fn=None):
    from collections import defaultdict

    stats = {
        "Arg_I": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
        "Arg_C": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
    }

    used_pred_global = {
        "Arg_I": set(),
        "Arg_C": set(),
    }

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for idx, (gold_roles, pred_roles) in enumerate(zip(gold_roles_all, pred_roles_all)):

        # --- ARGUMENT MATCHING (grouped by event type) ---
        gold_by_type_I = defaultdict(list)
        pred_by_type_I = defaultdict(list)
        gold_by_type_C = defaultdict(list)
        pred_by_type_C = defaultdict(list)

        for (tri_s, tri_e, etype), (arg_s, arg_e, role) in gold_roles:
            if role not in exclude_roles:
                gold_by_type_I[etype].append((arg_s, arg_e))
                gold_by_type_C[etype].append((arg_s, arg_e, role, etype))

        for (tri_s, tri_e, etype), (arg_s, arg_e, role) in pred_roles:
            if role not in exclude_roles:
                pred_by_type_I[etype].append((arg_s, arg_e))
                pred_by_type_C[etype].append((arg_s, arg_e, role, etype))

        # Argument I
        for etype in set(gold_by_type_I.keys()).union(pred_by_type_I.keys()):
            g_spans = gold_by_type_I.get(etype, [])
            p_spans = pred_by_type_I.get(etype, [])
            stats["Arg_I"][etype]["gold_total"] += len(g_spans)
            stats["Arg_I"][etype]["pred_total"] += len(p_spans)

            for i_g, g in enumerate(g_spans):
                for i_p, p in enumerate(p_spans):
                    key = (idx, etype, i_p)
                    if key in used_pred_global["Arg_I"]:
                        continue
                    if overlap_fn(p, g) if overlap_fn else p == g:
                        stats["Arg_I"][etype]["matched"] += 1
                        used_pred_global["Arg_I"].add(key)
                        break

        # Argument C
        for etype in set(gold_by_type_C.keys()).union(pred_by_type_C.keys()):
            g_spans = gold_by_type_C.get(etype, [])
            p_spans = pred_by_type_C.get(etype, [])
            stats["Arg_C"][etype]["gold_total"] += len(g_spans)
            stats["Arg_C"][etype]["pred_total"] += len(p_spans)

            for i_g, g in enumerate(g_spans):
                for i_p, p in enumerate(p_spans):
                    key = (idx, etype, i_p)
                    if key in used_pred_global["Arg_C"]:
                        continue
                    if p[2] == g[2] and p[3] == g[3] and (overlap_fn(p[:2], g[:2]) if overlap_fn else p[:2] == g[:2]):
                        stats["Arg_C"][etype]["matched"] += 1
                        used_pred_global["Arg_C"].add(key)
                        break

    return stats

def print_eventtype_stats_extended(name, stats_dict):
    print(f"\n[EVENT-TYPE ARGUMENT MATCHING - {name}] -----------------------------------------------------------")
    print(f"{'EventType':25s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>8s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>8s}")
    print("-" * 80)

    types = set()
    for task in stats_dict:
        types.update(stats_dict[task].keys())

    for event_type in sorted(types):
        fields = []
        for task in ["Arg_I", "Arg_C"]:
            stat = stats_dict[task].get(event_type, {"matched": 0, "pred_total": 0, "gold_total": 0})
            m, p, g = stat["matched"], stat["pred_total"], stat["gold_total"]
            prec = m / p if p > 0 else 0
            rec = m / g if g > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fields.append((prec * 100, rec * 100, f1 * 100))

        print(f"{event_type:25s} | "
              f"{fields[0][0]:7.2f} | {fields[0][1]:7.2f} | {fields[0][2]:8.2f} | "
              f"{fields[1][0]:7.2f} | {fields[1][1]:7.2f} | {fields[1][2]:8.2f}")
    print("-" * 80)


# domain wise:
def compute_domain_f1_extended(gold_triggers_all, pred_triggers_all, gold_roles_all, pred_roles_all, wnd_ids, overlap_fn=None):
    from collections import defaultdict

    def greedy_match(pred_list, gold_list, match_fn):
        matched_pred = set()
        matched_gold = set()
        for i_p, p in enumerate(pred_list):
            for i_g, g in enumerate(gold_list):
                if i_g in matched_gold or i_p in matched_pred:
                    continue
                if match_fn(p, g):
                    matched_pred.add(i_p)
                    matched_gold.add(i_g)
                    break
        return len(matched_pred)

    stats = {
        "Arg_I": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
        "Arg_C": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
    }

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for gold_trigs, pred_trigs, gold_roles, pred_roles, wnd_id in zip(
        gold_triggers_all, pred_triggers_all, gold_roles_all, pred_roles_all, wnd_ids):

        domain = wnd_id.strip().split('_')[0]

        # ---- Argument I (event-type sensitive) ----
        gold_args = [(arg[1][0], arg[1][1], arg[0][2]) for arg in gold_roles if arg[1][2] not in exclude_roles]
        pred_args = [(arg[1][0], arg[1][1], arg[0][2]) for arg in pred_roles if arg[1][2] not in exclude_roles]

        stats["Arg_I"][domain]["gold_total"] += len(gold_args)
        stats["Arg_I"][domain]["pred_total"] += len(pred_args)
        stats["Arg_I"][domain]["matched"] += greedy_match(
            pred_args, gold_args,
            match_fn=lambda p, g: (p[2] == g[2]) and (overlap_fn((p[0], p[1]), (g[0], g[1])) if overlap_fn else (p[0], p[1]) == (g[0], g[1]))
        )

        # ---- Argument C ----
        gold_args_c = [(arg[1][0], arg[1][1], arg[1][2], arg[0][2]) for arg in gold_roles if arg[1][2] not in exclude_roles]
        pred_args_c = [(arg[1][0], arg[1][1], arg[1][2], arg[0][2]) for arg in pred_roles if arg[1][2] not in exclude_roles]

        stats["Arg_C"][domain]["gold_total"] += len(gold_args_c)
        stats["Arg_C"][domain]["pred_total"] += len(pred_args_c)
        stats["Arg_C"][domain]["matched"] += greedy_match(
            pred_args_c, gold_args_c,
            match_fn=lambda p, g: (p[2] == g[2]) and (p[3] == g[3]) and (overlap_fn((p[0], p[1]), (g[0], g[1])) if overlap_fn else (p[0], p[1]) == (g[0], g[1]))
        )

    return stats

def print_domain_stats_extended(name, stats_dict):
    print(f"\n[DOMAIN-WISE ARGUMENT MATCHING - {name}] -----------------------------------------------------------")
    print(f"{'Domain':10s} | {'ArgI-P':>8s} | {'ArgI-R':>8s} | {'ArgI-F1':>8s} | {'ArgC-P':>8s} | {'ArgC-R':>8s} | {'ArgC-F1':>8s}")
    print("-" * 80)

    domains = set()
    for task in stats_dict:
        domains.update(stats_dict[task].keys())

    for domain in sorted(domains):
        fields = []
        for task in ["Arg_I", "Arg_C"]:
            stat = stats_dict[task].get(domain, {"matched": 0, "pred_total": 0, "gold_total": 0})
            m, p, g = stat["matched"], stat["pred_total"], stat["gold_total"]
            prec = m / p if p > 0 else 0
            rec = m / g if g > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fields.append((prec * 100, rec * 100, f1 * 100))

        print(f"{domain:10s} | "
              f"{fields[0][0]:8.2f} | {fields[0][1]:8.2f} | {fields[0][2]:8.2f} | "
              f"{fields[1][0]:8.2f} | {fields[1][1]:8.2f} | {fields[1][2]:8.2f}")
    print("-" * 80)

# with count:
def print_overlap_block(name, scores):
    print(f"[{name} MATCHING] ---------------------------------------------------")

    # Trigger Summary ROUGE-L
    if 'trigger_summary_rouge' in scores:
        p, r, f1 = scores['trigger_summary_rouge']
        print('Trigger Summary (ROUGE-L) - P: {:6.2f}, R: {:6.2f}, F: {:6.2f}'.format(
            p * 100.0, r * 100.0, f1 * 100.0))
    else:
        print("Trigger Summary (ROUGE-L) - Not Available")

    # Argument Identification
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        scores['arg_id'][3] * 100.0, scores['arg_id'][2], scores['arg_id'][1], 
        scores['arg_id'][4] * 100.0, scores['arg_id'][2], scores['arg_id'][0], scores['arg_id'][5] * 100.0))
    
    # Argument Classification
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        scores['arg_cls'][3] * 100.0, scores['arg_cls'][2], scores['arg_cls'][1], 
        scores['arg_cls'][4] * 100.0, scores['arg_cls'][2], scores['arg_cls'][0], scores['arg_cls'][5] * 100.0))

    print("---------------------------------------------------------------------")


def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    

    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        ##### debug
        # print("[get_span_idx WARNING] No matching span found.")
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]

def get_span_idx_tri(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return [(-1, -1)]
    else:
        if trigger_span is None:
            return candidates
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))

def cal_scores(gold_triggers, pred_triggers, gold_roles, pred_roles):
    assert len(gold_triggers) == len(pred_triggers)
    assert len(gold_roles) == len(pred_roles)

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    # Argument ID
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],) + r[1][:-1] for r in gold_role if r[1][2] not in exclude_roles])
        pred_set = set([(r[0][2],) + r[1][:-1] for r in pred_role if r[1][2] not in exclude_roles])
        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)

    # Argument CLS
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],) + r[1] for r in gold_role if r[1][2] not in exclude_roles])
        pred_set = set([(r[0][2],) + r[1] for r in pred_role if r[1][2] not in exclude_roles])
        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)

    scores = {
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }

    return scores

## addition
def cal_scores_overlap_variant(gold_triggers, pred_triggers, gold_roles, pred_roles, overlap_fn):
    assert len(gold_triggers) == len(pred_triggers)
    assert len(gold_roles) == len(pred_roles)

    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for gold_trigs, pred_trigs, gold_roles_list, pred_roles_list in zip(gold_triggers, pred_triggers, gold_roles, pred_roles):
        # Argument ID
        gold_id_set = [(r[0][2], r[1][0], r[1][1]) for r in gold_roles_list if r[1][2] not in exclude_roles]
        pred_id_set = [(r[0][2], r[1][0], r[1][1]) for r in pred_roles_list if r[1][2] not in exclude_roles]
        used_pred = set()
        for gid in gold_id_set:
            gold_arg_id_num += 1
            for i, pid in enumerate(pred_id_set):
                if i in used_pred:
                    continue
                if pid[0] == gid[0] and overlap_fn(pid[1:], gid[1:]):
                    match_arg_id_num += 1
                    used_pred.add(i)
                    break
        pred_arg_id_num += len(pred_id_set)

        # Argument CLS
        gold_cls_set = [(r[0][2], r[1][0], r[1][1], r[1][2]) for r in gold_roles_list if r[1][2] not in exclude_roles]
        pred_cls_set = [(r[0][2], r[1][0], r[1][1], r[1][2]) for r in pred_roles_list if r[1][2] not in exclude_roles]
        for gcl in gold_cls_set:
            gold_arg_cls_num += 1
            for pcl in pred_cls_set:
                if pcl[0] == gcl[0] and pcl[3] == gcl[3] and overlap_fn(pcl[1:3], gcl[1:3]):
                    match_arg_cls_num += 1
                    break
        pred_arg_cls_num += len(pred_cls_set)

    return {
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
cal_scores_overlap = lambda gtr, ptr, grl, prl: cal_scores_overlap_variant(gtr, ptr, grl, prl, spans_overlap)
cal_scores_scirex  = lambda gtr, ptr, grl, prl: cal_scores_overlap_variant(gtr, ptr, grl, prl, scirex_overlap)
cal_scores_iou     = lambda gtr, ptr, grl, prl: cal_scores_overlap_variant(gtr, ptr, grl, prl, iou_overlap)

## addition ends

# set GPU device
torch.cuda.set_device(config.gpu_device)

# check valid styles
assert np.all([style in ['event_type_sent', 'keywords', 'template'] for style in config.input_style])
assert np.all([style in ['trigger:sentence', 'argument:sentence'] for style in config.output_style])
              
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
special_tokens = ['<Trigger>', '<sep>']
tokenizer.add_tokens(special_tokens)

if args.eval_batch_size:
    config.eval_batch_size=args.eval_batch_size

# load data
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)
with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.e2e_model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.e2e_model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_gold_triggers, test_gold_roles, test_pred_triggers, test_pred_roles = [], [], [], []
all_test_wnd_ids = []
write_object = []
all_test_tokens = []  # NEW: collect tokens for ROUGE L

role_match_stats = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
role_match_stats_overlap = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})


for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    p_triggers = [[] for _ in range(len(batch.tokens))]
    p_roles = [[] for _ in range(len(batch.tokens))]
    p_texts = [[] for _ in range(len(batch.tokens))]
    for event_type in vocab['event_type_itos']:
        theclass = getattr(
            sys.modules[template_file],
            event_type.replace(':', '_').replace('-', '_').replace('/', '_'),
            False
        )

        inputs = []
        for tokens in batch.tokens:
            template = theclass(config.input_style, config.output_style, tokens, event_type)
            inputs.append(template.generate_input_str(''))
        
        inputs = tokenizer(inputs, return_tensors='pt', padding=True, max_length=config.max_length)
        enc_idxs = inputs['input_ids'].cuda()
        enc_attn = inputs['attention_mask'].cuda()
        
        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=config.beam_size, max_length=config.max_output_length)
        final_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        
        for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
            template = theclass(config.input_style, config.output_style, tokens, event_type)
            pred_object = template.decode(p_text)
            
            pred_trigger_object = []
            pred_argument_object = []
            for obj in pred_object:
                if obj[1] == event_type:
                    pred_trigger_object.append(obj)
                else:
                    pred_argument_object.append(obj)
            
            # decode triggers
            triggers_ = [mention + (event_type, kwargs) for span, _, kwargs in pred_trigger_object for mention in get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)]

            triggers_ = [t for t in triggers_ if t[0] != -1]
            p_triggers_ = [t[:-1] for t in triggers_]
            p_triggers[bid].extend(p_triggers_)
            
            # decode arguments
            tri_id2obj = {}
            for t in triggers_:
                tri_id2obj[t[3]['tri counter']] = (t[0], t[1], t[2])
            
            roles_ = []
            for span, role_type, kwargs in pred_argument_object:
                corres_tri_id = kwargs['cor tri cnt']
                if corres_tri_id in tri_id2obj.keys():
                    arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, tri_id2obj[corres_tri_id])
                    if arg_span[0] != -1:
                        roles_.append((tri_id2obj[corres_tri_id], (arg_span[0], arg_span[1], role_type)))

                else:
                    arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)
                    if arg_span[0] != -1:
                        roles_.append(((0, 1, event_type), (arg_span[0], arg_span[1], role_type)))
            
            p_roles[bid].extend(roles_)
            p_texts[bid].append(p_text)

    # post-process: remove duplicate triggers
    for bid in range(len(p_triggers)):
        seen = set()
        deduped = []
        for s, e, t in p_triggers[bid]:
            if (s, e) not in seen:
                seen.add((s, e))
                deduped.append((s, e, t))
        p_triggers[bid] = deduped

    p_roles = [list(set(role)) for role in p_roles]
    
    if config.ignore_first_header:
        for bid, wnd_id in enumerate(batch.wnd_ids):
            if int(wnd_id.split('-')[-1]) < 4:
                p_triggers[bid] = []
                p_roles[bid] = []
    
    test_gold_triggers.extend(batch.triggers)
    test_gold_roles.extend(batch.roles)
    test_pred_triggers.extend(p_triggers)
    test_pred_roles.extend(p_roles)

    all_test_tokens.extend(batch.tokens)  # ROUGE L

    all_test_wnd_ids.extend(batch.wnd_ids)

    # Accumulate per-role stats using raw argument text
    for bid in range(len(batch.tokens)):
        gold_roles_list = batch.roles[bid]
        pred_roles_list = p_roles[bid]
        tokens = batch.tokens[bid]

        gold_role_texts = defaultdict(set)
        pred_role_texts = defaultdict(set)

        for _, (s, e, role) in gold_roles_list:
            if s >= 0 and e <= len(tokens):  # safety check
                text = " ".join(tokens[s:e]).strip().lower()
                if text:
                    gold_role_texts[role].add(text)


        for _, (s, e, role) in pred_roles_list:
            if s >= 0 and e <= len(tokens):  # safety check
                text = " ".join(tokens[s:e]).strip().lower()
                if text:
                    pred_role_texts[role].add(text)


        for role in set(gold_role_texts.keys()).union(pred_role_texts.keys()):
            g_set = gold_role_texts.get(role, set())
            p_set = pred_role_texts.get(role, set())
            match = g_set & p_set

            role_match_stats[role]["gold_total"] += len(g_set)
            role_match_stats[role]["pred_total"] += len(p_set)
            role_match_stats[role]["matched"] += len(match)
            # Overlap-based role-wise stats (span match)

            gold_roles_list = batch.roles[bid]
            pred_roles_list = p_roles[bid]

            gold_by_role = defaultdict(list)
            pred_by_role = defaultdict(list)

            for _, (s, e, role) in gold_roles_list:
                gold_by_role[role].append((s, e))
            for _, (s, e, role) in pred_roles_list:
                pred_by_role[role].append((s, e))

            for role in set(gold_by_role.keys()).union(pred_by_role.keys()):
                g_list = gold_by_role.get(role, [])
                p_list = pred_by_role.get(role, [])
                match_count = 0
                matched = set()

                for i, g_span in enumerate(g_list):
                    for j, p_span in enumerate(p_list):
                        if (j not in matched) and spans_overlap(g_span, p_span):
                            match_count += 1
                            matched.add(j)
                            break

                role_match_stats_overlap[role]["gold_total"] += len(g_list)
                role_match_stats_overlap[role]["pred_total"] += len(p_list)
                role_match_stats_overlap[role]["matched"] += match_count

    for gt, gr, pt, pr, te in zip(batch.triggers, batch.roles, p_triggers, p_roles, p_texts):
        write_object.append({
            "pred text": te,
            "pred triggers": pt,
            "gold triggers": gt,
            "pred roles": pr,
            "gold roles": gr
        })
            
progress.close()

def compute_rouge_L_for_event_tuples(gold_roles_all, pred_roles_all, token_lists):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    p_scores = []
    r_scores = []
    f1_scores = []

    for i, (gold_roles, pred_roles, tokens) in enumerate(zip(gold_roles_all, pred_roles_all, token_lists)):
        def extract_span_seq(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            if not roles or not roles[0] or roles[0][0][0] < 0:
                trigger = ""
            else:
                trigger_span = roles[0][0]  # (start, end, event_type)
                trigger = " ".join(tokens[trigger_span[0]:trigger_span[1]])
            for (_, _, _), (s, e, role) in roles:
                if role in parts and not parts[role] and 0 <= s < len(tokens) and 0 < e <= len(tokens):
                    parts[role] = " ".join(tokens[s:e])
            return (parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"])

        g_tuple = extract_span_seq(gold_roles)
        p_tuple = extract_span_seq(pred_roles)

        if all(not v for v in g_tuple) and all(not v for v in p_tuple):
            continue

        gold_text = " ".join(g_tuple).strip()
        pred_text = " ".join(p_tuple).strip()

        score = scorer.score(gold_text, pred_text)["rougeL"]
        p_scores.append(score.precision)
        r_scores.append(score.recall)
        f1_scores.append(score.fmeasure)

    avg_p = sum(p_scores) / len(p_scores) if p_scores else 0.0
    avg_r = sum(r_scores) / len(r_scores) if r_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return avg_p, avg_r, avg_f1
def compute_rouge_L_eventtype_wise(gold_roles_all, pred_roles_all, token_lists):


    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for gold_roles, pred_roles, tokens in zip(gold_roles_all, pred_roles_all, token_lists):
        def extract_span_seq(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            if not roles or not roles[0] or roles[0][0][0] < 0:
                trig = ""
                etype = "UNKNOWN"
            else:
                trig_span = roles[0][0]
                trig = " ".join(tokens[trig_span[0]:trig_span[1]])
                etype = trig_span[2]

            for (_, _, _), (s, e, role) in roles:
                if role in parts and not parts[role] and 0 <= s < len(tokens) and 0 < e <= len(tokens):
                    parts[role] = " ".join(tokens[s:e])

            return (parts["Agent"], trig, parts["PrimaryObject"], parts["SecondaryObject"]), etype

        (g_tuple, g_type) = extract_span_seq(gold_roles)
        (p_tuple, _)      = extract_span_seq(pred_roles)

        if all(not v for v in g_tuple) and all(not v for v in p_tuple):
            continue

        g_text = " ".join(g_tuple).strip()
        p_text = " ".join(p_tuple).strip()

        score = scorer.score(g_text, p_text)["rougeL"]
        stats[g_type].append((score.precision, score.recall, score.fmeasure))

    # aggregate by event type
    avg_stats = {}
    for etype, scores in stats.items():
        if scores:
            p = sum(s[0] for s in scores) / len(scores)
            r = sum(s[1] for s in scores) / len(scores)
            f = sum(s[2] for s in scores) / len(scores)
        else:
            p, r, f = 0.0, 0.0, 0.0
        avg_stats[etype] = (p, r, f)

    return avg_stats
def compute_rouge_L_domain_wise(gold_roles_all, pred_roles_all, token_lists, wnd_ids):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for gold_roles, pred_roles, tokens, wnd_id in zip(gold_roles_all, pred_roles_all, token_lists, wnd_ids):
        domain = wnd_id.strip().split("_")[0]

        def extract_span_seq(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            if not roles or not roles[0] or roles[0][0][0] < 0:
                trig = ""
            else:
                trig_span = roles[0][0]
                trig = " ".join(tokens[trig_span[0]:trig_span[1]])

            for (_, _, _), (s, e, role) in roles:
                if role in parts and not parts[role] and 0 <= s < len(tokens) and 0 < e <= len(tokens):
                    parts[role] = " ".join(tokens[s:e])

            return (parts["Agent"], trig, parts["PrimaryObject"], parts["SecondaryObject"])

        g_tuple = extract_span_seq(gold_roles)
        p_tuple = extract_span_seq(pred_roles)

        if all(not v for v in g_tuple) and all(not v for v in p_tuple):
            continue

        g_text = " ".join(g_tuple).strip()
        p_text = " ".join(p_tuple).strip()

        score = scorer.score(g_text, p_text)["rougeL"]
        stats[domain].append((score.precision, score.recall, score.fmeasure))

    # aggregate by domain
    avg_stats = {}
    for domain, scores in stats.items():
        if scores:
            p = sum(s[0] for s in scores) / len(scores)
            r = sum(s[1] for s in scores) / len(scores)
            f = sum(s[2] for s in scores) / len(scores)
        else:
            p, r, f = 0.0, 0.0, 0.0
        avg_stats[domain] = (p, r, f)

    return avg_stats
def print_rouge_stats(name, stats):
    print(f"\n[ROUGE-L SUMMARY - {name}] --------------------------------------------------------")
    print(f"{'Key':25s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 50)
    for k in sorted(stats.keys()):
        p, r, f1 = stats[k]
        print(f"{k:25s} | {p*100:6.2f} | {r*100:6.2f} | {f1*100:6.2f}")
    print("-" * 50)


# calculate scores
test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)

test_scores_overlap = cal_scores_overlap(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)
test_scores_scirex = cal_scores_scirex(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)
test_scores_iou  = cal_scores_iou(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)

""" test_pred_roles is a list of tuples:
(
  (trigger_start_idx, trigger_end_idx, event_type),  # trigger info
  (arg_start_idx, arg_end_idx, role_type)            # argument span + role
)
"""

overall_rouge_p, overall_rouge_r, overall_rouge_f1 = compute_rouge_L_for_event_tuples(
    test_gold_roles, test_pred_roles, all_test_tokens
)

print(f"\n[ROUGE-L OVERALL] P: {overall_rouge_p:.2%}, R: {overall_rouge_r:.2%}, F1: {overall_rouge_f1:.2%}")

eventtype_rouge_scores = compute_rouge_L_eventtype_wise(
    test_gold_roles, test_pred_roles, all_test_tokens
)

print_rouge_stats("Event Type", eventtype_rouge_scores)

domain_rouge_scores = compute_rouge_L_domain_wise(
    test_gold_roles, test_pred_roles, all_test_tokens, all_test_wnd_ids
)

print_rouge_stats("Domain", domain_rouge_scores)


print("========= [EXACT MATCHING] ===========================================")
print_overlap_block("EXACT", test_scores)
# for Role-wise
role_stats_C_em = compute_rolewise_f1_dual(test_gold_roles, test_pred_roles, overlap_fn=None)
print_rolewise_stats("EXACT", role_stats_C_em)
# for Event type wise
etype_stats_em = compute_eventtype_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, overlap_fn=None)
print_eventtype_stats_extended("EXACT", etype_stats_em)
# for Domain wise
domain_stats_em_ext = compute_domain_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, all_test_wnd_ids, overlap_fn=None)
print_domain_stats_extended("EXACT", domain_stats_em_ext)

print("========= [OVERLAP MATCHING] =========================================")
print_overlap_block("OVERLAP", test_scores_overlap)
# for Role-wise
role_stats_C_ov = compute_rolewise_f1_dual(test_gold_roles, test_pred_roles, overlap_fn=spans_overlap)
print_rolewise_stats("OVERLAP", role_stats_C_ov)
# for Event type wise
etype_stats_ov = compute_eventtype_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, overlap_fn=spans_overlap)
print_eventtype_stats_extended("OVERLAP", etype_stats_ov)
# for Domain wise
domain_stats_ov_ext = compute_domain_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, all_test_wnd_ids, overlap_fn=spans_overlap)
print_domain_stats_extended("OVERLAP", domain_stats_ov_ext)

print("========= [SCIReX MATCHING] ==========================================")
print_overlap_block("SCIReX", test_scores_scirex)
# for Role-wise
role_stats_C_sci = compute_rolewise_f1_dual(test_gold_roles, test_pred_roles, overlap_fn=scirex_overlap)
print_rolewise_stats("SCIReX", role_stats_C_sci)
# for Event type wise
etype_stats_sci = compute_eventtype_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, overlap_fn=scirex_overlap)
print_eventtype_stats_extended("SCIReX", etype_stats_sci)
# for Domain wise
domain_stats_sci_ext = compute_domain_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, all_test_wnd_ids, overlap_fn=scirex_overlap)
print_domain_stats_extended("SCIReX", domain_stats_sci_ext)

print("========= [IoU MATCHING] =============================================")
print_overlap_block("IoU", test_scores_iou)
# for Role-wise
role_stats_C_iou = compute_rolewise_f1_dual(test_gold_roles, test_pred_roles, overlap_fn=iou_overlap)
print_rolewise_stats("IoU", role_stats_C_iou)
# for Event type wise
etype_stats_iou = compute_eventtype_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, overlap_fn=iou_overlap)
print_eventtype_stats_extended("IoU", etype_stats_iou)
# for Domain wise
domain_stats_iou_ext = compute_domain_f1_extended(
    test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles, all_test_wnd_ids, overlap_fn=iou_overlap)
print_domain_stats_extended("IoU", domain_stats_iou_ext)

if args.write_file:
    with open(args.write_file, 'w') as fw:
        json.dump(write_object, fw, indent=4)
