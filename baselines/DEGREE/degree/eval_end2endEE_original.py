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



#### addition
def spans_overlap(span1, span2):
    s1, e1 = span1
    s2, e2 = span2
    return max(s1, s2) <= min(e1, e2)
#### addition ends


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
    
    ##### debug
    # print("="*80)
    # print("[get_span_idx DEBUG]")
    # print(f"Original span text: '{span}'")
    # print(f"Tokenized span (word pieces): {words}")
    # print(f"Input pieces: {pieces}")
    # print(f"Token start idxs: {token_start_idxs}")
    # if trigger_span:
    #     print(f"Trigger span: {trigger_span}")
    # print("="*80)

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
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t[0], t[1]) for t in gold_trigger])
        pred_set = set([(t[0], t[1]) for t in pred_trigger])

        # print("========== ARG DEBUG ==========")
        # print("Gold role example:", gold_role)
        # print("Pred role example:", pred_role)
        # print("Gold set:", gold_set)
        # print("Pred set:", pred_set)
        # print("Intersection:", gold_set & pred_set)
        # print("================================")

        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    
    # arg_id
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1][:-1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1][:-1] for r in pred_role])
        
        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)
        
    # arg_cls
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1] for r in pred_role])
        
        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)
    
    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
    
    return scores

#### addition
def cal_scores_overlap(gold_triggers, pred_triggers, gold_roles, pred_roles):
    assert len(gold_triggers) == len(pred_triggers)
    assert len(gold_roles) == len(pred_roles)

    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0

    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0

    for gold_trigs, pred_trigs, gold_roles_list, pred_roles_list in zip(gold_triggers, pred_triggers, gold_roles, pred_roles):
        # Deduplicate trigger spans and role mentions
        gold_trigs = list({(t[0], t[1], t[2]) for t in gold_trigs})
        pred_trigs = list({(t[0], t[1], t[2]) for t in pred_trigs})

        # Flatten role format to deduplicate
        gold_roles_flat = list({(r[0][2], r[1][0], r[1][1], r[1][2]) for r in gold_roles_list})
        pred_roles_flat = list({(r[0][2], r[1][0], r[1][1], r[1][2]) for r in pred_roles_list})

        # Rebuild original structure: ((trigger_start, trigger_end, event_type), (arg_start, arg_end, role_type))
        gold_roles_list = [((0, 0, t), (s, e, r)) for (t, s, e, r) in gold_roles_flat]
        pred_roles_list = [((0, 0, t), (s, e, r)) for (t, s, e, r) in pred_roles_flat]

        

        # --- TRIGGER ID (span only)
        matched_tri_id = set()
        used_preds = set()
        for g in gold_trigs:
            for i, p in enumerate(pred_trigs):
                if i in used_preds:
                    continue
                if spans_overlap((g[0], g[1]), (p[0], p[1])):
                    matched_tri_id.add((g, p))
                    used_preds.add(i)
                    break

        gold_tri_id_num += len(gold_trigs)
        pred_tri_id_num += len(pred_trigs)
        match_tri_id_num += len(matched_tri_id)

        # --- TRIGGER CLS (span + type)
        matched_tri_cls = set()
        used_preds = set()
        for g in gold_trigs:
            for i, p in enumerate(pred_trigs):
                if i in used_preds:
                    continue
                if g[2] == p[2] and spans_overlap((g[0], g[1]), (p[0], p[1])):
                    matched_tri_cls.add((g, p))
                    used_preds.add(i)
                    break

        gold_tri_cls_num += len(gold_trigs)
        pred_tri_cls_num += len(pred_trigs)
        match_tri_cls_num += len(matched_tri_cls)

        # --- ARGUMENT ID
        gold_id_set = [(r[0][2], r[1][0], r[1][1]) for r in gold_roles_list]
        pred_id_set = [(r[0][2], r[1][0], r[1][1]) for r in pred_roles_list]
        
        matched_id = set()
        used_preds = set()
        for i, gid in enumerate(gold_id_set):
            for j, pid in enumerate(pred_id_set):
                if j in used_preds:
                    continue
                if gid[0] == pid[0] and spans_overlap(gid[1:], pid[1:]):
                    matched_id.add((gid, pid))
                    used_preds.add(j)
                    break

        gold_arg_id_num += len(gold_id_set)
        pred_arg_id_num += len(pred_id_set)
        match_arg_id_num += len(matched_id)

        # --- ARGUMENT CLS
        gold_cls_set = [(r[0][2], r[1][0], r[1][1], r[1][2]) for r in gold_roles_list]
        pred_cls_set = [(r[0][2], r[1][0], r[1][1], r[1][2]) for r in pred_roles_list]
        matched_cls = set()
        used_preds = set()
        for i, gcl in enumerate(gold_cls_set):
            for j, pcl in enumerate(pred_cls_set):
                if j in used_preds:
                    continue
                if gcl[0] == pcl[0] and gcl[3] == pcl[3] and spans_overlap(gcl[1:3], pcl[1:3]):
                    matched_cls.add((gcl, pcl))
                    used_preds.add(j)
                    break

        gold_arg_cls_num += len(gold_cls_set)
        pred_arg_cls_num += len(pred_cls_set)
        match_arg_cls_num += len(matched_cls)

    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
    return scores
#### addition ends

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

# eval dev set
if not args.no_dev:
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
    dev_gold_triggers, dev_gold_roles, dev_pred_triggers, dev_pred_roles = [], [], [], []
    # #### addition
    # from collections import defaultdict
    # role_match_stats = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    # role_match_stats_overlap = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    # #### addition ends
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        
        p_triggers = [[] for _ in range(len(batch.tokens))]
        p_roles = [[] for _ in range(len(batch.tokens))]
        for event_type in vocab['event_type_itos']:
            # replaced this (for our data to run)
            # theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
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
                #### replaced
                
                # pred_trigger_object = []
                # pred_argument_object = []
                # for obj in pred_object:
                #     if obj[1] == event_type:
                #         pred_trigger_object.append(obj)
                #     else:
                #         pred_argument_object.append(obj)
                #### replaced end

                # Keep only trigger objects of this event_type
                pred_trigger_object = [
                    obj for obj in pred_object if obj[1] == event_type
                ]

                # Get tri counters for this event type's triggers
                allowed_tri_ids = {
                    obj[2].get("tri counter")
                    for obj in pred_trigger_object
                    if isinstance(obj, tuple) and len(obj) > 2
                }

                # Keep only arguments attached to those triggers
                pred_argument_object = []
                for obj in pred_object:
                    if obj[1] != event_type:
                        kwargs = obj[2] if len(obj) > 2 else {}
                        tri_id = kwargs.get("cor tri cnt")
                        if tri_id in allowed_tri_ids:
                            pred_argument_object.append(obj)


                
                # decode triggers
                triggers_ = [mention + (event_type, kwargs) for span, _, kwargs in pred_trigger_object for mention in get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)]
                triggers_ = [t for t in triggers_ if t[0] != -1]
                p_triggers_ = [t[:-1] for t in triggers_]
                p_triggers_ = list(set(p_triggers_))
                # p_triggers[bid].extend(p_triggers_)
                # addition: Deduplicate using just (s, e, type), ignore kwargs
                # Deduplicate across event types by (s, e)
                seen_trigger_spans = set()
                new_trigger_preds = []
                for (s, e, t) in p_triggers_:
                    if (s, e) not in seen_trigger_spans:
                        new_trigger_preds.append((s, e, t))
                        seen_trigger_spans.add((s, e))

                p_triggers[bid].extend(new_trigger_preds)
                # addition ends
                
                # decode arguments
                tri_id2obj = {}
                for t in triggers_:
                    try:
                        tri_id2obj[t[3]['tri counter']] = (t[0], t[1], t[2])
                    except:
                        ipdb.set_trace()
                
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
                
        p_roles = [list(set(role)) for role in p_roles]
        
        dev_gold_triggers.extend(batch.triggers)
        dev_gold_roles.extend(batch.roles)
        dev_pred_triggers.extend(p_triggers)
        dev_pred_roles.extend(p_roles)
                
    progress.close()
    
    # calculate scores
    dev_scores = cal_scores(dev_gold_triggers, dev_pred_triggers, dev_gold_roles, dev_pred_roles)

    #### addition
    dev_scores_overlap = cal_scores_overlap(dev_gold_triggers, dev_pred_triggers, dev_gold_roles, dev_pred_roles)
    #### addition ends
    
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1], 
        dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0], dev_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1], 
        dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0], dev_scores['tri_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_id'][3] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][1], 
        dev_scores['arg_id'][4] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][0], dev_scores['arg_id'][5] * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['arg_cls'][3] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][1], 
        dev_scores['arg_cls'][4] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][0], dev_scores['arg_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    
    #### addition
    print("[OVERLAP MATCHING] ---------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores_overlap['tri_id'][3] * 100.0, dev_scores_overlap['tri_id'][2], dev_scores_overlap['tri_id'][1], 
        dev_scores_overlap['tri_id'][4] * 100.0, dev_scores_overlap['tri_id'][2], dev_scores_overlap['tri_id'][0], dev_scores_overlap['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores_overlap['tri_cls'][3] * 100.0, dev_scores_overlap['tri_cls'][2], dev_scores_overlap['tri_cls'][1], 
        dev_scores_overlap['tri_cls'][4] * 100.0, dev_scores_overlap['tri_cls'][2], dev_scores_overlap['tri_cls'][0], dev_scores_overlap['tri_cls'][5] * 100.0))
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores_overlap['arg_id'][3] * 100.0, dev_scores_overlap['arg_id'][2], dev_scores_overlap['arg_id'][1], 
        dev_scores_overlap['arg_id'][4] * 100.0, dev_scores_overlap['arg_id'][2], dev_scores_overlap['arg_id'][0], dev_scores_overlap['arg_id'][5] * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores_overlap['arg_cls'][3] * 100.0, dev_scores_overlap['arg_cls'][2], dev_scores_overlap['arg_cls'][1], 
        dev_scores_overlap['arg_cls'][4] * 100.0, dev_scores_overlap['arg_cls'][2], dev_scores_overlap['arg_cls'][0], dev_scores_overlap['arg_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    #### addition ends
    
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_gold_triggers, test_gold_roles, test_pred_triggers, test_pred_roles = [], [], [], []
write_object = []

#### addition
role_match_stats = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
role_match_stats_overlap = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})

#### addition ends

for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    p_triggers = [[] for _ in range(len(batch.tokens))]
    p_roles = [[] for _ in range(len(batch.tokens))]
    p_texts = [[] for _ in range(len(batch.tokens))]
    for event_type in vocab['event_type_itos']:
        #replaced this
        # theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
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
            #### replaced
            # pred_trigger_object = []
            # pred_argument_object = []
            # for obj in pred_object:
            #     if obj[1] == event_type:
            #         pred_trigger_object.append(obj)
            #     else:
            #         pred_argument_object.append(obj)
            #### replaced end

            # Keep only trigger objects of this event_type
            pred_trigger_object = [
                obj for obj in pred_object if obj[1] == event_type
            ]

            # Get tri counters for this event type's triggers
            allowed_tri_ids = {
                obj[2].get("tri counter")
                for obj in pred_trigger_object
                if isinstance(obj, tuple) and len(obj) > 2
            }

            # Keep only arguments attached to those triggers
            pred_argument_object = []
            for obj in pred_object:
                if obj[1] != event_type:
                    kwargs = obj[2] if len(obj) > 2 else {}
                    tri_id = kwargs.get("cor tri cnt")
                    if tri_id in allowed_tri_ids:
                        pred_argument_object.append(obj)


            #### debug
            # print("="*80)
            # print(f"[DEBUG] Event Type: {event_type}")
            # print(f"[DEBUG] Model Output Text:\n{p_text}")
            # print(f"[DEBUG] Decoded Object:\n{pred_object}")
            # print(f"[DEBUG] Pred Argument Objects:\n{pred_argument_object}")
            # print("="*80)

            
            # decode triggers
            triggers_ = [mention + (event_type, kwargs) for span, _, kwargs in pred_trigger_object for mention in get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)]

            triggers_ = [t for t in triggers_ if t[0] != -1]
            p_triggers_ = [t[:-1] for t in triggers_]
            p_triggers_ = list(set(p_triggers_))

            # p_triggers[bid].extend(p_triggers_)

            # addition Deduplicate across event types by (s, e)
            seen_trigger_spans = set()
            new_trigger_preds = []
            for (s, e, t) in p_triggers_:
                if (s, e) not in seen_trigger_spans:
                    new_trigger_preds.append((s, e, t))
                    seen_trigger_spans.add((s, e))

            p_triggers[bid].extend(new_trigger_preds)
            # addition ends
            
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
                    # ##### debug
                    # else:
                    #     print("="*80)
                    #     print("[DEBUG] Skipped due to arg_span -1")
                    #     print(f"Tokens: {' '.join(batch.tokens[bid])}")
                    #     print(f"Span Text: {span}")
                    #     print(f"Event Type: {event_type}")
                    #     print(f"Decoded Output Text: {p_text}")
                    #     print(f"TriID: {corres_tri_id} | Mapped Trigger: {tri_id2obj.get(corres_tri_id, 'N/A')}")
                    #     print("="*80)

                else:
                    arg_span = get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer)
                    if arg_span[0] != -1:
                        roles_.append(((0, 1, event_type), (arg_span[0], arg_span[1], role_type)))
                    # ##### debug
                    # else:
                    #     print("="*80)
                    #     print("[DEBUG] Skipped (no matching trigger)")
                    #     print(f"Tokens: {' '.join(batch.tokens[bid])}")
                    #     print(f"Span Text: {span}")
                    #     print(f"Event Type: {event_type}")
                    #     print(f"Decoded Output Text: {p_text}")
                    #     print("="*80)
            
            p_roles[bid].extend(roles_)
            p_texts[bid].append(p_text)

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

    #### addition --- Role-wise F1 stats ---
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
        # print("="*80)
        # print(f"[TEST DEBUG] Example #{bid}")
        # print(f"Tokens: {' '.join(tokens)}\n")

        # print("Gold Roles:")
        # for role, texts in gold_role_texts.items():
        #     for text in texts:
        #         print(f"  - Role: {role:15s} | Text: '{text}'")

        # print("\nPredicted Roles:")
        # for role, texts in pred_role_texts.items():
        #     for text in texts:
        #         print(f"  - Role: {role:15s} | Text: '{text}'")

        # print("="*80 + "\n")

        for _, (s, e, role) in pred_roles_list:
            if s >= 0 and e <= len(tokens):  # safety check
                text = " ".join(tokens[s:e]).strip().lower()
                if text:
                    pred_role_texts[role].add(text)
        
        # print("="*80)
        # print(f"[TEST DEBUG] Example #{bid}")
        # print(f"Tokens: {' '.join(tokens)}\n")

        # print("Gold Roles:")
        # for role, texts in gold_role_texts.items():
        #     for text in texts:
        #         print(f"  - Role: {role:15s} | Text: '{text}'")

        # print("\nPredicted Roles:")
        # for role, texts in pred_role_texts.items():
        #     for text in texts:
        #         print(f"  - Role: {role:15s} | Text: '{text}'")

        # print("="*80 + "\n")

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

    #### addition ends

    for gt, gr, pt, pr, te in zip(batch.triggers, batch.roles, p_triggers, p_roles, p_texts):
        write_object.append({
            "pred text": te,
            "pred triggers": pt,
            "gold triggers": gt,
            "pred roles": pr,
            "gold roles": gr
        })
            
progress.close()
    
# calculate scores
test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)
#### addition
test_scores_overlap = cal_scores_overlap(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)
#### addition ends

print("---------------------------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
print("---------------------------------------------------------------------")
print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_id'][3] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][1], 
    test_scores['arg_id'][4] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][0], test_scores['arg_id'][5] * 100.0))
print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_cls'][3] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][1], 
    test_scores['arg_cls'][4] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][0], test_scores['arg_cls'][5] * 100.0))
print("---------------------------------------------------------------------")
#### addition
print("[OVERLAP MATCHING] ---------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores_overlap['tri_id'][3] * 100.0, test_scores_overlap['tri_id'][2], test_scores_overlap['tri_id'][1], 
    test_scores_overlap['tri_id'][4] * 100.0, test_scores_overlap['tri_id'][2], test_scores_overlap['tri_id'][0], test_scores_overlap['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores_overlap['tri_cls'][3] * 100.0, test_scores_overlap['tri_cls'][2], test_scores_overlap['tri_cls'][1], 
    test_scores_overlap['tri_cls'][4] * 100.0, test_scores_overlap['tri_cls'][2], test_scores_overlap['tri_cls'][0], test_scores_overlap['tri_cls'][5] * 100.0))
print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores_overlap['arg_id'][3] * 100.0, test_scores_overlap['arg_id'][2], test_scores_overlap['arg_id'][1], 
    test_scores_overlap['arg_id'][4] * 100.0, test_scores_overlap['arg_id'][2], test_scores_overlap['arg_id'][0], test_scores_overlap['arg_id'][5] * 100.0))
print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores_overlap['arg_cls'][3] * 100.0, test_scores_overlap['arg_cls'][2], test_scores_overlap['arg_cls'][1], 
    test_scores_overlap['arg_cls'][4] * 100.0, test_scores_overlap['arg_cls'][2], test_scores_overlap['arg_cls'][0], test_scores_overlap['arg_cls'][5] * 100.0))
print("---------------------------------------------------------------------")


print("[TEST STATS] Argument Role-wise F1 (aggregated):")
for role, stats in sorted(role_match_stats.items()):
    m = stats["matched"]
    p = stats["pred_total"]
    g = stats["gold_total"]
    prec = m / p if p > 0 else 0
    rec = m / g if g > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  Role: {role:15s} | Match: {m:3d} | Pred: {p:3d} | Gold: {g:3d} | P: {prec:.2f} | R: {rec:.2f} | F1: {f1:.2f}")
print("---------------------------------------------------------------------")  

print("[TEST STATS] Argument Role-wise F1 (overlap span-based):")
for role, stats in sorted(role_match_stats_overlap.items()):
    m = stats["matched"]
    p = stats["pred_total"]
    g = stats["gold_total"]
    prec = m / p if p > 0 else 0
    rec = m / g if g > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  Role: {role:15s} | Match: {m:3d} | Pred: {p:3d} | Gold: {g:3d} | P: {prec:.2f} | R: {rec:.2f} | F1: {f1:.2f}")
print("---------------------------------------------------------------------")
#### addition ends

if args.write_file:
    with open(args.write_file, 'w') as fw:
        json.dump(write_object, fw, indent=4)
