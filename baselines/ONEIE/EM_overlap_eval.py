import json
from collections import defaultdict

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

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

def extract_triggers_and_roles(data):
    trigger_dict = {}
    role_dict = {}

    for entry in data:
        sent_id = entry["sent_id"]
        triggers = []
        roles = []

        entity_map = {}
        for ent in entry.get("entity_mentions", []):
            entity_map[ent["id"]] = (ent["start"], ent["end"])

        for event in entry.get("event_mentions", []):
            trig = event["trigger"]
            event_type = event["event_type"]
            triggers.append((trig["start"], trig["end"], event_type))

            for arg in event.get("arguments", []):
                if "entity_id" in arg and arg["entity_id"] in entity_map:
                    start, end = entity_map[arg["entity_id"]]
                elif "start" in arg and "end" in arg:
                    start, end = arg["start"], arg["end"]
                else:
                    print(f"[WARNING] Skipping argument due to missing span: {arg}")
                    continue
                roles.append(((trig["start"], trig["end"], event_type), (start, end, arg["role"])))

        trigger_dict[sent_id] = triggers
        role_dict[sent_id] = roles

    return trigger_dict, role_dict




# def compute_f1(pred, gold, match_fn):
#     total_pred = sum(len(p) for p in pred)
#     total_gold = sum(len(g) for g in gold)
#     matched = 0
#     for p_list, g_list in zip(pred, gold):
#         matched_set = set()
#         for p in p_list:
#             for g in g_list:
#                 if match_fn(p, g):
#                     matched_set.add((p, g))
#                     break
#         matched += len(matched_set)
#     prec = matched / total_pred if total_pred > 0 else 0
#     rec = matched / total_gold if total_gold > 0 else 0
#     f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
#     return prec, rec, f1, matched, total_pred, total_gold
def compute_f1(pred_dict, gold_dict, match_fn, label=""):
    """
    Computes F1 with matched, total_pred, total_gold.
    For OneIE: Event-type sensitive, trigger-insensitive.
    Filters out Agent, PrimaryObject, SecondaryObject roles.
    match_fn: function (pred, gold) -> bool
    Format of pred/gold: ((tri_s, tri_e, event_type), (arg_s, arg_e, role))
    """
    matched = 0
    total_pred = 0
    total_gold = 0

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_dict:
        pred_list = [x for x in pred_dict.get(sent_id, []) if x[1][2] not in exclude_roles]
        gold_list = [x for x in gold_dict.get(sent_id, []) if x[1][2] not in exclude_roles]

        total_pred += len(pred_list)
        total_gold += len(gold_list)

        matched_set = set()
        used_gold = set()

        for p in pred_list:
            for idx, g in enumerate(gold_list):
                if idx in used_gold:
                    continue
                # Only compare if event types match
                if p[0][2] != g[0][2]:
                    continue
                if match_fn(p, g):
                    matched_set.add((p, g))
                    used_gold.add(idx)
                    break

        matched += len(matched_set)

    prec = matched / total_pred if total_pred > 0 else 0
    rec = matched / total_gold if total_gold > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return prec, rec, f1, matched, total_pred, total_gold



"""
Role wise
"""
def compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=None):
    from collections import defaultdict

    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        g_list = [r for r in gold_roles[sent_id] if r[1][2] not in exclude_roles]
        p_list = [r for r in pred_roles.get(sent_id, []) if r[1][2] not in exclude_roles]

        # Arg C: match by role + event type
        g_by_role = defaultdict(list)
        p_by_role = defaultdict(list)
        for ((_, _, e), (s, e_, r)) in g_list:
            g_by_role[r].append((s, e_, r, e))  # (start, end, role, event_type)
        for ((_, _, e), (s, e_, r)) in p_list:
            p_by_role[r].append((s, e_, r, e))

        for role in set(g_by_role) | set(p_by_role):
            g_spans = g_by_role[role]
            p_spans = p_by_role[role]
            stats_C[role]["gold_total"] += len(g_spans)
            stats_C[role]["pred_total"] += len(p_spans)
            matched = set()
            for i_g, g in enumerate(g_spans):
                for i_p, p in enumerate(p_spans):
                    if i_p in matched:
                        continue
                    if p[2] == g[2] and p[3] == g[3] and (overlap_fn(p[:2], g[:2]) if overlap_fn else p[:2] == g[:2]):
                        matched.add(i_p)
                        stats_C[role]["matched"] += 1
                        break

    return stats_C

def print_rolewise_f1_dual(stats_C, name="EXACT"):
    print(f"\n[ROLE-WISE ARGUMENT CLASSIFICATION - {name}] ----------------------------------")
    print(f"{'Role':20s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}")
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

"""
event type wise
"""
def compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=None):
    """
    Computes Arg_I and Arg_C F1 grouped by event type.
    - Event-type–sensitive
    - Trigger-insensitive
    - Removes Agent, PrimaryObject, SecondaryObject
    """
    from collections import defaultdict

    stats = {
        "Arg_I": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
        "Arg_C": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
    }

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        g_list = [r for r in gold_roles[sent_id] if r[1][2] not in exclude_roles]
        p_list = [r for r in pred_roles.get(sent_id, []) if r[1][2] not in exclude_roles]

        # Group argument spans by event type
        g_by_type_I = defaultdict(list)
        p_by_type_I = defaultdict(list)
        g_by_type_C = defaultdict(list)
        p_by_type_C = defaultdict(list)

        for ((_, _, etype), (s, e, _)) in g_list:
            g_by_type_I[etype].append((s, e))
        for ((_, _, etype), (s, e, _)) in p_list:
            p_by_type_I[etype].append((s, e))

        for ((_, _, etype), (s, e, r)) in g_list:
            g_by_type_C[etype].append((s, e, r, etype))
        for ((_, _, etype), (s, e, r)) in p_list:
            p_by_type_C[etype].append((s, e, r, etype))

        # Compute Arg_I F1
        for etype in set(g_by_type_I) | set(p_by_type_I):
            g_spans = g_by_type_I[etype]
            p_spans = p_by_type_I[etype]
            stats["Arg_I"][etype]["gold_total"] += len(g_spans)
            stats["Arg_I"][etype]["pred_total"] += len(p_spans)
            matched = set()
            for i_p, p in enumerate(p_spans):
                for i_g, g in enumerate(g_spans):
                    if i_g in matched:
                        continue
                    if overlap_fn(p, g) if overlap_fn else p == g:
                        stats["Arg_I"][etype]["matched"] += 1
                        matched.add(i_g)
                        break

        # Compute Arg_C F1
        for etype in set(g_by_type_C) | set(p_by_type_C):
            g_spans = g_by_type_C[etype]
            p_spans = p_by_type_C[etype]
            stats["Arg_C"][etype]["gold_total"] += len(g_spans)
            stats["Arg_C"][etype]["pred_total"] += len(p_spans)
            matched = set()
            for i_p, p in enumerate(p_spans):
                for i_g, g in enumerate(g_spans):
                    if i_g in matched:
                        continue
                    if p[2] == g[2] and p[3] == g[3] and (overlap_fn(p[:2], g[:2]) if overlap_fn else p[:2] == g[:2]):
                        stats["Arg_C"][etype]["matched"] += 1
                        matched.add(i_g)
                        break

    return stats


def print_eventtype_stats_extended(name, stats_dict):
    print(f"\n[EVENT-TYPE ARGUMENT MATCH - {name}] -----------------------------------------------------------")
    print(f"{'EventType':25s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>8s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>8s}")
    print("-" * 90)

    types = set(stats_dict["Arg_I"].keys()) | set(stats_dict["Arg_C"].keys())

    for etype in sorted(types):
        def compute(typedict):
            s = typedict.get(etype, {"matched": 0, "pred_total": 0, "gold_total": 0})
            m, p, g = s["matched"], s["pred_total"], s["gold_total"]
            prec = m / p if p else 0
            rec = m / g if g else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            return f"{prec*100:7.2f}", f"{rec*100:7.2f}", f"{f1*100:8.2f}"

        pi, ri, fi = compute(stats_dict["Arg_I"])
        pc, rc, fc = compute(stats_dict["Arg_C"])

        print(f"{etype:25s} | {pi} | {ri} | {fi} | {pc} | {rc} | {fc}")
    print("-" * 90)

"""
domain wise
"""
def compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=None):
    """
    Computes Arg_I and Arg_C F1 scores per domain.
    - Event-type–sensitive
    - Trigger-insensitive
    - Filters Agent, PrimaryObject, SecondaryObject
    """
    from collections import defaultdict

    stats = {
        "Arg_I": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
        "Arg_C": defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0}),
    }

    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        domain = wnd_ids.get(sent_id, "unknown").split("_")[0]
        g_list = [x for x in gold_roles[sent_id] if x[1][2] not in exclude_roles]
        p_list = [x for x in pred_roles.get(sent_id, []) if x[1][2] not in exclude_roles]

        # --- Arg_I: match span and event_type ---
        g_I = [(x[0][2], (x[1][0], x[1][1])) for x in g_list]  # (event_type, span)
        p_I = [(x[0][2], (x[1][0], x[1][1])) for x in p_list]
        stats["Arg_I"][domain]["gold_total"] += len(g_I)
        stats["Arg_I"][domain]["pred_total"] += len(p_I)
        matched = set()
        for i_p, (evt_p, span_p) in enumerate(p_I):
            for i_g, (evt_g, span_g) in enumerate(g_I):
                if i_g in matched:
                    continue
                if evt_p == evt_g and (overlap_fn(span_p, span_g) if overlap_fn else span_p == span_g):
                    stats["Arg_I"][domain]["matched"] += 1
                    matched.add(i_g)
                    break

        # --- Arg_C: match span, event_type, and role ---
        g_C = [(x[0][2], (x[1][0], x[1][1]), x[1][2]) for x in g_list]  # (event_type, span, role)
        p_C = [(x[0][2], (x[1][0], x[1][1]), x[1][2]) for x in p_list]
        stats["Arg_C"][domain]["gold_total"] += len(g_C)
        stats["Arg_C"][domain]["pred_total"] += len(p_C)
        matched = set()
        for i_p, (evt_p, span_p, role_p) in enumerate(p_C):
            for i_g, (evt_g, span_g, role_g) in enumerate(g_C):
                if i_g in matched:
                    continue
                if evt_p == evt_g and role_p == role_g and (overlap_fn(span_p, span_g) if overlap_fn else span_p == span_g):
                    stats["Arg_C"][domain]["matched"] += 1
                    matched.add(i_g)
                    break

    return stats

def print_domain_f1_dual(stats_dict, name="EXACT"):
    print(f"\n[DOMAIN-WISE ARGUMENT MATCH - {name}] -------------------------------------------")
    print(f"{'Domain':12s} | {'ArgI-P':>8s} | {'ArgI-R':>8s} | {'ArgI-F1':>8s} | {'ArgC-P':>8s} | {'ArgC-R':>8s} | {'ArgC-F1':>8s}")
    print("-" * 80)

    all_domains = sorted(set(stats_dict["Arg_I"]) | set(stats_dict["Arg_C"]))

    for domain in all_domains:
        def f1_line(stat):
            m, p, g = stat["matched"], stat["pred_total"], stat["gold_total"]
            prec = m / p if p else 0
            rec = m / g if g else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            return f"{prec*100:8.2f}", f"{rec*100:8.2f}", f"{f1*100:8.2f}"

        s_I = stats_dict["Arg_I"].get(domain, {"matched": 0, "pred_total": 0, "gold_total": 0})
        s_C = stats_dict["Arg_C"].get(domain, {"matched": 0, "pred_total": 0, "gold_total": 0})

        pI, rI, f1I = f1_line(s_I)
        pC, rC, f1C = f1_line(s_C)

        print(f"{domain:12s} | {pI} | {rI} | {f1I} | {pC} | {rC} | {f1C}")
    print("-" * 80)





# def main(pred_file, gold_file):
#     pred_data = load_jsonl(pred_file)
#     gold_data = load_jsonl(gold_file)

#     pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
#     gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)

#     print("== Exact Match Evaluation ==")
#     exact_span_match = lambda x, y: x == y
#     exact_trigger_id = compute_f1(pred_trigs, gold_trigs, lambda x, y: x[:2] == y[:2])
#     exact_trigger_cls = compute_f1(pred_trigs, gold_trigs, exact_span_match)
#     exact_arg_id = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][:2] == y[1][:2])
#     exact_arg_cls = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1] == y[1])

#     print("Trigger Identification (Exact): P {:.2f} R {:.2f} F1 {:.2f}".format(*exact_trigger_id[:3]))
#     print("Trigger Classification (Exact): P {:.2f} R {:.2f} F1 {:.2f}".format(*exact_trigger_cls[:3]))
#     print("Argument Identification (Exact): P {:.2f} R {:.2f} F1 {:.2f}".format(*exact_arg_id[:3]))
#     print("Argument Classification (Exact): P {:.2f} R {:.2f} F1 {:.2f}".format(*exact_arg_cls[:3]))

#     print("\n== Overlap Evaluation ==")
#     overlap_trigger_id = compute_f1(pred_trigs, gold_trigs, lambda x, y: spans_overlap(x[:2], y[:2]))
#     overlap_trigger_cls = compute_f1(pred_trigs, gold_trigs, lambda x, y: x[2] == y[2] and spans_overlap(x[:2], y[:2]))
#     overlap_arg_id = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and spans_overlap(x[1][:2], y[1][:2]))
#     overlap_arg_cls = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and spans_overlap(x[1][:2], y[1][:2]))

#     print("Trigger Identification (Overlap): P {:.2f} R {:.2f} F1 {:.2f}".format(*overlap_trigger_id[:3]))
#     print("Trigger Classification (Overlap): P {:.2f} R {:.2f} F1 {:.2f}".format(*overlap_trigger_cls[:3]))
#     print("Argument Identification (Overlap): P {:.2f} R {:.2f} F1 {:.2f}".format(*overlap_arg_id[:3]))
#     print("Argument Classification (Overlap): P {:.2f} R {:.2f} F1 {:.2f}".format(*overlap_arg_cls[:3]))
def print_score(label, score_tuple):
    prec, rec, f1, match, pred_total, gold_total = score_tuple
    print(f"{label:<35} - P: {prec * 100:.2f} ({match}/{pred_total})  "
          f"R: {rec * 100:.2f} ({match}/{gold_total})  F1: {f1 * 100:.2f}")


# def main(pred_file, gold_file):
#     pred_data = load_jsonl(pred_file)
#     gold_data = load_jsonl(gold_file)

#     pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
#     gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)

#     exact_trigger_id = compute_f1(pred_trigs, gold_trigs, lambda x, y: x[:2] == y[:2], label="Trigger_ID_Exact")
#     exact_trigger_cls = compute_f1(pred_trigs, gold_trigs, lambda x, y: x == y, label="Trigger_CLS_Exact")
#     exact_arg_id = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][:2] == y[1][:2], label="Arg_ID_Exact")
#     exact_arg_cls = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1] == y[1], label="Arg_CLS_Exact")

#     overlap_trigger_id = compute_f1(pred_trigs, gold_trigs, lambda x, y: spans_overlap(x[:2], y[:2]), label="Trigger_ID_Overlap")
#     overlap_trigger_cls = compute_f1(pred_trigs, gold_trigs, lambda x, y: x[2] == y[2] and spans_overlap(x[:2], y[:2]), label="Trigger_CLS_Overlap")
#     overlap_arg_id = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and spans_overlap(x[1][:2], y[1][:2]), label="Arg_ID_Overlap")
#     overlap_arg_cls = compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and spans_overlap(x[1][:2], y[1][:2]), label="Arg_CLS_Overlap")

#     print_score("Trigger Identification (Exact)", exact_trigger_id)
#     print_score("Trigger Classification (Exact)", exact_trigger_cls)
#     print_score("Argument Identification (Exact)", exact_arg_id)
#     print_score("Argument Classification (Exact)", exact_arg_cls)
#     print()
#     print_score("Trigger Identification (Overlap)", overlap_trigger_id)
#     print_score("Trigger Classification (Overlap)", overlap_trigger_cls)
#     print_score("Argument Identification (Overlap)", overlap_arg_id)
#     print_score("Argument Classification (Overlap)", overlap_arg_cls)

from rouge_score import rouge_scorer

def extract_summary_tuple(roles, tokens, sent_id=None):
    parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
    trigger = ""
    etype = "UNKNOWN"

    if isinstance(tokens, str):
        tokens = []

    try:
        if not roles:
            return ("", "", "", ""), "UNKNOWN"

        if not isinstance(roles[0], tuple) or len(roles[0]) != 2:
            return ("", "", "", ""), "UNKNOWN"

        trigger_span = roles[0][0]
        if not (isinstance(trigger_span, tuple) and len(trigger_span) == 3):
            return ("", "", "", ""), "UNKNOWN"

        if not isinstance(trigger_span[0], int) or not isinstance(trigger_span[1], int):
            return ("", "", "", ""), "UNKNOWN"

        trigger = " ".join(tokens[trigger_span[0]:trigger_span[1]])
        etype = trigger_span[2]

    except Exception:
        return ("", "", "", ""), "UNKNOWN"

    for item in roles:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        (_, _, _), (s, e, role) = item
        try:
            if role in parts and not parts[role]:
                if 0 <= s < len(tokens) and 0 < e <= len(tokens):
                    parts[role] = " ".join(tokens[s:e])
        except Exception:
            continue

    return (parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]), etype


def compute_rougeL_overall(gold_roles_all, pred_roles_all, token_lists, sent_ids):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    p_scores, r_scores, f1_scores = [], [], []

    for gold_roles, pred_roles, tokens, sent_id in zip(gold_roles_all, pred_roles_all, token_lists, sent_ids):
        g_text = " ".join(extract_summary_tuple(gold_roles, tokens, sent_id)[0]).strip()
        p_text = " ".join(extract_summary_tuple(pred_roles, tokens, sent_id)[0]).strip()
        if not g_text and not p_text:
            continue
        score = scorer.score(g_text, p_text)['rougeL']
        p_scores.append(score.precision)
        r_scores.append(score.recall)
        f1_scores.append(score.fmeasure)

    return (
        sum(p_scores) / len(p_scores) if p_scores else 0.0,
        sum(r_scores) / len(r_scores) if r_scores else 0.0,
        sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    )

def compute_rougeL_eventtype(gold_roles_all, pred_roles_all, token_lists, sent_ids):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for gold_roles, pred_roles, tokens, sent_id in zip(gold_roles_all, pred_roles_all, token_lists, sent_ids):
        g_tuple, etype = extract_summary_tuple(gold_roles, tokens, sent_id)
        p_tuple, _ = extract_summary_tuple(pred_roles, tokens, sent_id)
        g_text = " ".join(g_tuple).strip()
        p_text = " ".join(p_tuple).strip()
        if not g_text and not p_text:
            continue
        score = scorer.score(g_text, p_text)['rougeL']
        stats[etype].append((score.precision, score.recall, score.fmeasure))

    return {
        etype: (
            sum(x[0] for x in scores) / len(scores),
            sum(x[1] for x in scores) / len(scores),
            sum(x[2] for x in scores) / len(scores),
        )
        for etype, scores in stats.items()
    }

def compute_rougeL_domain(gold_roles_all, pred_roles_all, token_lists, wnd_ids):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for gold_roles, pred_roles, tokens, wnd_id in zip(gold_roles_all, pred_roles_all, token_lists, wnd_ids):
        domain = wnd_id.strip().split("_")[0]
        g_tuple, _ = extract_summary_tuple(gold_roles, tokens, wnd_id)
        p_tuple, _ = extract_summary_tuple(pred_roles, tokens, wnd_id)
        g_text = " ".join(g_tuple).strip()
        p_text = " ".join(p_tuple).strip()
        if not g_text and not p_text:
            continue
        score = scorer.score(g_text, p_text)['rougeL']
        stats[domain].append((score.precision, score.recall, score.fmeasure))

    return {
        domain: (
            sum(x[0] for x in scores) / len(scores),
            sum(x[1] for x in scores) / len(scores),
            sum(x[2] for x in scores) / len(scores),
        )
        for domain, scores in stats.items()
    }
def print_rouge_stats(name, stats_dict):
    print(f"\n[ROUGE-L SUMMARY - {name}] --------------------------------------------------------")
    print(f"{'Key':25s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 50)
    for key in sorted(stats_dict.keys()):
        p, r, f1 = stats_dict[key]
        print(f"{key:25s} | {p*100:6.2f} | {r*100:6.2f} | {f1*100:6.2f}")
    print("-" * 50)









def main(pred_file, gold_file):
    pred_data = load_jsonl(pred_file)
    gold_data = load_jsonl(gold_file)
    tokens = {}
    for entry in gold_data:
        tokens[entry["sent_id"]] = entry["tokens"]
    wnd_ids = {}
    for entry in gold_data:
        wnd_ids[entry["sent_id"]] = entry["sent_id"]

    pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
    gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)
    # print("p",pred_roles,"\n")
    # print("g",gold_roles)

    # print("\n[ROUGE-L Tuple]")
    # p_r, r_r, f1_r = compute_rouge_L_oneie(gold_roles, pred_roles, tokens)
    # print(f"Precision: {p_r:.4f}, Recall: {r_r:.4f}, F1: {f1_r:.4f}")

    sent_ids = list(gold_roles.keys())
    gold_roles_all = [gold_roles[sid] for sid in sent_ids]
    pred_roles_all = [pred_roles.get(sid, []) for sid in sent_ids]
    token_lists = [tokens[sid] for sid in sent_ids]
    # Overall
    p, r, f1 = compute_rougeL_overall(gold_roles_all, pred_roles_all, token_lists, sent_ids)
    print(f"\n[ROUGE-L OVERALL] P: {p:.2%}, R: {r:.2%}, F1: {f1:.2%}")

    # Event-type
    etype_stats = compute_rougeL_eventtype(gold_roles_all, pred_roles_all, token_lists, sent_ids)
    print_rouge_stats("Event Type", etype_stats)

    # Domain-wise
    domain_stats = compute_rougeL_domain(gold_roles_all, pred_roles_all, token_lists, wnd_ids)
    print_rouge_stats("Domain", domain_stats)





    # === Exact Match ===
    # print_score("Trigger Identification (Exact)", compute_f1(pred_trigs, gold_trigs, lambda x, y: x[:2] == y[:2], "Trigger_ID_Exact"))
    # print_score("Trigger Classification (Exact)", compute_f1(pred_trigs, gold_trigs, lambda x, y: x == y, "Trigger_CLS_Exact"))
    print_score("Argument Identification (Exact)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][:2] == y[1][:2], "Arg_ID_Exact"))
    print_score("Argument Classification (Exact)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1] == y[1], "Arg_CLS_Exact"))
    # === Role-wise Scores ===
    stats_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=None)
    print_rolewise_f1_dual(stats_C, name="Exact")
    # === Event-type-wise Scores ===
    etype_stats = compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=None)
    print_eventtype_stats_extended("Exact", etype_stats)
    # === Domain-wise Scores ===
    domain_stats = compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=None)
    print_domain_f1_dual(domain_stats, name="Exact")
    print()

    # === Simple Overlap ===
    # print_score("Trigger Identification (Overlap)", compute_f1(pred_trigs, gold_trigs, lambda x, y: spans_overlap(x[:2], y[:2]), "Trigger_ID_Overlap"))
    # print_score("Trigger Classification (Overlap)", compute_f1(pred_trigs, gold_trigs, lambda x, y: x[2] == y[2] and spans_overlap(x[:2], y[:2]), "Trigger_CLS_Overlap"))
    print_score("Argument Identification (Overlap)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and spans_overlap(x[1][:2], y[1][:2]), "Arg_ID_Overlap"))
    print_score("Argument Classification (Overlap)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and spans_overlap(x[1][:2], y[1][:2]), "Arg_CLS_Overlap"))
    # === Role-wise Scores ===
    stats_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=spans_overlap)
    print_rolewise_f1_dual(stats_C, name="Overlap")
    # === Event-type-wise Scores ===
    etype_stats = compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=spans_overlap)
    print_eventtype_stats_extended("Overlap", etype_stats)
    # === Domain-wise Scores ===
    domain_stats = compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=spans_overlap)
    print_domain_f1_dual(domain_stats, name="Overlap")
    print()
    # === SciREX Overlap ===
    # print_score("Trigger Identification (SciREX)", compute_f1(pred_trigs, gold_trigs, lambda x, y: scirex_overlap(x[:2], y[:2]), "Trigger_ID_SciREX"))
    # print_score("Trigger Classification (SciREX)", compute_f1(pred_trigs, gold_trigs, lambda x, y: x[2] == y[2] and scirex_overlap(x[:2], y[:2]), "Trigger_CLS_SciREX"))
    print_score("Argument Identification (SciREX)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and scirex_overlap(x[1][:2], y[1][:2]), "Arg_ID_SciREX"))
    print_score("Argument Classification (SciREX)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and scirex_overlap(x[1][:2], y[1][:2]), "Arg_CLS_SciREX"))
    # === Role-wise Scores ===
    stats_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=scirex_overlap)
    print_rolewise_f1_dual(stats_C, name="SciREX")
    # === Event-type-wise Scores ===
    etype_stats = compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=lambda p, g: scirex_overlap(p, g))
    print_eventtype_stats_extended("SciREX", etype_stats)
    # === Domain-wise Scores ===
    domain_stats = compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=lambda p, g: scirex_overlap(p, g))
    print_domain_f1_dual(domain_stats, name="SciREX")
    print()
    # === IoU Overlap ===
    # print_score("Trigger Identification (IoU)", compute_f1(pred_trigs, gold_trigs, lambda x, y: iou_overlap(x[:2], y[:2]), "Trigger_ID_IoU"))
    # print_score("Trigger Classification (IoU)", compute_f1(pred_trigs, gold_trigs, lambda x, y: x[2] == y[2] and iou_overlap(x[:2], y[:2]), "Trigger_CLS_IoU"))
    print_score("Argument Identification (IoU)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and iou_overlap(x[1][:2], y[1][:2]), "Arg_ID_IoU"))
    print_score("Argument Classification (IoU)", compute_f1(pred_roles, gold_roles, lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and iou_overlap(x[1][:2], y[1][:2]), "Arg_CLS_IoU"))
    # === Role-wise Scores ===
    stats_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=iou_overlap)
    print_rolewise_f1_dual(stats_C, name="IoU")
    # === Event-type-wise Scores ===
    etype_stats = compute_eventtype_f1_extended(pred_roles, gold_roles, overlap_fn=iou_overlap)
    print_eventtype_stats_extended("IoU", etype_stats)
    # === Domain-wise Scores ===
    domain_stats = compute_domain_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=iou_overlap)
    print_domain_f1_dual(domain_stats, name="IoU")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL')
    parser.add_argument('--gold', required=True, help='Path to gold JSONL')
    args = parser.parse_args()
    main(args.pred, args.gold)

# python EM_overlap_eval.py --pred oneie_preds_degree_format.json --gold processed_data/mydata/test.oneie.json




# def compute_tuple_f1_avg_oneie(gold_roles, pred_roles, tokens, match_fn):
#     total_p, total_r, total_f1, count = 0, 0, 0, 0

#     for sent_id in gold_roles:
#         g_roles = gold_roles.get(sent_id, [])
#         p_roles = pred_roles.get(sent_id, [])
#         toks = tokens[sent_id]

#         def extract(roles):
#             parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
#             trigger = ""
#             roles = sorted(roles, key=lambda x: (x[1][0], x[1][1]))
#             for trig, arg in roles:
#                 ts, te, _ = trig
#                 rs, re, role = arg
#                 if not trigger:
#                     trigger = " ".join(toks[ts:te]).strip()
#                 if role in parts and not parts[role]:
#                     parts[role] = " ".join(toks[rs:re]).strip()
#             return parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]

#         g = extract(g_roles)
#         p = extract(p_roles)

#         if all(not x for x in g) and all(not x for x in p):
#             continue

#         matches = [match_fn(gv, pv) if gv and pv else False for gv, pv in zip(g, p)]
#         match_count = sum(matches)
#         pred_filled = sum(1 for x in p if x)
#         gold_filled = sum(1 for x in g if x)

#         if pred_filled == 0 or gold_filled == 0:
#             continue

#         prec = match_count / pred_filled
#         rec = match_count / gold_filled
#         f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

#         total_p += prec
#         total_r += rec
#         total_f1 += f1
#         count += 1

#     return total_p / count, total_r / count, total_f1 / count

# def compute_tuple_f1_global_oneie(gold_roles, pred_roles, tokens, match_fn):
#     total_matched = 0
#     total_pred = 0
#     total_gold = 0

#     for sent_id in gold_roles:
#         g_roles = gold_roles.get(sent_id, [])
#         p_roles = pred_roles.get(sent_id, [])
#         toks = tokens[sent_id]

#         def extract(roles):
#             parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
#             trigger = ""
#             roles = sorted(roles, key=lambda x: (x[1][0], x[1][1]))
#             for trig, arg in roles:
#                 ts, te, _ = trig
#                 rs, re, role = arg
#                 if not trigger:
#                     trigger = " ".join(toks[ts:te]).strip()
#                 if role in parts and not parts[role]:
#                     parts[role] = " ".join(toks[rs:re]).strip()
#             return parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]

#         g = extract(g_roles)
#         p = extract(p_roles)

#         for gv, pv in zip(g, p):
#             if gv:
#                 total_gold += 1
#             if pv:
#                 total_pred += 1
#             if gv and pv and match_fn(gv, pv):
#                 total_matched += 1

#     precision = total_matched / total_pred if total_pred else 0.0
#     recall = total_matched / total_gold if total_gold else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
#     return precision, recall, f1
# per event
    # print("\n[Tuple Match - Per Event Avg - Exact Match]")
    # p2, r2, f12 = compute_tuple_f1_avg_oneie(gold_roles, pred_roles, tokens, exact_match)
    # print(f"Precision: {p2:.4f}, Recall: {r2:.4f}, F1: {f12:.4f}")

    # print("[Tuple Match - Per Event Avg - One Word Overlap]")
    # p1, r1, f11 = compute_tuple_f1_avg_oneie(gold_roles, pred_roles, tokens, one_word_overlap)
    # print(f"Precision: {p1:.4f}, Recall: {r1:.4f}, F1: {f11:.4f}")

    # print("[Tuple Match - Per Event Avg - IoU > 0.5]")
    # p3, r3, f13 = compute_tuple_f1_avg_oneie(gold_roles, pred_roles, tokens, iou_match)
    # print(f"Precision: {p3:.4f}, Recall: {r3:.4f}, F1: {f13:.4f}")

    # ### global
    # print("\n[Tuple Match - Global Count - Exact Match]")
    # pg2, rg2, f1g2 = compute_tuple_f1_global_oneie(gold_roles, pred_roles, tokens, exact_match)
    # print(f"Precision: {pg2:.4f}, Recall: {rg2:.4f}, F1: {f1g2:.4f}")

    # print("[Tuple Match - Global Count - One Word Overlap]")
    # pg, rg, f1g = compute_tuple_f1_global_oneie(gold_roles, pred_roles, tokens, one_word_overlap)
    # print(f"Precision: {pg:.4f}, Recall: {rg:.4f}, F1: {f1g:.4f}")

    # print("[Tuple Match - Global Count - IoU > 0.5]")
    # pg3, rg3, f1g3 = compute_tuple_f1_global_oneie(gold_roles, pred_roles, tokens, iou_match)
    # print(f"Precision: {pg3:.4f}, Recall: {rg3:.4f}, F1: {f1g3:.4f}")
    # print("")


    