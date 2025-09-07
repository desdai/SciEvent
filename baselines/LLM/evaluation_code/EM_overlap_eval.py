import json
# import pandas as pd
from collections import defaultdict
from rouge_score import rouge_scorer
import sys, contextlib

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()

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
        event_id = entry["event_code"]
        triggers = []
        roles = []

        for item in entry["Argument"]:
            if len(item) == 3 and item[2] == "trigger":
                trigger_start, trigger_end = item[0], item[1]
                triggers.append((trigger_start, trigger_end, "trigger"))
            elif len(item) == 3:
                arg_start, arg_end, role = item
                if triggers:
                    linked_trigger = triggers[0]
                    roles.append((linked_trigger, (arg_start, arg_end, role)))
                else:
                    print(f"[WARN] No trigger found for argument in {event_id}")
        trigger_dict[event_id] = triggers
        role_dict[event_id] = roles

    return trigger_dict, role_dict

def compute_f1_argument_only(pred_dict, gold_dict, match_fn, label=""):
    excluded_roles = {"Agent", "Object:Primary Object", "Object:Secondary Object"}

    total_matched = 0
    total_pred = 0
    total_gold = 0

    for event_code in gold_dict:
        pred_list = pred_dict.get(event_code, [])
        gold_list = gold_dict[event_code]

        pred_args = [
            x for x in pred_list
            if isinstance(x, (list, tuple)) and len(x) == 2 and x[1][2] not in excluded_roles
        ]
        gold_args = [
            x for x in gold_list
            if isinstance(x, (list, tuple)) and len(x) == 2 and x[1][2] not in excluded_roles
        ]

        pred_args = [x for x in pred_args if x[1][0] >= 0 and x[1][1] > x[1][0]]
        gold_args = [x for x in gold_args if x[1][0] >= 0 and x[1][1] > x[1][0]]

        used_gold = set()
        matched = 0

        for p in pred_args:
            for idx, g in enumerate(gold_args):
                if idx in used_gold:
                    continue
                if match_fn(p, g):
                    matched += 1
                    used_gold.add(idx)
                    break

        total_matched += matched
        total_pred += len(pred_args)
        total_gold += len(gold_args)

    precision = total_matched / total_pred if total_pred > 0 else 0
    recall = total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, total_matched, total_pred, total_gold

def print_score(label, score_tuple, results_list=None):
    prec, rec, f1, match, pred_total, gold_total = score_tuple
    print(f"{label:<30} - P: {prec:.4f} ({match}/{pred_total})  R: {rec:.4f} ({match}/{gold_total})  F1: {f1:.4f}")
    if results_list is not None:
        results_list.append({
            "Metric": label,
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1": f"{f1:.4f}",
            "Matched": match,
            "Predicted Total": pred_total,
            "Gold Total": gold_total
        })
"""
Role-wise
"""
def compute_rolewise_f1_dual(pred_roles, gold_roles, match_fn=None):
    from collections import defaultdict
    exclude_roles = {"Agent", "Object:Primary Object", "Object:Secondary Object"}

    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})

    for sent_id in gold_roles:
        # Filter invalid roles/spans
        g_list = [r for r in gold_roles[sent_id] if r[1][2] not in exclude_roles and r[1][0] >= 0 and r[1][1] > r[1][0]]
        p_list = [r for r in pred_roles.get(sent_id, []) if r[1][2] not in exclude_roles and r[1][0] >= 0 and r[1][1] > r[1][0]]

        # Group by role
        g_by_role = defaultdict(list)
        p_by_role = defaultdict(list)
        for ((_, _, et), (s, e, r)) in g_list:
            g_by_role[r].append((s, e, r, et))
        for ((_, _, et), (s, e, r)) in p_list:
            p_by_role[r].append((s, e, r, et))

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
                    if p[2] == g[2] and p[3] == g[3] and (match_fn(p[:2], g[:2]) if match_fn else p[:2] == g[:2]):
                        matched.add(i_p)
                        stats_C[role]["matched"] += 1
                        break

    return stats_C

def print_rolewise_f1_dual(stats_C, name="CLASSIFICATION ONLY"):
    print(f"\n[ROLE-WISE ARGUMENT CLASSIFICATION - {name}] -------------------------------")
    print(f"{'Role':25s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 60)

    for role in sorted(stats_C):
        s = stats_C[role]
        p_total = s["pred_total"]
        g_total = s["gold_total"]
        matched = s["matched"]

        prec = matched / p_total if p_total else 0
        rec = matched / g_total if g_total else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"{role:25s} | {prec*100:6.2f} | {rec*100:6.2f} | {f1*100:6.2f}")
    print("-" * 60)

"""
Event type-wise
"""
def compute_eventtype_f1(pred_data, gold_data, pred_trigs, gold_trigs, pred_roles, gold_roles, match_fn_span):
    """
    Computes event-type–wise argument identification and classification scores.
    Excludes trigger evaluation and filters Agent, Primary/Secondary Object roles.
    """
    from collections import defaultdict

    exclude_roles = {"Agent", "Object:Primary Object", "Object:Secondary Object"}

    stats = defaultdict(lambda: {
        "Arg_I": {"matched": 0, "pred_total": 0, "gold_total": 0},
        "Arg_C": {"matched": 0, "pred_total": 0, "gold_total": 0}
    })

    # event_id -> event_type from gold
    eventtype_dict = {item["event_code"]: item.get("event_type", "Unknown") for item in gold_data}

    for event_id in eventtype_dict:
        event_type = eventtype_dict[event_id]

        # ---- Get filtered arguments
        pred_args = [
            x for x in pred_roles.get(event_id, [])
            if x[1][2] not in exclude_roles and x[1][0] >= 0 and x[1][1] > x[1][0]
        ]
        gold_args = [
            x for x in gold_roles.get(event_id, [])
            if x[1][2] not in exclude_roles and x[1][0] >= 0 and x[1][1] > x[1][0]
        ]

        # ---- Argument Identification (span match only, same event type)
        matched_id = set()
        used_gold_id = set()
        for p in pred_args:
            for idx, g in enumerate(gold_args):
                if idx in used_gold_id:
                    continue
                if p[0][2] == g[0][2] and match_fn_span(p[1][:2], g[1][:2]):
                    matched_id.add((p, g))
                    used_gold_id.add(idx)
                    break

        stats[event_type]["Arg_I"]["matched"] += len(matched_id)
        stats[event_type]["Arg_I"]["pred_total"] += len(pred_args)
        stats[event_type]["Arg_I"]["gold_total"] += len(gold_args)

        # ---- Argument Classification (span + role + event type match)
        matched_cls = set()
        used_gold_cls = set()
        for p in pred_args:
            for idx, g in enumerate(gold_args):
                if idx in used_gold_cls:
                    continue
                if (
                    p[0][2] == g[0][2] and
                    p[1][2] == g[1][2] and
                    match_fn_span(p[1][:2], g[1][:2])
                ):
                    matched_cls.add((p, g))
                    used_gold_cls.add(idx)
                    break

        stats[event_type]["Arg_C"]["matched"] += len(matched_cls)
        stats[event_type]["Arg_C"]["pred_total"] += len(pred_args)
        stats[event_type]["Arg_C"]["gold_total"] += len(gold_args)

    return stats

def print_eventtype_f1(stats, title="EVENT-TYPE-WISE ARGUMENT MATCH"):
    print(f"\n[{title}] ------------------------------------------------------------------------------")
    print(f"{'Event Type':30} | {'ArgI-P':>7} | {'ArgI-R':>7} | {'ArgI-F1':>7} | {'ArgC-P':>7} | {'ArgC-R':>7} | {'ArgC-F1':>7}")
    print(f"{'':30} | {'(m/p/g)':>7} | {'(m/p/g)':>7} | {'(m/p/g)':>7} | {'(m/p/g)':>7} | {'(m/p/g)':>7} | {'(m/p/g)':>7}")
    print("-" * 90)

    for etype in sorted(stats.keys()):
        row = f"{etype:30} |"
        for key in ["Arg_I", "Arg_C"]:
            matched = stats[etype][key]["matched"]
            pred_total = stats[etype][key]["pred_total"]
            gold_total = stats[etype][key]["gold_total"]
            prec = matched / pred_total if pred_total > 0 else 0
            rec = matched / gold_total if gold_total > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            row += f" {prec*100:7.2f} | {rec*100:7.2f} | {f1*100:7.2f} |"
        print(row)

        # # Optional: print counts (comment out if you want P/R/F1 only)
        # count_row = f"{'':30} |"
        # for key in ["Arg_I", "Arg_C"]:
        #     matched = stats[etype][key]["matched"]
        #     pred_total = stats[etype][key]["pred_total"]
        #     gold_total = stats[etype][key]["gold_total"]
        #     count_row += f" ({matched}/{pred_total}/{gold_total}) |" * 3
        # print(count_row)
    print("-" * 90)

def compute_domain_f1(pred_data, gold_data, pred_trigs, gold_trigs, pred_roles, gold_roles, match_fn_span):
    from collections import defaultdict

    exclude_roles = {"Agent", "Object:Primary Object", "Object:Secondary Object"}

    stats = defaultdict(lambda: {
        "Arg_I": {"matched": 0, "pred_total": 0, "gold_total": 0},
        "Arg_C": {"matched": 0, "pred_total": 0, "gold_total": 0}
    })

    def get_domain(event_code):
        return event_code.split("_")[0] if "_" in event_code else "Unknown"

    for event in gold_data:
        event_id = event["event_code"]
        domain = get_domain(event_id)

        pred_args = [
            x for x in pred_roles.get(event_id, [])
            if x[1][2] not in exclude_roles and x[1][0] >= 0 and x[1][1] > x[1][0]
        ]
        gold_args = [
            x for x in gold_roles.get(event_id, [])
            if x[1][2] not in exclude_roles and x[1][0] >= 0 and x[1][1] > x[1][0]
        ]

        # Arg I
        matched_id = set()
        used_gold_id = set()
        for p in pred_args:
            for idx, g in enumerate(gold_args):
                if idx in used_gold_id:
                    continue
                if p[0][2] == g[0][2] and match_fn_span(p[1][:2], g[1][:2]):
                    matched_id.add((p, g))
                    used_gold_id.add(idx)
                    break
        stats[domain]["Arg_I"]["matched"] += len(matched_id)
        stats[domain]["Arg_I"]["pred_total"] += len(pred_args)
        stats[domain]["Arg_I"]["gold_total"] += len(gold_args)

        # Arg C
        matched_cls = set()
        used_gold_cls = set()
        for p in pred_args:
            for idx, g in enumerate(gold_args):
                if idx in used_gold_cls:
                    continue
                if (
                    p[0][2] == g[0][2] and
                    p[1][2] == g[1][2] and
                    match_fn_span(p[1][:2], g[1][:2])
                ):
                    matched_cls.add((p, g))
                    used_gold_cls.add(idx)
                    break
        stats[domain]["Arg_C"]["matched"] += len(matched_cls)
        stats[domain]["Arg_C"]["pred_total"] += len(pred_args)
        stats[domain]["Arg_C"]["gold_total"] += len(gold_args)

    return stats

def print_domain_f1(stats, title="DOMAIN-WISE ARGUMENT MATCH"):
    print(f"\n[{title}] ------------------------------------------------------------------------------")
    print(f"{'Domain':25s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>7s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>7s}")
    print(f"{'':25s} | {'(m/p/g)':>7s} | {'(m/p/g)':>7s} | {'(m/p/g)':>7s} | {'(m/p/g)':>7s} | {'(m/p/g)':>7s} | {'(m/p/g)':>7s}")
    print("-" * 90)

    for domain in sorted(stats.keys()):
        row = f"{domain:25s} |"
        for key in ["Arg_I", "Arg_C"]:
            m = stats[domain][key]["matched"]
            p = stats[domain][key]["pred_total"]
            g = stats[domain][key]["gold_total"]
            prec = m / p if p else 0
            rec = m / g if g else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            row += f" {prec*100:7.2f} | {rec*100:7.2f} | {f1*100:7.2f} |"
        print(row)

    print("-" * 90)


def extract_summary_from_spans(event_id, roles):
    parts = {
        "Agent": "",
        "Object:Primary Object": "",
        "Object:Secondary Object": ""
    }
    trigger_text = ""
    etype = "UNKNOWN"

    if not roles:
        return "", "UNKNOWN", event_id.split("_")[0]

    trigger_span = roles[0][0]
    etype = trigger_span[2] if len(trigger_span) == 3 else "UNKNOWN"
    trigger_text = f"[T:{trigger_span[0]}-{trigger_span[1]}]"

    for (_, (s, e, role)) in roles:
        if role in parts and not parts[role]:
            parts[role] = f"[{role}:{s}-{e}]"

    domain = event_id.split("_")[0]

    summary = " ".join([
        parts["Agent"],
        trigger_text,
        parts["Object:Primary Object"],
        parts["Object:Secondary Object"]
    ]).strip()

    return summary, etype, domain
def compute_rougeL_overall_llm(gold_roles, pred_roles):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    p_scores, r_scores, f1_scores = [], [], []

    for event_id in gold_roles:
        g_summary, _, _ = extract_summary_from_spans(event_id, gold_roles[event_id])
        p_summary, _, _ = extract_summary_from_spans(event_id, pred_roles.get(event_id, []))
        if not g_summary and not p_summary:
            continue
        score = scorer.score(g_summary, p_summary)['rougeL']
        p_scores.append(score.precision)
        r_scores.append(score.recall)
        f1_scores.append(score.fmeasure)

    return (
        sum(p_scores) / len(p_scores) if p_scores else 0.0,
        sum(r_scores) / len(r_scores) if r_scores else 0.0,
        sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    )

def compute_rougeL_eventtype_llm(gold_roles, pred_roles, gold_data):

    # Build mapping from event_id -> true event type
    eventtype_dict = {item["event_code"]: item.get("event_type", "Unknown") for item in gold_data}

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for event_id in gold_roles:
        etype = eventtype_dict.get(event_id, "Unknown")
        g_summary, _, _ = extract_summary_from_spans(event_id, gold_roles[event_id])
        p_summary, _, _ = extract_summary_from_spans(event_id, pred_roles.get(event_id, []))

        if not g_summary and not p_summary:
            continue

        score = scorer.score(g_summary, p_summary)['rougeL']
        stats[etype].append((score.precision, score.recall, score.fmeasure))

    return {
        etype: (
            sum(x[0] for x in scores) / len(scores),
            sum(x[1] for x in scores) / len(scores),
            sum(x[2] for x in scores) / len(scores),
        )
        for etype, scores in stats.items()
    }

def compute_rougeL_domain_llm(gold_roles, pred_roles):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    stats = defaultdict(list)

    for event_id in gold_roles:
        g_summary, _, domain = extract_summary_from_spans(event_id, gold_roles[event_id])
        p_summary, _, _ = extract_summary_from_spans(event_id, pred_roles.get(event_id, []))
        if not g_summary and not p_summary:
            continue
        score = scorer.score(g_summary, p_summary)['rougeL']
        stats[domain].append((score.precision, score.recall, score.fmeasure))

    return {
        domain: (
            sum(x[0] for x in scores) / len(scores),
            sum(x[1] for x in scores) / len(scores),
            sum(x[2] for x in scores) / len(scores),
        )
        for domain, scores in stats.items()
    }
def print_rougeL_stats(name, stats_dict):
    print(f"\n[ROUGE-L SUMMARY - {name}] ------------------------------------------------")
    print(f"{'Key':25s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 50)
    for key in sorted(stats_dict.keys()):
        p, r, f1 = stats_dict[key]
        print(f"{key:25s} | {p*100:6.2f} | {r*100:6.2f} | {f1*100:6.2f}")
    print("-" * 50)

def main(pred_file, gold_file):
    results = []
    pred_data = load_jsonl(pred_file)
    gold_data = load_jsonl(gold_file)

    pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
    gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)

    # Overall
    p, r, f1 = compute_rougeL_overall_llm(gold_roles, pred_roles)
    print(f"[ROUGE-L OVERALL] P: {p*100:.2f}%  R: {r*100:.2f}%  F1: {f1*100:.2f}%")

    # Event Type
    rouge_eventtype = compute_rougeL_eventtype_llm(gold_roles, pred_roles, gold_data)
    print_rougeL_stats("Event Type", rouge_eventtype)

    # Domain
    rouge_domain = compute_rougeL_domain_llm(gold_roles, pred_roles)
    print_rougeL_stats("Domain", rouge_domain)

    
    print("=== Exact Match ===")
    print_score("Argument Identification (Exact)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and x[1][:2] == y[1][:2]
    ))

    print_score("Argument Classification (Exact)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and x[1] == y[1]
    ))
    # Role-wise Argument Classification
    sC = compute_rolewise_f1_dual(pred_roles, gold_roles,
        match_fn=lambda p, g: p == g)
    print_rolewise_f1_dual(sC, "Exact")
    match_fn_span = lambda x, y: x == y

    stats = compute_eventtype_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_eventtype_f1(stats, title="EVENT-TYPE-WISE ARGUMENT MATCH (Exact)")
    stats = compute_domain_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_domain_f1(stats, title="DOMAIN-WISE ARGUMENT MATCH (Exact)")

    print("=== Simple Overlap Match ===")
    print_score("Argument Identification (Overlap)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and spans_overlap(x[1][:2], y[1][:2])
    ))

    print_score("Argument Classification (Overlap)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and spans_overlap(x[1][:2], y[1][:2])
    ))
    sC = compute_rolewise_f1_dual(pred_roles, gold_roles,
        match_fn=spans_overlap)
    print_rolewise_f1_dual(sC, "Overlap")
    match_fn_span = lambda x, y: spans_overlap(x, y)

    stats = compute_eventtype_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_eventtype_f1(stats, title="EVENT-TYPE-WISE ARGUMENT MATCH (Overlap)")
    stats = compute_domain_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_domain_f1(stats, title="DOMAIN-WISE ARGUMENT MATCH (Overlap)")

    print("=== SciREX Overlap (>50% of predicted span) ===")
    print_score("Argument Identification (SciREX)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and scirex_overlap(x[1][:2], y[1][:2])
    ))

    print_score("Argument Classification (SciREX)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and scirex_overlap(x[1][:2], y[1][:2])
    ))
    sC = compute_rolewise_f1_dual(pred_roles, gold_roles,
        match_fn=scirex_overlap)
    print_rolewise_f1_dual(sC, "SciREX")
    match_fn_span = lambda x, y: scirex_overlap(x, y)

    stats = compute_eventtype_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_eventtype_f1(stats, title="EVENT-TYPE-WISE ARGUMENT MATCH (SciREX)")
    stats = compute_domain_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_domain_f1(stats, title="DOMAIN-WISE ARGUMENT MATCH (SciREX)")

    print("=== IoU Overlap (>50% of union) ===")
    print_score("Argument Identification (IoU)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and iou_overlap(x[1][:2], y[1][:2])
    ))

    print_score("Argument Classification (IoU)", compute_f1_argument_only(
        pred_roles, gold_roles,
        lambda x, y: x[0][2] == y[0][2] and x[1][2] == y[1][2] and iou_overlap(x[1][:2], y[1][:2])
    ))
    sC = compute_rolewise_f1_dual(pred_roles, gold_roles,
        match_fn=iou_overlap)
    print_rolewise_f1_dual(sC, "IoU")
    match_fn_span = lambda x, y: iou_overlap(x, y)

    stats = compute_eventtype_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_eventtype_f1(stats, title="EVENT-TYPE-WISE ARGUMENT MATCH (IoU)")
    stats = compute_domain_f1(
        pred_data, gold_data,
        pred_trigs, gold_trigs,
        pred_roles, gold_roles,
        match_fn_span=match_fn_span
    )
    print_domain_f1(stats, title="DOMAIN-WISE ARGUMENT MATCH (IoU)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL')
    parser.add_argument('--gold', required=True, help='Path to gold JSONL')
    parser.add_argument('--out', help='Path to save all printed output (txt).')
    parser.add_argument('--append', action='store_true', help='Append to --out instead of overwrite.')
    args = parser.parse_args()

    if args.out:
        mode = 'a' if args.append else 'w'
        with open(args.out, mode, encoding='utf-8') as f, \
             contextlib.redirect_stdout(Tee(sys.stdout, f)):
            main(args.pred, args.gold)
    else:
        main(args.pred, args.gold)

""" improved """
# Improve zeroshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_2shot_eval/gold_event_level.json

# Improve 5shot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/evetype_true_eval/gold_event_level.json

# pred-event-type-zeroshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/pred_event_level.json --gold SciEvent_data/LLM/data/qwen_2.5_7b/evetype_pred_eval/gold_event_level.json

# Improve zeroshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_2shot_eval/gold_event_level.json

# Improve 5shot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_true_eval/gold_event_level.json

# pred-event-type-zeroshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/pred_event_level.json --gold SciEvent_data/LLM/data/llama_3_8b/improve_pred_eval/gold_event_level.json

"""Distilled ones follow:"""
# Improve zeroshot-distill-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-distill-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/gold_event_level.json

# Improve zeroshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/gold_event_level.json

# Improve oneshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/gold_event_level.json

# Improve 2shot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_2shot_eval/gold_event_level.json

# Improve 5shot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_5shot_eval/gold_event_level.json

# true-event-type-zeroshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_true_eval/gold_event_level.json

# pred-event-type-zeroshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/pred_event_level.json --gold SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_pred_eval/gold_event_level.json

"""GPT-4.1""" 
# zeroshot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/gold_event_level.json

# oneshot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/gold_event_level.json

# 2shot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/2shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/2shot_eval/gold_event_level.json

# 5shot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/5shot_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/5shot_eval/gold_event_level.json

# true-evetype-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/evetype_true_eval/gold_event_level.json

# pred-evetype-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/pred_event_level.json --gold SciEvent_data/LLM/data/gpt_4.1/evetype_pred_eval/gold_event_level.json

"""human""" 
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/human/human_eval/pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/gold_event_level.json

"""subsets"""
# human
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/human/human_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# zeroshot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/zeroshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# oneshot-GPT-4.1
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/gpt_4.1/oneshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve zeroshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve oneshot-distill-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve zeroshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve oneshot-qwen
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/qwen_2.5_7b/improve_oneshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve zeroshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_zeroshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

# Improve oneshot-llama
# python baselines/LLM/evaluation_code/EM_overlap_eval.py --pred SciEvent_data/LLM/data/llama_3_8b/improve_oneshot_eval/filtered_pred_event_level.json --gold SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json
