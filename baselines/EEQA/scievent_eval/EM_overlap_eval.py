import json
from collections import defaultdict
from rouge_score import rouge_scorer


with open("SciEvent_data/DEGREE/all_splits/test.json", "r", encoding="utf-8") as f:
    oneie_wnd_ids = [json.loads(line)["wnd_id"] for line in f]


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

    for i, entry in enumerate(data):
        sent_id = str(i)  # use line index as sentence ID
        s_start = entry.get("s_start", 0)
        triggers = []
        roles = []

        for event in entry.get("event", []):
            if not event or not isinstance(event, list) or not isinstance(event[0], list):
                continue

            # Extract trigger
            trigger_offset, event_type = event[0]
            trigger_start = trigger_offset
            trigger_end = trigger_start + 1  # assume single-token trigger
            triggers.append((trigger_start, trigger_end, event_type))

            # Extract arguments
            for arg in event[1:]:
                if len(arg) == 3:
                    arg_start, arg_end, role = arg
                    roles.append(((trigger_start, trigger_end, event_type), (arg_start, arg_end, role)))
                else:
                    print(f"[WARN] Skipping malformed argument: {arg}")

        trigger_dict[sent_id] = triggers
        role_dict[sent_id] = roles

    # Deduplicate triggers per sentence
    for sent_id in trigger_dict:
        trigger_dict[sent_id] = list(set(trigger_dict[sent_id]))

    return trigger_dict, role_dict


"""
Role-wise (rememeber, EEQA is event type insenstive, only trigger sensitive)
"""
def compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=None):
    from collections import defaultdict

    stats_I = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        g_list = [x for x in gold_roles[sent_id] if x[1][2] not in exclude_roles]
        p_list = [x for x in pred_roles.get(sent_id, []) if x[1][2] not in exclude_roles]

        matched_I = set()
        matched_C = set()

        # --- Argument Identification ---
        for (_, (_, _, prole)) in p_list:
            stats_I[prole]["pred_total"] += 1
        for (_, (_, _, grole)) in g_list:
            stats_I[grole]["gold_total"] += 1

        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_list):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_list):
                if (i_p, i_g) in matched_I:
                    continue
                if ptrig[:2] == gtrig[:2]:  # Trigger span match
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_I[prole]["matched"] += 1
                        matched_I.add((i_p, i_g))
                        break

        # --- Argument Classification ---
        for (_, (_, _, prole)) in p_list:
            stats_C[prole]["pred_total"] += 1
        for (_, (_, _, grole)) in g_list:
            stats_C[grole]["gold_total"] += 1

        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_list):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_list):
                if (i_p, i_g) in matched_C:
                    continue
                if ptrig[:2] == gtrig[:2] and prole == grole:
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_C[prole]["matched"] += 1
                        matched_C.add((i_p, i_g))
                        break

    return stats_I, stats_C


def print_rolewise_f1_dual(stats_I, stats_C, name="EXACT"):
    print(f"\n[ROLE-WISE ARGUMENT MATCH - {name}]")
    print(f"{'Role':20s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>8s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>8s}")
    print("-" * 90)

    all_roles = sorted(set(stats_I) | set(stats_C))
    for role in all_roles:
        sI = stats_I.get(role, {"matched": 0, "pred_total": 0, "gold_total": 0})
        sC = stats_C.get(role, {"matched": 0, "pred_total": 0, "gold_total": 0})

        def calc_prf(stat):
            p = stat["matched"] / stat["pred_total"] if stat["pred_total"] > 0 else 0
            r = stat["matched"] / stat["gold_total"] if stat["gold_total"] > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            return p * 100, r * 100, f1 * 100

        pi, ri, f1i = calc_prf(sI)
        pc, rc, f1c = calc_prf(sC)

        print(f"{role:20s} | {pi:7.2f} | {ri:7.2f} | {f1i:8.2f} | {pc:7.2f} | {rc:7.2f} | {f1c:8.2f}")


"""
event type-wise (rememeber, EEQA is event type insenstive, only trigger sensitive)
"""
def compute_eventtype_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=None):
    """
    Computes Arg I and Arg C scores grouped by gold event type.
    - Trigger-sensitive matching
    - Grouped by event type (from gold_roles trigger)
    - Matched only within same wnd_id (via sent_id)
    - Filters Agent, PrimaryObject, SecondaryObject
    """
    from collections import defaultdict

    stats_I = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        etype_set = {trig[2] for (trig, arg) in gold_roles[sent_id]}  # all event types in this sentence
        if len(etype_set) != 1:
            raise ValueError(f"Multiple event types in one sentence {sent_id}: {etype_set}")
        etype = next(iter(etype_set))
        wnd = wnd_ids[int(sent_id)]

        g_roles = [x for x in gold_roles[sent_id] if x[1][2] not in exclude_roles]
        p_roles = [x for x in pred_roles.get(sent_id, []) if x[1][2] not in exclude_roles]

        stats_I[etype]["gold_total"] += len(g_roles)
        stats_I[etype]["pred_total"] += len(p_roles)
        stats_C[etype]["gold_total"] += len(g_roles)
        stats_C[etype]["pred_total"] += len(p_roles)

        # Argument Identification
        matched_I = set()
        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_roles):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_roles):
                if (i_p, i_g) in matched_I:
                    continue
                if ptrig[:2] == gtrig[:2]:  # trigger-sensitive
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_I[etype]["matched"] += 1
                        matched_I.add((i_p, i_g))
                        break

        # Argument Classification
        matched_C = set()
        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_roles):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_roles):
                if (i_p, i_g) in matched_C:
                    continue
                if ptrig[:2] == gtrig[:2] and prole == grole:
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_C[etype]["matched"] += 1
                        matched_C.add((i_p, i_g))
                        break

    return stats_I, stats_C


def print_eventtype_f1_dual(stats_I, stats_C, name="EXACT"):
    print(f"\n[EVENT-TYPE ARGUMENT MATCH - {name}] ----------------------------------")
    print(f"{'EventType':30s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>8s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>8s}")
    print("-" * 90)

    all_types = sorted(set(stats_I) | set(stats_C))
    for etype in all_types:
        sI = stats_I.get(etype, {"matched": 0, "pred_total": 0, "gold_total": 0})
        sC = stats_C.get(etype, {"matched": 0, "pred_total": 0, "gold_total": 0})

        def calc_prf(stat):
            p = stat["matched"] / stat["pred_total"] if stat["pred_total"] > 0 else 0
            r = stat["matched"] / stat["gold_total"] if stat["gold_total"] > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            return p * 100, r * 100, f1 * 100

        pi, ri, f1i = calc_prf(sI)
        pc, rc, f1c = calc_prf(sC)

        print(f"{etype:30s} | {pi:7.2f} | {ri:7.2f} | {f1i:8.2f} | {pc:7.2f} | {rc:7.2f} | {f1c:8.2f}")


"""
Domain-wise (rememeber, EEQA is event type insenstive, only trigger sensitive)
"""
def compute_domainwise_f1_dual(pred_roles, gold_roles, wnd_ids, overlap_fn=None):
    from collections import defaultdict

    stats_I = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    stats_C = defaultdict(lambda: {"matched": 0, "pred_total": 0, "gold_total": 0})
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    for sent_id in gold_roles:
        domain = wnd_ids[int(sent_id)].split("_")[0]
        g_roles = [x for x in gold_roles[sent_id] if x[1][2] not in exclude_roles]
        p_roles = [x for x in pred_roles.get(sent_id, []) if x[1][2] not in exclude_roles]

        stats_I[domain]["pred_total"] += len(p_roles)
        stats_I[domain]["gold_total"] += len(g_roles)
        stats_C[domain]["pred_total"] += len(p_roles)
        stats_C[domain]["gold_total"] += len(g_roles)

        matched_I = set()
        matched_C = set()

        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_roles):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_roles):
                if (i_p, i_g) in matched_I:
                    continue
                if ptrig[:2] == gtrig[:2]:  # trigger-sensitive
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_I[domain]["matched"] += 1
                        matched_I.add((i_p, i_g))
                        break

        for i_p, (ptrig, (ps, pe, prole)) in enumerate(p_roles):
            for i_g, (gtrig, (gs, ge, grole)) in enumerate(g_roles):
                if (i_p, i_g) in matched_C:
                    continue
                if ptrig[:2] == gtrig[:2] and prole == grole:
                    span_match = overlap_fn((ps, pe), (gs, ge)) if overlap_fn else (ps, pe) == (gs, ge)
                    if span_match:
                        stats_C[domain]["matched"] += 1
                        matched_C.add((i_p, i_g))
                        break

    return stats_I, stats_C

def print_domainwise_f1_dual(stats_I, stats_C, name="EXACT"):
    print(f"\n[DOMAIN-WISE ARGUMENT MATCH - {name}] ----------------------------------")
    print(f"{'Domain':12s} | {'ArgI-P':>7s} | {'ArgI-R':>7s} | {'ArgI-F1':>8s} | {'ArgC-P':>7s} | {'ArgC-R':>7s} | {'ArgC-F1':>8s}")
    print("-" * 80)

    all_domains = sorted(set(stats_I) | set(stats_C))
    for domain in all_domains:
        def calc_p_r_f1(stat):
            p = stat["matched"] / stat["pred_total"] if stat["pred_total"] > 0 else 0
            r = stat["matched"] / stat["gold_total"] if stat["gold_total"] > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            return p * 100, r * 100, f1 * 100

        pi, ri, f1i = calc_p_r_f1(stats_I.get(domain, {}))
        pc, rc, f1c = calc_p_r_f1(stats_C.get(domain, {}))

        print(f"{domain:12s} | {pi:7.2f} | {ri:7.2f} | {f1i:8.2f} | {pc:7.2f} | {rc:7.2f} | {f1c:8.2f}")


def compute_f1(pred_dict, gold_dict):
    """
    Trigger-sensitive, event-type-insensitive F1 for Arg_I and Arg_C under:
    - EM, OVERLAP, IOU, SCIREX
    Agent, PrimaryObject, SecondaryObject are excluded.
    """
    exclude_roles = {"Agent", "PrimaryObject", "SecondaryObject"}

    def filter_roles(d):
        return {
            sid: [x for x in spans if x[1][2] not in exclude_roles]
            for sid, spans in d.items()
        }

    pred = filter_roles(pred_dict)
    gold = filter_roles(gold_dict)

    def spans_overlap(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])

    def iou_overlap(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union > 0.5 if union > 0 else False

    def scirex_overlap(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        pred_len = a[1] - a[0]
        return inter / pred_len > 0.5 if pred_len > 0 else False

    matchers = {
        "EM": {
            "Arg_I": lambda x, y: x[0][:2] == y[0][:2] and x[1][:2] == y[1][:2],
            "Arg_C": lambda x, y: x[0][:2] == y[0][:2] and x[1] == y[1],
        },
        "OVERLAP": {
            "Arg_I": lambda x, y: x[0][:2] == y[0][:2] and spans_overlap(x[1][:2], y[1][:2]),
            "Arg_C": lambda x, y: x[0][:2] == y[0][:2] and x[1][2] == y[1][2] and spans_overlap(x[1][:2], y[1][:2]),
        },
        "IOU": {
            "Arg_I": lambda x, y: x[0][:2] == y[0][:2] and iou_overlap(x[1][:2], y[1][:2]),
            "Arg_C": lambda x, y: x[0][:2] == y[0][:2] and x[1][2] == y[1][2] and iou_overlap(x[1][:2], y[1][:2]),
        },
        "SCIREX": {
            "Arg_I": lambda x, y: x[0][:2] == y[0][:2] and scirex_overlap(x[1][:2], y[1][:2]),
            "Arg_C": lambda x, y: x[0][:2] == y[0][:2] and x[1][2] == y[1][2] and scirex_overlap(x[1][:2], y[1][:2]),
        }
    }

    def f1_score(pred_d, gold_d, match_fn):
        matched = 0
        total_pred = 0
        total_gold = 0
        for sent_id in gold_d:
            pred_list = pred_d.get(sent_id, [])
            gold_list = gold_d.get(sent_id, [])
            total_pred += len(pred_list)
            total_gold += len(gold_list)
            matched_set = set()
            for p in pred_list:
                for g in gold_list:
                    if (p, g) not in matched_set and match_fn(p, g):
                        matched_set.add((p, g))
                        break
            matched += len(matched_set)
        prec = matched / total_pred if total_pred > 0 else 0
        rec = matched / total_gold if total_gold > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1, matched, total_pred, total_gold

    scores = {}
    for mode in matchers:
        for task in ["Arg_I", "Arg_C"]:
            scores[(task, mode)] = f1_score(pred, gold, matchers[mode][task])
    return scores



def print_score(label, score_tuple):
    prec, rec, f1, matched, pred_total, gold_total = score_tuple
    print(f"{label:<35} - "
          f"P: {prec*100:6.2f}  R: {rec*100:6.2f}  F1: {f1*100:6.2f}  "
          f"Count: {matched}/{pred_total}/{gold_total}")

def compute_rougeL_eeqa(gold_roles, pred_roles, tokens):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    p_scores, r_scores, f1_scores = [], [], []

    for sent_id in gold_roles:
        g_roles = gold_roles.get(sent_id, [])
        p_roles = pred_roles.get(sent_id, [])
        toks = tokens[sent_id]

        def extract_tuple_text(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            trigger = ""
            for trig, arg in sorted(roles, key=lambda x: (x[1][0], x[1][1])):
                ts, te, _ = trig
                rs, re, role = arg
                if not trigger:
                    trigger = " ".join(toks[ts:te]).strip()
                if role in parts and not parts[role]:
                    parts[role] = " ".join(toks[rs:re]).strip()
            return parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]

        g = extract_tuple_text(g_roles)
        p = extract_tuple_text(p_roles)
        if all(v == "" for v in g) and all(v == "" for v in p):
            continue

        g_str, p_str = " ".join(g).strip(), " ".join(p).strip()
        score = scorer.score(g_str, p_str)["rougeL"]
        p_scores.append(score.precision)
        r_scores.append(score.recall)
        f1_scores.append(score.fmeasure)

    return sum(p_scores)/len(p_scores), sum(r_scores)/len(r_scores), sum(f1_scores)/len(f1_scores)

from collections import defaultdict

def compute_rougeL_eeqa_eventtype(gold_roles, pred_roles, tokens):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = defaultdict(list)

    for sent_id in gold_roles:
        g_roles = gold_roles.get(sent_id, [])
        p_roles = pred_roles.get(sent_id, [])
        toks = tokens[sent_id]

        if not g_roles:
            continue
        etype = g_roles[0][0][2]

        def extract_tuple_text(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            trigger = ""
            for trig, arg in sorted(roles, key=lambda x: (x[1][0], x[1][1])):
                ts, te, _ = trig
                rs, re, role = arg
                if not trigger:
                    trigger = " ".join(toks[ts:te]).strip()
                if role in parts and not parts[role]:
                    parts[role] = " ".join(toks[rs:re]).strip()
            return parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]

        g = extract_tuple_text(g_roles)
        p = extract_tuple_text(p_roles)
        if all(v == "" for v in g) and all(v == "" for v in p):
            continue

        g_str, p_str = " ".join(g).strip(), " ".join(p).strip()
        score = scorer.score(g_str, p_str)["rougeL"]
        scores[etype].append((score.precision, score.recall, score.fmeasure))

    return {
        etype: (
            sum(x[0] for x in lst)/len(lst),
            sum(x[1] for x in lst)/len(lst),
            sum(x[2] for x in lst)/len(lst)
        )
        for etype, lst in scores.items()
    }

def compute_rougeL_eeqa_domain(gold_roles, pred_roles, tokens, wnd_ids):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = defaultdict(list)

    for sent_id in gold_roles:
        g_roles = gold_roles.get(sent_id, [])
        p_roles = pred_roles.get(sent_id, [])
        toks = tokens[sent_id]
        domain = wnd_ids[int(sent_id)].split("_")[0]

        def extract_tuple_text(roles):
            parts = {"Agent": "", "PrimaryObject": "", "SecondaryObject": ""}
            trigger = ""
            for trig, arg in sorted(roles, key=lambda x: (x[1][0], x[1][1])):
                ts, te, _ = trig
                rs, re, role = arg
                if not trigger:
                    trigger = " ".join(toks[ts:te]).strip()
                if role in parts and not parts[role]:
                    parts[role] = " ".join(toks[rs:re]).strip()
            return parts["Agent"], trigger, parts["PrimaryObject"], parts["SecondaryObject"]

        g = extract_tuple_text(g_roles)
        p = extract_tuple_text(p_roles)
        if all(v == "" for v in g) and all(v == "" for v in p):
            continue

        g_str, p_str = " ".join(g).strip(), " ".join(p).strip()
        score = scorer.score(g_str, p_str)["rougeL"]
        scores[domain].append((score.precision, score.recall, score.fmeasure))

    return {
        domain: (
            sum(x[0] for x in lst)/len(lst),
            sum(x[1] for x in lst)/len(lst),
            sum(x[2] for x in lst)/len(lst)
        )
        for domain, lst in scores.items()
    }

def print_rouge_stats(name, stats_dict):
    print(f"\n[ROUGE-L SUMMARY - {name}] ----------------------------------------")
    print(f"{'Key':25s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 50)
    for key in sorted(stats_dict):
        p, r, f1 = stats_dict[key]
        print(f"{key:25s} | {p*100:6.2f} | {r*100:6.2f} | {f1*100:6.2f}")
    print("-" * 50)

def main(pred_file, gold_file):
    pred_data = load_jsonl(pred_file)
    gold_data = load_jsonl(gold_file)

    pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
    gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)

    tokens_all_dict = {str(i): entry["sentence"] for i, entry in enumerate(gold_data)}

    scores = compute_f1(pred_roles, gold_roles)
    # === Rouge L ===
    p_rouge, r_rouge, f1_rouge = compute_rougeL_eeqa(gold_roles, pred_roles, tokens_all_dict)
    print(f"\n[ROUGE-L OVERALL] P: {p_rouge*100:.2f}%, R: {r_rouge*100:.2f}%, F1: {f1_rouge*100:.2f}%")

    rouge_event_stats = compute_rougeL_eeqa_eventtype(gold_roles, pred_roles, tokens_all_dict)
    print_rouge_stats("Event Type", rouge_event_stats)

    rouge_domain_stats = compute_rougeL_eeqa_domain(gold_roles, pred_roles, tokens_all_dict, oneie_wnd_ids)
    print_rouge_stats("Domain", rouge_domain_stats)
    print("")

    # === Exact Match ===
    print_score("Argument Identification (Exact)", scores[("Arg_I", "EM")])
    print_score("Argument Classification (Exact)", scores[("Arg_C", "EM")])
    stats_em_I, stats_em_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=None)
    print_rolewise_f1_dual(stats_em_I, stats_em_C, name="EM")
    stats_em_I, stats_em_C = compute_eventtype_f1_dual(pred_roles, gold_roles, oneie_wnd_ids)
    print_eventtype_f1_dual(stats_em_I, stats_em_C, name="EM")
    s_em_I, s_em_C = compute_domainwise_f1_dual(pred_roles, gold_roles, oneie_wnd_ids)
    print_domainwise_f1_dual(s_em_I, s_em_C, name="EM")
    print("")
    # === Overlap Match ===
    print_score("Argument Identification (Overlap)", scores[("Arg_I", "OVERLAP")])
    print_score("Argument Classification (Overlap)", scores[("Arg_C", "OVERLAP")])
    stats_overlap_I, stats_overlap_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=spans_overlap)
    print_rolewise_f1_dual(stats_overlap_I, stats_overlap_C, name="OVERLAP")
    stats_overlap_I, stats_overlap_C = compute_eventtype_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=spans_overlap)
    print_eventtype_f1_dual(stats_overlap_I, stats_overlap_C, name="OVERLAP")
    s_overlap_I, s_overlap_C = compute_domainwise_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=spans_overlap)
    print_domainwise_f1_dual(s_overlap_I, s_overlap_C, name="OVERLAP")
    print("")
    # === Overlap Match ===
    print_score("Argument Identification (SCIREX)", scores[("Arg_I", "SCIREX")])
    print_score("Argument Classification (SCIREX)", scores[("Arg_C", "SCIREX")])
    stats_scirex_I, stats_scirex_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=scirex_overlap)
    print_rolewise_f1_dual(stats_scirex_I, stats_scirex_C, name="SciREX")
    stats_scirex_I, stats_scirex_C = compute_eventtype_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=scirex_overlap)
    print_eventtype_f1_dual(stats_scirex_I, stats_scirex_C, name="SciREX")
    s_scirex_I, s_scirex_C = compute_domainwise_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=scirex_overlap)
    print_domainwise_f1_dual(s_scirex_I, s_scirex_C, name="SciREX")

    print("")
    # === IoU Match ===
    print_score("Argument Identification (IoU)", scores[("Arg_I", "IOU")])
    print_score("Argument Classification (IoU)", scores[("Arg_C", "IOU")])
    stats_iou_I, stats_iou_C = compute_rolewise_f1_dual(pred_roles, gold_roles, overlap_fn=iou_overlap)
    print_rolewise_f1_dual(stats_iou_I, stats_iou_C, name="IoU")
    stats_iou_I, stats_iou_C = compute_eventtype_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=iou_overlap)
    print_eventtype_f1_dual(stats_iou_I, stats_iou_C, name="IoU")
    s_iou_I, s_iou_C = compute_domainwise_f1_dual(pred_roles, gold_roles, oneie_wnd_ids, overlap_fn=iou_overlap)
    print_domainwise_f1_dual(s_iou_I, s_iou_C, name="IoU")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to prediction JSONL')
    parser.add_argument('--gold', required=True, help='Path to gold JSONL')
    args = parser.parse_args()
    main(args.pred, args.gold)