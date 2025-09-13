import json

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            event_id = entry["event_code"]
            if not event_id.startswith("ACL_"):
                continue
            triggers = [tuple(arg[:2]) for arg in entry["Argument"] if arg[2] == "trigger"]
            arguments = [tuple(arg) for arg in entry["Argument"] if arg[2] != "trigger"]
            data[event_id] = {
                "triggers": triggers,      # keep as list for loose overlap check
                "arguments": arguments
            }
    return data

def spans_overlap(span1, span2):
    return span1[0] < span2[1] and span2[0] < span1[1]

def compute_agreement_with_loose_trigger(file1, file2):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    total_events = 0
    exact_trigger_mismatches = 0
    loose_trigger_mismatches = 0

    total_gold_args = 0
    exact_matched_args = 0
    loose_matched_args = 0

    for event_code in data1:
        if event_code not in data2:
            continue

        total_events += 1

        gold_triggers = data1[event_code]["triggers"]
        pred_triggers = data2[event_code]["triggers"]

        # Exact trigger match
        if set(gold_triggers) != set(pred_triggers):
            exact_trigger_mismatches += 1

        # Loose trigger match
        trigger_matched = False
        for g in gold_triggers:
            for p in pred_triggers:
                if spans_overlap(g, p):
                    trigger_matched = True
                    break
            if trigger_matched:
                break
        if not trigger_matched:
            loose_trigger_mismatches += 1

        # Arguments
        gold_args = data1[event_code]["arguments"]
        pred_args = data2[event_code]["arguments"]

        total_gold_args += len(gold_args)

        for g_start, g_end, g_role in gold_args:
            if (g_start, g_end, g_role) in pred_args:
                exact_matched_args += 1
                loose_matched_args += 1
            else:
                for p_start, p_end, p_role in pred_args:
                    if p_role == g_role and spans_overlap((g_start, g_end), (p_start, p_end)):
                        loose_matched_args += 1
                        break

    # Agreement scores
    exact_trigger_agreement = (total_events - exact_trigger_mismatches) / total_events if total_events > 0 else 0.0
    loose_trigger_agreement = (total_events - loose_trigger_mismatches) / total_events if total_events > 0 else 0.0
    exact_argument_agreement = exact_matched_args / total_gold_args if total_gold_args > 0 else 0.0
    loose_argument_agreement = loose_matched_args / total_gold_args if total_gold_args > 0 else 0.0

    # Print results
    print(f"Total matched ACL_ events: {total_events}")
    print(f"Exact trigger mismatches: {exact_trigger_mismatches}")
    print(f"Loose trigger mismatches: {loose_trigger_mismatches}")
    print(f"Exact trigger agreement: {exact_trigger_agreement:.4f}  ({total_events - exact_trigger_mismatches}/{total_events})")
    print(f"Loose trigger agreement: {loose_trigger_agreement:.4f}  ({total_events - loose_trigger_mismatches}/{total_events})")
    print()
    print(f"Total gold arguments: {total_gold_args}")
    print(f"Exact matched arguments: {exact_matched_args}")
    print(f"Loose matched arguments (span-overlap + same role): {loose_matched_args}")
    print(f"Exact argument agreement: {exact_argument_agreement:.4f}  ({exact_matched_args}/{total_gold_args})")
    print(f"Loose argument agreement: {loose_argument_agreement:.4f}  ({loose_matched_args}/{total_gold_args})")

# Example usage
# compute_agreement_with_loose_trigger("baselines/LLM/data/human/human_eval/pred_event_level.json", "baselines/LLM/data/human/human_eval/gold_event_level.json")

import json
from sklearn.metrics import cohen_kappa_score

def load_trigger_spans(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if not ex["event_code"].startswith("ACL_"):
                continue
            for span in ex["Argument"]:
                if span[2] == "trigger":
                    data[ex["event_code"]] = tuple(span[:2])
                    break
    return data

def compute_trigger_kappa(file1, file2):
    spans1 = load_trigger_spans(file1)
    spans2 = load_trigger_spans(file2)

    common_keys = sorted(set(spans1.keys()) & set(spans2.keys()))
    total = len(common_keys)

    labels1 = []
    labels2 = []

    for key in common_keys:
        span1 = spans1[key]
        span2 = spans2[key]
        labels1.append(f"{span1[0]}-{span1[1]}")
        labels2.append(f"{span2[0]}-{span2[1]}")


    kappa = cohen_kappa_score(labels1, labels2)
    exact_matches = sum([1 for a, b in zip(labels1, labels2) if a == b])

    print(f"Total ACL_ events compared: {total}")
    print(f"Trigger span exact matches: {exact_matches}")
    print(f"Cohen’s Kappa (trigger span per event): {kappa:.4f}")


# Example usage
compute_trigger_kappa("baselines/LLM/data/human/human_eval/pred_event_level.json", "baselines/LLM/data/human/human_eval/gold_event_level.json")
compute_trigger_kappa("baselines/LLM/data/human/human_eval_finalround/pred_event_level.json", "baselines/LLM/data/human/human_eval_finalround/gold_event_level.json")



import json
from sklearn.metrics import cohen_kappa_score

def load_role_to_span(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if not ex["event_code"].startswith("ACL_"):
                continue
            role_map = {}
            for start, end, role in ex["Argument"]:
                if role != "trigger":
                    role_map[role] = (start, end)  # assume one span per role
            data[ex["event_code"]] = role_map
    return data

def compute_argument_span_kappa_on_agreed_roles(file1, file2):
    a1 = load_role_to_span(file1)
    a2 = load_role_to_span(file2)

    labels1 = []
    labels2 = []
    matched_role_count = 0

    for event_code in (set(a1) & set(a2)):
        roles1 = a1[event_code]
        roles2 = a2[event_code]
        common_roles = set(roles1) & set(roles2)

        for role in common_roles:
            span1 = roles1[role]
            span2 = roles2[role]
            labels1.append(f"{span1[0]}-{span1[1]}")
            labels2.append(f"{span2[0]}-{span2[1]}")
            matched_role_count += 1

    kappa = cohen_kappa_score(labels1, labels2)
    exact_matches = sum([1 for a, b in zip(labels1, labels2) if a == b])

    print(f"Total agreed roles compared: {matched_role_count}")
    print(f"Exact span matches: {exact_matches}")
    print(f"Cohen’s Kappa (on agreed roles' spans): {kappa:.4f}")

# Example usage
compute_argument_span_kappa_on_agreed_roles("baselines/LLM/data/human/human_eval/pred_event_level.json", "baselines/LLM/data/human/human_eval/gold_event_level.json")
compute_argument_span_kappa_on_agreed_roles("baselines/LLM/data/human/human_eval_finalround/pred_event_level.json", "baselines/LLM/data/human/human_eval_finalround/gold_event_level.json")


import json
from sklearn.metrics import cohen_kappa_score

def load_span_to_role(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if not ex["event_code"].startswith("ACL_"):
                continue
            span_map = {}
            for start, end, role in ex["Argument"]:
                if role != "trigger":
                    span_map[(start, end)] = role  # assume one role per span
            data[ex["event_code"]] = span_map
    return data

def compute_role_kappa_on_agreed_spans(file1, file2):
    a1 = load_span_to_role(file1)
    a2 = load_span_to_role(file2)

    roles1 = []
    roles2 = []
    agreed_span_count = 0

    for event_code in (set(a1) & set(a2)):
        spans1 = a1[event_code]
        spans2 = a2[event_code]
        common_spans = set(spans1) & set(spans2)

        for span in common_spans:
            roles1.append(spans1[span])
            roles2.append(spans2[span])
            agreed_span_count += 1

    if agreed_span_count == 0:
        print("No agreed spans found — cannot compute Kappa.")
        return

    kappa = cohen_kappa_score(roles1, roles2)
    match_count = sum(1 for r1, r2 in zip(roles1, roles2) if r1 == r2)

    print(f"Total agreed spans: {agreed_span_count}")
    print(f"Matching role labels: {match_count}")
    print(f"Cohen’s Kappa (on roles for agreed spans): {kappa:.4f}")

compute_role_kappa_on_agreed_spans("baselines/LLM/data/human/human_eval/pred_event_level.json", "baselines/LLM/data/human/human_eval/gold_event_level.json")
compute_role_kappa_on_agreed_spans("baselines/LLM/data/human/human_eval_finalround/pred_event_level.json", "baselines/LLM/data/human/human_eval_finalround/gold_event_level.json")
