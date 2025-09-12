import json
from typing import List, Tuple, Dict, Callable

ROLES = ["Agent", "trigger", "Object:Primary Object", "Object:Secondary Object"]

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_tuple(entry: Dict) -> Tuple[str, str, str, str]:
    text = entry.get("event_text") or entry.get("abstract", "")
    role_vals = {role: "" for role in ROLES}
    for start, end, role in entry.get("Argument", []):
        if role in role_vals and 0 <= start < end <= len(text):
            if role_vals[role] == "":
                role_vals[role] = text[start:end].strip()
    return tuple(role_vals[role] for role in ROLES)

# Matching methods
def exact_match(p: str, g: str) -> bool:
    return p.strip() == g.strip()

def one_word_overlap(p: str, g: str) -> bool:
    return bool(set(p.strip().split()) & set(g.strip().split()))

def iou_overlap(p: str, g: str) -> bool:
    p_tokens = set(p.strip().split())
    g_tokens = set(g.strip().split())
    inter = len(p_tokens & g_tokens)
    union = len(p_tokens | g_tokens)
    return (inter / union) > 0.5 if union > 0 else False

# Per-event normalized evaluation
def compute_metrics(gold_tuples: List[Tuple[str, str, str, str]],
                    pred_tuples: List[Tuple[str, str, str, str]],
                    match_fn: Callable[[str, str], bool]) -> Dict[str, float]:
    assert len(gold_tuples) == len(pred_tuples)
    total_prec, total_rec, total_f1 = 0.0, 0.0, 0.0
    n = len(gold_tuples)

    for gold, pred in zip(gold_tuples, pred_tuples):
        matched = 0
        gold_count = 0
        pred_count = 0

        for g, p in zip(gold, pred):
            g_present = bool(g.strip())
            p_present = bool(p.strip())

            if g_present:
                gold_count += 1
            if p_present:
                pred_count += 1
            if g_present and p_present and match_fn(p, g):
                matched += 1

        prec = matched / pred_count if pred_count > 0 else 0.0
        rec = matched / gold_count if gold_count > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        total_prec += prec
        total_rec += rec
        total_f1 += f1

    return {
        "Precision": total_prec / n if n else 0.0,
        "Recall": total_rec / n if n else 0.0,
        "F1": total_f1 / n if n else 0.0
    }

# Main function
def evaluate(gold_path: str, pred_path: str):
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    gold_map = {e["event_code"]: e for e in gold_data}
    pred_map = {e["event_code"]: e for e in pred_data}
    shared_keys = sorted(set(gold_map) & set(pred_map))

    gold_tuples = [extract_tuple(gold_map[k]) for k in shared_keys]
    pred_tuples = [extract_tuple(pred_map[k]) for k in shared_keys]

    print("\n📌 Exact Match")
    em = compute_metrics(gold_tuples, pred_tuples, exact_match)
    print(f"Precision: {em['Precision']:.4f}")
    print(f"Recall:    {em['Recall']:.4f}")
    print(f"F1:        {em['F1']:.4f}")

    print("\n📌 One-Word Overlap")
    ow = compute_metrics(gold_tuples, pred_tuples, one_word_overlap)
    print(f"Precision: {ow['Precision']:.4f}")
    print(f"Recall:    {ow['Recall']:.4f}")
    print(f"F1:        {ow['F1']:.4f}")

    print("\n📌 IoU > 50%")
    iou = compute_metrics(gold_tuples, pred_tuples, iou_overlap)
    print(f"Precision: {iou['Precision']:.4f}")
    print(f"Recall:    {iou['Recall']:.4f}")
    print(f"F1:        {iou['F1']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to gold JSONL")
    parser.add_argument("--pred", required=True, help="Path to predicted JSONL")
    args = parser.parse_args()

    evaluate(args.gold, args.pred)



# python baselines/LLM/code/eval/anchor_eval_tuple.py --gold baselines/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json --pred baselines/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json