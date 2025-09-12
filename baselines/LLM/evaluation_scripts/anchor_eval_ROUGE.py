import json
from rouge_score import rouge_scorer

DESIRED_ROLES = ["Agent", "trigger", "Object:Primary Object", "Object:Secondary Object"]

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_ordered_span(entry):
    text = entry.get("event_text") or entry.get("abstract", "")
    role_map = {role: [] for role in DESIRED_ROLES}

    for start, end, role in entry.get("Argument", []):
        if role in DESIRED_ROLES and 0 <= start < end <= len(text):
            role_map[role].append(text[start:end].strip())

    ordered = [role_map[role][0] if role_map[role] else "" for role in DESIRED_ROLES]
    return " ".join(ordered).strip()

def align_by_event_code(gold_entries, pred_entries):
    gold_map = {e["event_code"]: e for e in gold_entries}
    pred_map = {e["event_code"]: e for e in pred_entries}
    shared_keys = sorted(set(gold_map) & set(pred_map))
    return [(gold_map[k], pred_map[k]) for k in shared_keys]

def compute_rouge_all(gold_list, pred_list):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(g, p)["rougeL"] for g, p in zip(gold_list, pred_list)]
    if not scores:
        return 0.0, 0.0, 0.0
    p_avg = sum(s.precision for s in scores) / len(scores)
    r_avg = sum(s.recall for s in scores) / len(scores)
    f_avg = sum(s.fmeasure for s in scores) / len(scores)
    return p_avg, r_avg, f_avg

def main(gold_path, pred_path):
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)
    matched = align_by_event_code(gold_data, pred_data)

    gold_texts = [extract_ordered_span(g) for g, _ in matched]
    pred_texts = [extract_ordered_span(p) for _, p in matched]

    p, r, f = compute_rouge_all(gold_texts, pred_texts)

    print("\n📈 Final ROUGE-L (Macro Average):")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to gold JSONL")
    parser.add_argument("--pred", required=True, help="Path to predicted JSONL")
    args = parser.parse_args()

    main(args.gold, args.pred)


# python baselines/LLM/code/eval/anchor_eval_ROUGE.py --gold baselines/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json --pred baselines/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json