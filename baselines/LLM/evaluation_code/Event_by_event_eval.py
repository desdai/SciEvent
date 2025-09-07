import json
import pandas as pd

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def compute_event_f1(pred_list, gold_list, match_fn):
    matched = 0
    pred_total = len(pred_list)
    gold_total = len(gold_list)

    used_gold = set()
    for p in pred_list:
        for idx, g in enumerate(gold_list):
            if idx in used_gold:
                continue
            if match_fn(p, g):
                matched += 1
                used_gold.add(idx)
                break

    prec = matched / pred_total if pred_total > 0 else 0
    rec = matched / gold_total if gold_total > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1  # Only return F1 now

def extract_triggers_and_roles(data):
    trigger_dict = {}
    role_dict = {}

    for entry in data:
        event_id = entry["event_code"]
        triggers = []
        roles = []

        for item in entry["Argument"]:
            if len(item) == 3 and item[2] == "trigger":
                triggers.append((item[0], item[1], "trigger"))
            elif len(item) == 3:
                roles.append((item[0], item[1], item[2]))

        trigger_dict[event_id] = triggers
        role_dict[event_id] = roles

    return trigger_dict, role_dict

def main(pred_file, gold_file, output_excel):
    pred_data = load_jsonl(pred_file)
    gold_data = load_jsonl(gold_file)

    pred_trigs, pred_roles = extract_triggers_and_roles(pred_data)
    gold_trigs, gold_roles = extract_triggers_and_roles(gold_data)

    event_ids = sorted(set(pred_trigs.keys()) | set(gold_trigs.keys()))

    rows = []

    for eid in event_ids:
        pred_trig_list = pred_trigs.get(eid, [])
        gold_trig_list = gold_trigs.get(eid, [])
        pred_role_list = pred_roles.get(eid, [])
        gold_role_list = gold_roles.get(eid, [])

        # Trigger Identification (Exact span match)
        trig_f1 = compute_event_f1(
            pred_trig_list, gold_trig_list,
            match_fn=lambda x, y: x[:2] == y[:2]
        )

        # Argument Identification (Exact span match)
        argi_f1 = compute_event_f1(
            [(r[0], r[1], "dummy") for r in pred_role_list],
            [(r[0], r[1], "dummy") for r in gold_role_list],
            match_fn=lambda x, y: x[:2] == y[:2]
        )

        # Argument Classification (Exact span + role match)
        argc_f1 = compute_event_f1(
            pred_role_list,
            gold_role_list,
            match_fn=lambda x, y: x[2] == y[2] and x[:2] == y[:2]
        )

        rows.append({
            "event_code": eid,
            "TriggerI-EM-F1": round(trig_f1 * 100, 2),
            "ArgumentI-EM-F1": round(argi_f1 * 100, 2),
            "ArgumentC-EM-F1": round(argc_f1 * 100, 2),
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"Saved per-event F1 to {output_excel}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Prediction jsonl')
    parser.add_argument('--gold', required=True, help='Gold jsonl')
    parser.add_argument('--out', required=True, help='Output excel path')
    args = parser.parse_args()
    main(args.pred, args.gold, args.out)



# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/pred_event_level.json --gold baselines/LLM/data/qwen_2.5_7b/improve_zeroshot_eval/gold_event_level.json --out baselines/LLM/qwen_zero.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/qwen_2.5_7b/improve_oneshot_eval/pred_event_level.json --gold baselines/LLM/data/qwen_2.5_7b/improve_oneshot_eval/gold_event_level.json --out baselines/LLM/qwen_one.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/llama_3_8b/improve_zeroshot_eval/pred_event_level.json --gold baselines/LLM/data/llama_3_8b/improve_zeroshot_eval/gold_event_level.json --out baselines/LLM/llama_zero.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/llama_3_8b/improve_oneshot_eval/pred_event_level.json --gold baselines/LLM/data/llama_3_8b/improve_oneshot_eval/gold_event_level.json --out baselines/LLM/llama_one.xlsx

# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/pred_event_level.json --gold baselines/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_zeroshot_eval/gold_event_level.json --out baselines/LLM/distill_qwen_zero.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/pred_event_level.json --gold baselines/LLM/data/DeepSeek-R1-Distill-Qwen-7B/improve_oneshot_eval/gold_event_level.json --out baselines/LLM/distill_qwen_one.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/pred_event_level.json --gold baselines/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_zeroshot_eval/gold_event_level.json --out baselines/LLM/distill_llama_zero.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/pred_event_level.json --gold baselines/LLM/data/DeepSeek-R1-Distill-Llama-8B/improve_oneshot_eval/gold_event_level.json --out baselines/LLM/distill_llama_one.xlsx

# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/gpt_4.1/zeroshot_eval/pred_event_level.json --gold baselines/LLM/data/gpt_4.1/zeroshot_eval/gold_event_level.json --out baselines/LLM/gpt_zero.xlsx
# python baselines/LLM/code/eval/Event_by_event_eval.py --pred baselines/LLM/data/gpt_4.1/oneshot_eval/pred_event_level.json --gold baselines/LLM/data/gpt_4.1/oneshot_eval/gold_event_level.json --out baselines/LLM/gpt_one.xlsx
