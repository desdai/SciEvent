#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
import tempfile
import os

def step1_filter_best(input_pred_raw, fout):
    for line in input_pred_raw:
        if not line.strip():
            fout.write("[]\n")
            continue
        preds = json.loads(line)
        best_by_span = {}
        for item in preds:
            if not isinstance(item, list) or len(item) < 3:
                continue
            label, span, score = item[:3]
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            span_key = (int(span[0]), int(span[1]))
            if span_key not in best_by_span or score < best_by_span[span_key][1]:
                best_by_span[span_key] = (label, score)
        cleaned = [[label, [s0, s1], score] for (s0, s1), (label, score) in best_by_span.items()]
        fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

def step2_structure_args(f_best, f_gold, fout, dummy_trigger_idx=-1):
    for pred_line, gold_line in zip(f_best, f_gold):
        preds = json.loads(pred_line) if pred_line.strip() else []
        gold_data = json.loads(gold_line) if gold_line.strip() else {}
        event_dict = defaultdict(list)
        for item in preds:
            if not isinstance(item, list) or len(item) < 3:
                continue
            label, span, score = item[:3]
            try:
                event_type, role = label.split("_", 1)
            except ValueError:
                continue
            event_key = (dummy_trigger_idx, event_type)
            event_dict[event_key].append([int(span[0]), int(span[1]), role])
        events = []
        for (trigger_start, evt_type), args_list in event_dict.items():
            events.append([[trigger_start, evt_type]] + args_list)
        gold_data["event"] = events
        fout.write(json.dumps(gold_data, ensure_ascii=False) + "\n")

def step3_merge_triggers(f_struct, f_trig, fout):
    for struct_line, trig_line in zip(f_struct, f_trig):
        arg_entry = json.loads(struct_line) if struct_line.strip() else {}
        trig_entry = json.loads(trig_line) if trig_line.strip() else {}
        new_events = []
        predicted_triggers = trig_entry.get("event", [])
        type_to_args = defaultdict(list)
        for dummy_event in arg_entry.get("event", []):
            if not dummy_event or not isinstance(dummy_event[0], list) or len(dummy_event[0]) < 2:
                continue
            evt_type = dummy_event[0][1]
            args_list = []
            for a in dummy_event[1:]:
                if isinstance(a, list) and len(a) >= 3:
                    args_list.append([int(a[0]), int(a[1]), a[2]])
            if args_list:
                type_to_args[evt_type].append(args_list)
        for trigger_group in predicted_triggers:
            if not isinstance(trigger_group, list) or len(trigger_group) != 1:
                continue
            head = trigger_group[0]
            if not (isinstance(head, list) and len(head) >= 2):
                continue
            trigger_start, trigger_type = head[0], head[1]
            new_event = [[int(trigger_start), trigger_type]]
            if trigger_type in type_to_args and len(type_to_args[trigger_type]) > 0:
                new_event += type_to_args[trigger_type][0]
            new_events.append(new_event)
        arg_entry["event"] = new_events
        fout.write(json.dumps(arg_entry, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: arg preds → best per span → structured → merged with triggers."
    )
    parser.add_argument("-arg_raw", required=True, help="Raw arg predictions JSONL")
    parser.add_argument("-gold", required=True, help="Gold sentences JSONL")
    parser.add_argument("-trig_pred", required=True, help="Trigger predictions JSONL")
    parser.add_argument("-final_out", required=True, help="Final output JSONL")
    args = parser.parse_args()

    # Create temp files for step1 and step2
    with tempfile.TemporaryDirectory() as tmpdir:
        best_path = os.path.join(tmpdir, "best.json")
        struct_path = os.path.join(tmpdir, "struct.json")

        with open(args.arg_raw, "r", encoding="utf-8") as f_in, \
             open(best_path, "w", encoding="utf-8") as f_best:
            step1_filter_best(f_in, f_best)

        with open(best_path, "r", encoding="utf-8") as f_best, \
             open(args.gold, "r", encoding="utf-8") as f_gold, \
             open(struct_path, "w", encoding="utf-8") as f_struct:
            step2_structure_args(f_best, f_gold, f_struct)

        with open(struct_path, "r", encoding="utf-8") as f_struct, \
             open(args.trig_pred, "r", encoding="utf-8") as f_trig, \
             open(args.final_out, "w", encoding="utf-8") as f_out:
            step3_merge_triggers(f_struct, f_trig, f_out)

    print(f"✓ Pipeline finished. Final file: {args.final_out}")

if __name__ == "__main__":
    main()
