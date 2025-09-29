import os
import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_span(text, phrase):
    start = text.find(phrase)
    return (start, start + len(phrase)) if start != -1 else None

def clean_args(arg_list):
    """Flatten nested lists and remove invalid placeholders."""
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            elif isinstance(item, str):
                yield item

    return [a.strip() for a in flatten(arg_list) if a.strip() not in {"", "<NONE>", "Error"}]


def combine_with_modifier(base_list, mod_list):
    base = clean_args(base_list)
    mods = clean_args(mod_list)
    return [f"{b} {mods[0]}" for b in base] if mods else base

def convert_event_level(data, is_true=True):
    results = []
    for paper in data["papers"]:
        paper_code = paper.get("paper_code", "UNKNOWN")
        for idx, event in enumerate(paper.get("events", [])):
            try:
                text = event.get("Text", "")
                trigger_text = event.get("Main Action", "").strip()
                if trigger_text in {"", "<NONE>", "Error"}:
                    trigger_span = (-1, -1)
                else:
                    trigger_span = find_span(text, trigger_text)
                    if not trigger_span:
                        print(f"[WARN] Trigger not found: {paper_code}-{idx} → '{trigger_text}'")
                        trigger_span = (-1, -1)

                possible_types = ["Background/Introduction", "Methods/Approach", "Results/Findings", "Conclusions/Implications"]
                event_type = "Unknown"
                for et in possible_types:
                    if et in event and isinstance(event[et], str):
                        event_type = et
                        break

                entry = {
                    "event_code": f"{paper_code}-{idx}",
                    "abstract": text,
                    "Argument": [[trigger_span[0], trigger_span[1], "trigger"]],
                    "event_type": event_type,
                    "s_start": 0
                }

                args = event.get("Arguments", {})
                arg_list = []

                for role, val in args.items():
                    if isinstance(val, list):
                        for v in clean_args(val):
                            span = find_span(text, v)
                            if span:
                                arg_list.append([span[0], span[1], role])

                    elif isinstance(val, dict):
                        for subrole, subval in val.items():
                            try:
                                if role == "Object" and subrole == "Primary Object":
                                    pm = args[role].get("Primary Modifier", []) if is_true else []
                                    combined = combine_with_modifier(subval, pm)
                                    for phrase in combined:
                                        span = find_span(text, phrase)
                                        if span:
                                            arg_list.append([span[0], span[1], "Object:Primary Object"])
                                elif role == "Object" and subrole == "Secondary Object":
                                    sm = args[role].get("Secondary Modifier", []) if is_true else []
                                    combined = combine_with_modifier(subval, sm)
                                    for phrase in combined:
                                        span = find_span(text, phrase)
                                        if span:
                                            arg_list.append([span[0], span[1], "Object:Secondary Object"])
                                elif subrole in {"Primary Modifier", "Secondary Modifier"}:
                                    continue
                                else:
                                    for v in clean_args(subval):
                                        span = find_span(text, v)
                                        if span:
                                            arg_list.append([span[0], span[1], f"{role}:{subrole}"])
                            except Exception as sub_e:
                                print(f"[ERROR] At {paper_code}-{idx}, subrole={subrole}, value={subval}")
                                raise sub_e

                entry["Argument"].extend(arg_list)
                results.append(entry)

            except Exception as e:
                print(f"[ERROR] Crashed on event {paper_code}-{idx}")
                raise e

    return results

def process_folder(input_folder, output_file, is_true=True):
    lines = []
    for fname in os.listdir(input_folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(input_folder, fname)
        try:
            data = load_json(fpath)
            converted = convert_event_level(data, is_true=is_true)
            lines.extend(converted)
        except Exception as e:
            print(f"[ERROR] Failed on file: {fname}")
            raise e  # Re-raise to see full traceback after logging file name

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            json.dump(line, f)
            f.write('\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', required=True)
    parser.add_argument('--gold_folder', required=True)
    parser.add_argument('--pred_out', default='pred_event_level.json')
    parser.add_argument('--gold_out', default='gold_event_level.json')
    args = parser.parse_args()

    process_folder(args.pred_folder, args.pred_out, is_true=False)
    process_folder(args.gold_folder, args.gold_out, is_true=True)