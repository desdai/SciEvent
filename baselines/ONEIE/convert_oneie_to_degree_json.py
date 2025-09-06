import os
import json
import glob
from tqdm import tqdm
from collections import defaultdict

def load_oneie_predictions(pred_dir):
    """Load all OneIE-style .json files and group by doc_id."""
    file_list = glob.glob(os.path.join(pred_dir, "*.json"))
    grouped = defaultdict(list)
    for file_path in tqdm(file_list, desc="Loading OneIE predictions"):
        with open(file_path) as f:
            for line in f:
                entry = json.loads(line)
                grouped[entry["doc_id"]].append(entry)
    return grouped

def convert_to_degree_jsonl_format(grouped_preds, output_file):
    """Write predictions in true DEGREE .jsonl format (one JSON per line, no commas)."""
    with open(output_file, "w") as out_f:
        for doc_id, entries in grouped_preds.items():
            for sent in entries:
                event_mentions = []
                for trig_idx, (start, end, event_type, score) in enumerate(sent["graph"]["triggers"]):
                    trigger = {
                        "start": start,
                        "end": end,
                        "text": " ".join(sent["tokens"][start:end])
                    }
                    arguments = []
                    for t_idx, e_idx, role, role_score in sent["graph"]["roles"]:
                        if t_idx == trig_idx:
                            ent_start, ent_end, _, _, _ = sent["graph"]["entities"][e_idx]
                            arguments.append({
                                "role": role,
                                "text": " ".join(sent["tokens"][ent_start:ent_end]),
                                "start": ent_start,
                                "end": ent_end
                            })
                    event_mentions.append({
                        "event_type": event_type,
                        "trigger": trigger,
                        "arguments": arguments
                    })

                one_obj = {
                    "doc_id": sent["sent_id"],
                    "sent_id": sent["sent_id"],
                    "tokens": sent["tokens"],
                    "event_mentions": event_mentions
                }
                out_f.write(json.dumps(one_obj) + "\n")

    print(f"Saved DEGREE-style .jsonl file to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing OneIE .json prediction files")
    parser.add_argument("--output_file", type=str, required=True, help="Output .json file path")
    args = parser.parse_args()
    grouped_preds = load_oneie_predictions(args.pred_dir)
    convert_to_degree_jsonl_format(grouped_preds, args.output_file)


# python convert_oneie_to_degree_json.py --pred_dir output_json --output_file oneie_preds_degree_format.json
