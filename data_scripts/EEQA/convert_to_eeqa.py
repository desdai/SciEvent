import json
from tqdm import tqdm
import argparse

def convert_to_eeqa_format(scientific_instance):
    tokens = scientific_instance["tokens"]
    s_start = scientific_instance["sentence_starts"][0]
    entity_mentions = scientific_instance["entity_mentions"]
    event_mentions = scientific_instance["event_mentions"]
    relation_mentions = scientific_instance.get("relation_mentions", [])

    # Map entity_id to (start, end, type)
    entity_map = {
        ent["id"]: (ent["start"], ent["end"], ent.get("entity_type", "ENT"))
        for ent in entity_mentions
    }

    ner = [[start, end, ent_type] for (_, (start, end, ent_type)) in entity_map.items()]

    relation = []
    for rel in relation_mentions:
        head = rel.get("head")
        tail = rel.get("tail")
        if head in entity_map and tail in entity_map:
            h_start, h_end, _ = entity_map[head]
            t_start, t_end, _ = entity_map[tail]
            relation.append([h_start, h_end, t_start, t_end, rel["relation_type"]])

    event = []
    for ev in event_mentions:
        event_type = ev["event_type"]
        trigger = ev["trigger"]
        trigger_start = trigger["start"]
        event_entry = [[trigger_start, event_type]]

        for arg in ev["arguments"]:
            ent_id = arg["entity_id"]
            role = arg["role"]
            if ent_id in entity_map:
                arg_start, arg_end, _ = entity_map[ent_id]
                event_entry.append([arg_start, arg_end, role])

        event.append(event_entry)

    return {
        "sentence": tokens,
        "s_start": s_start,
        "ner": ner,
        "relation": relation,
        "event": event
    }


def convert_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Converting"):
            line = line.strip()
            if not line or line == "{}":
                continue  # skip empty or dummy entries
            instance = json.loads(line)
            converted = convert_to_eeqa_format(instance)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Saved converted EEQA-format data to {output_file} (JSONL)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert scientific JSONL data into EEQA format.")
    parser.add_argument("-input", "--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("-output", "--output_file", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    convert_dataset(args.input_file, args.output_file)