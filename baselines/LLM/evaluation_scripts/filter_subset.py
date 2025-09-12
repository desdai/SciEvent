import json
import argparse

def main(input_file, filter_file, output_file):
    # Step 1: Collect allowed wnd_ids
    allowed_event_codes = set()
    with open(filter_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            allowed_event_codes.add(entry["wnd_id"])

    # Step 2: Keep only events whose event_code is in allowed set
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                event = json.loads(line)
                if event["event_code"] in allowed_event_codes:
                    fout.write(json.dumps(event) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter events based on reference file wnd_ids")
    parser.add_argument("--input", required=True, help="Path to input event JSONL file")
    parser.add_argument("--filter", required=True, help="Path to reference JSONL file (with allowed wnd_ids)")
    parser.add_argument("--output", required=True, help="Path to save filtered JSONL file")

    args = parser.parse_args()
    main(args.input, args.filter, args.output)
