#!/usr/bin/env python3
# wnd_id_rename.py
# Usage: python wnd_id_rename.py -input path/to/in.json -output path/to/out.json

import argparse
import json
from collections import OrderedDict

def process_file(in_path: str, out_path: str) -> None:
    new_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            inst = json.loads(line, object_pairs_hook=OrderedDict)

            if "wnd_id" in inst:
                value = inst["wnd_id"]
                del inst["wnd_id"]

                # Insert sent_id right after doc_id if present (for readability)
                items = list(inst.items())
                idx = [i for i, (k, _) in enumerate(items) if k == "doc_id"]
                if idx:
                    insert_at = idx[0] + 1
                    items.insert(insert_at, ("sent_id", value))
                    inst = OrderedDict(items)
                else:
                    inst["sent_id"] = value

            new_lines.append(json.dumps(inst, ensure_ascii=False))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

def main():
    parser = argparse.ArgumentParser(description="Rename 'wnd_id' to 'sent_id' in a JSONL file and position it after 'doc_id'.")
    parser.add_argument("-input", "--input", "-i", dest="input_path", required=True, help="Input JSONL file path")
    parser.add_argument("-output", "--output", "-o", dest="output_path", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    process_file(args.input_path, args.output_path)

if __name__ == "__main__":
    main()