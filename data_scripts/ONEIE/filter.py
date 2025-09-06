#!/usr/bin/env python3
# filter.py
# Usage:
#   python data_scripts/ONEIE/filter.py -filter SciEvent_data/HUMAN/human_annotation.json \
#                    -input baselines/ONEIE/txt/full_data \
#                    -output baselines/ONEIE/txt/human_subset

import os
import json
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Filter TXT files by keeping only those with event_codes present in a JSONL guide."
    )
    parser.add_argument("-filter", required=True,
                        help="Path to JSONL file containing valid event_code entries.")
    parser.add_argument("-input", required=True,
                        help="Path to input TXT folder (full set).")
    parser.add_argument("-output", required=True,
                        help="Path to output TXT folder (filtered subset).")
    args = parser.parse_args()

    # Step 1: Load valid event codes
    valid_event_codes = set()
    with open(args.filter, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                valid_event_codes.add(entry["event_code"])

    os.makedirs(args.output, exist_ok=True)

    # Step 2: Copy only files that match the valid codes
    kept, skipped = 0, 0
    for fname in os.listdir(args.input):
        if fname.endswith(".txt"):
            base = fname[:-4]  # remove ".txt"
            src_path = os.path.join(args.input, fname)
            dst_path = os.path.join(args.output, fname)
            if base in valid_event_codes:
                shutil.copy(src_path, dst_path)
                kept += 1
            else:
                skipped += 1

    print(f"[Done] Kept {kept} files, skipped {skipped}. Output saved to {args.output}")

if __name__ == "__main__":
    main()
