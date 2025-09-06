# usage: checking if any token or sentence is missing, usually not needed
import json
import os

folder = "baselines/oneie/processed_data/mydata"
for fname in ["train.oneie.json", "dev.oneie.json", "test.oneie.json"]:
    path = os.path.join(folder, fname)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                inst = json.loads(line)
                if 'tokens' not in inst or not inst['tokens']:
                    print(f"[{fname}] Line {i+1} has missing or empty tokens.")
                elif not inst.get('sentence', '').strip():
                    print(f"[{fname}] Line {i+1} has empty sentence.")
            except Exception as e:
                print(f"[{fname}] Line {i+1} is broken: {e}")
