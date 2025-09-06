# save as convert_test_jsonl_to_txt.py
import os, json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_jsonl", required=True, help="Path to test .jsonl file")
parser.add_argument("--output_dir", required=True, help="Where to save .txt files")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.input_jsonl) as f:
    for line in f:
        item = json.loads(line)
        tokens = item["tokens"]
        sent_id = item["sent_id"]
        txt_path = os.path.join(args.output_dir, f"{sent_id}.txt")
        with open(txt_path, "w") as out:
            out.write(" ".join(tokens))
