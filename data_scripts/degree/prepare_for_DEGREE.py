import os
import json
import shutil
from data_processing import load_annotated_data, init_tokenizer, extract_event_chunks

def main(input_dir: str, output_dir: str, merged_filename="all_data.json"):
    os.makedirs(output_dir, exist_ok=True)
    all_entries = []

    tokenizer = init_tokenizer()

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            print(f"📄 Processing: {input_path}")
            papers = load_annotated_data(input_path)
            entries = extract_event_chunks(papers, tokenizer)
            all_entries.extend(entries)

    # Save merged entries to a single file
    merged_path = os.path.join(output_dir, merged_filename)
    with open(merged_path, "w") as fout:
        for entry in all_entries:
            fout.write(json.dumps(entry) + "\n")

    # Copy vocab.json
    # shutil.copy(vocab_path, os.path.join(output_dir, "vocab.json"))

    print(f"\nFinished! {len(all_entries)} event chunks merged into: {merged_path}")
    print(f"Output folder: {output_dir}")
    # print(f"vocab.json copied from: {vocab_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="SciEvent_data/raw", help="Folder with input .json files (original format)")
    parser.add_argument("--output", default="SciEvent_data/DEGREE/processed", help="Folder to save DEGREE-format output")
    # parser.add_argument("--vocab", required=True, default="SciEvent_data/DEGREE/processed/vocab.json", help="Path to vocab.json file")
    args = parser.parse_args()
    # main(args.input, args.output, args.vocab)
    main(args.input, args.output)
