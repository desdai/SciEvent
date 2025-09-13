import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-folder", type=str, required=True, help="Input folder path")
parser.add_argument("--output-folder", type=str, required=True, help="Output folder path")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True)

section_labels = ['"[Background]"', '"[Method]"', '"[Results]"', '"[Implications]"']

def deduplicate_text(text):
    seen = set()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for sentence in sentences:
        s_norm = sentence.strip().rstrip('.,!?').lower()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            result.append(sentence.strip())
    return ' '.join(result)

for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and line != "{" and line != "}"]

    output_lines = []
    current_paper = None
    current_sections = {}

    def flush():
        if current_paper:
            output_lines.append(f"Paper Code: {current_paper}")
            output_lines.append("")
            for label in section_labels:
                if label in current_sections:
                    clean_label = label.strip('"')
                    dedup = deduplicate_text(current_sections[label])
                    output_lines.append(f"{clean_label}: {dedup}")
                    output_lines.append("")
            output_lines.append("")

    for line in lines:
        if line.endswith("{") and line.startswith('"') and '":' in line:
            flush()
            current_paper = line.split(":", 1)[0].strip('"')
            current_sections = {}
        else:
            for label in section_labels:
                if line.startswith(label):
                    try:
                        value = line.split(":", 1)[1].strip().strip('",')
                        current_sections[label] = value
                    except IndexError:
                        current_sections[label] = "<NONE>"

    flush()

    # 🔁 Rename: chunked.txt → extracted.txt
    output_name = filename.replace("chunked.txt", "extracted.txt")
    output_path = os.path.join(output_folder, output_name)

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(output_lines))

    print(f"✅ Saved: {output_path} ({len(output_lines)} lines)")
