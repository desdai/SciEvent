import json
# this create the 75 filtered gold, or pre event level json.
# Input files
event_file = "baselines/LLM/data/human/human_eval/gold_event_level.json"             # The JSONL file to filter
reference_file = "baselines/degree/aaa_my_data_conversion/splitted_data/filtered_test.oneie.json"     # The file you already filtered for dh/jmir

# Output file
output_file = "baselines/LLM/data/human/human_eval/filtered_gold_event_level.json" 

# Step 1: Collect allowed wnd_ids
allowed_event_codes = set()
with open(reference_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        allowed_event_codes.add(entry["wnd_id"])

# Step 2: Keep only events whose event_code is in allowed set
with open(event_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            event = json.loads(line)
            if event["event_code"] in allowed_event_codes:
                fout.write(json.dumps(event) + "\n")
