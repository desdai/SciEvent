import json

# File paths
left_file = "baselines/LLM/data/qwen_2.5_7b/improve_oneshot_eval/pred_event_level.json"                  # file to be filtered
right_file = "baselines/LLM/data/human/human_eval/filtered_gold_event_level.json"        # reference file
output_file = "baselines/LLM/data/qwen_2.5_7b/improve_oneshot_eval/filtered_pred_event_level.json"       # result

# Step 1: Load valid event codes from right file
valid_codes = set()
with open(right_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            valid_codes.add(obj["event_code"])

# Step 2: Filter left file using event_code match
with open(left_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            obj = json.loads(line)
            if obj.get("event_code") in valid_codes:
                fout.write(json.dumps(obj) + "\n")
