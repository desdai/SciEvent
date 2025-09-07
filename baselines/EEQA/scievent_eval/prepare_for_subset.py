import json

# File paths
reference_file = "SciEvent_data/EEQA/human_subset/test_subset.eeqa.json"      # contains correct sentences
input_file = "baselines/EEQA/scievent_eval/full_data/arg_predictions_final.json"        # sentences to filter
output_file = "baselines/EEQA/scievent_eval/human_subset/arg_predictions_subset.json"

# Step 1: Load reference sentences (as tuple for hashability)
valid_sentences = set()
with open(reference_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        sentence = tuple(obj["sentence"])
        valid_sentences.add(sentence)

# Step 2: Filter input sentences
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        sentence = tuple(obj["sentence"])
        if sentence in valid_sentences:
            fout.write(json.dumps(obj) + "\n")
