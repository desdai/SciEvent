import os
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def count_all(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    included_roles = {
        "Context", "Purpose", "Method", "Results", "Analysis",
        "Challenge", "Ethical", "Implications", "Contradictions"
    }

    total_events = 0
    total_roles = 0
    total_role_tokens = 0
    total_event_tokens = 0
    total_abstracts = 0
    total_abstract_tokens = 0

    for paper in data.get("papers", []):
        abstract = paper.get("abstract", "").strip()
        if abstract:
            abs_tokens = word_tokenize(abstract)
            total_abstracts += 1
            total_abstract_tokens += len(abs_tokens)

        for event in paper.get("events", []):
            total_events += 1
            event_text = event.get("Text", "").strip()
            ev_tokens = word_tokenize(event_text)
            total_event_tokens += len(ev_tokens)

            args = event.get("Arguments", {})
            for role, values in args.items():
                if role not in included_roles:
                    continue
                for val in values:
                    val = val.strip()
                    if val:
                        tokens = word_tokenize(val)
                        total_roles += 1
                        total_role_tokens += len(tokens)

    return (total_events, total_roles, total_role_tokens,
            total_event_tokens, total_abstracts, total_abstract_tokens)

# Directory containing JSON files
json_folder = "baselines/data_stat/ours/all_checked_data"

# Accumulators
total_events = 0
total_roles = 0
total_role_tokens = 0
total_event_tokens = 0
total_abstracts = 0
total_abstract_tokens = 0

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(json_folder, filename)
        e, r, rt, et, a, at = count_all(filepath)
        total_events += e
        total_roles += r
        total_role_tokens += rt
        total_event_tokens += et
        total_abstracts += a
        total_abstract_tokens += at

# Final Report
print(f"Total abstracts: {total_abstracts}")
print(f"Total events: {total_events}")
print(f"Total non-empty roles (filtered): {total_roles}")
print(f"Total tokens (abstracts only): {total_abstract_tokens}")
print()
print(f"Average roles per event: {total_roles / total_events:.2f}" if total_events else "N/A")
print(f"Average tokens per role: {total_role_tokens / total_roles:.2f}" if total_roles else "N/A")
print(f"Average tokens per event: {total_event_tokens / total_events:.2f}" if total_events else "N/A")
print(f"Average tokens per abstract: {total_abstract_tokens / total_abstracts:.2f}" if total_abstracts else "N/A")

print(f"Ratio of annotated tokens: {total_role_tokens / total_abstract_tokens:.2f}")
