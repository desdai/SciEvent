import json
from collections import defaultdict

# I manually fixed the vocab, DON'T run this unless you want to redo everything!
def generate_vocab(entries):
    """
    Generate vocab.json from converted entries.
    Returns a dictionary with event_type and role_type vocab mappings.
    """
    event_types = set()
    role_types = set()

    for entry in entries:
        for event in entry.get("event_mentions", []):
            event_types.add(event["event_type"])
            for arg in event.get("arguments", []):
                role_types.add(arg["role"])

    vocab = {
        "event_type_itos": sorted(list(event_types)),
        "event_type_stoi": {etype: i for i, etype in enumerate(sorted(list(event_types)))},
        "role_type_itos": sorted(list(role_types)),
        "role_type_stoi": {role: i for i, role in enumerate(sorted(list(role_types)))}
    }

    return vocab

# Example usage (put this in your __main__ if running end-to-end)
from data_processing import extract_event_chunks, load_annotated_data, init_tokenizer
papers = load_annotated_data("ACL_annotation_23_11_annotated.json")
tokenizer = init_tokenizer()
entries = extract_event_chunks(papers, tokenizer)
vocab = generate_vocab(entries)
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)
