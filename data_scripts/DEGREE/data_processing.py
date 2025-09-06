import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

def load_annotated_data(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    return data["papers"]

def init_tokenizer(model_name="facebook/bart-large", cache_dir="baselines/DEGREE/cache"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    special_tokens = ['<Trigger>', '<sep>']
    tokenizer.add_tokens(special_tokens)
    return tokenizer

def tokenize_with_bert(text, tokenizer):
    words = text.strip().split()
    pieces = []
    token_lens = []
    for word in words:
        word_pieces = tokenizer.tokenize(word)
        token_lens.append(len(word_pieces))
        pieces.extend(word_pieces)
    return words, pieces, token_lens

def find_token_span(phrase: str, tokens: List[str]) -> Tuple[int, int]:
    norm = lambda s: re.sub(r'\W+', '', s).lower()
    target = [norm(w) for w in phrase.strip().split()]
    token_norms = [norm(tok) for tok in tokens]

    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)+1):
            if token_norms[i:j] == target:
                return i, j
    return -1, -1

def align_trigger_and_arguments(entry: Dict, event_dict: Dict, event_type: str) -> Dict:
    tokens = entry["tokens"]
    wnd_id = entry["wnd_id"]
    trigger_text = event_dict.get("Main Action", "").strip()

    trigger_start, trigger_end = find_token_span(trigger_text, tokens)
    if trigger_start == -1:
        print(f"[WARN] Trigger not found: '{trigger_text}'")
        return entry

    trigger_id = f"{wnd_id}-EV0"
    arg_id_counter = 0
    entity_mentions = []
    event_arguments = []

    def add_argument_span(phrase: str, role: str):
        nonlocal arg_id_counter
        span_start, span_end = find_token_span(phrase, tokens)
        if span_start == -1:
            print(f"[WARN] Argument not found: '{phrase}' ({role})")
            return
        eid = f"{wnd_id}-E{arg_id_counter}"
        arg_id_counter += 1
        entity_mentions.append({
            "id": eid,
            "text": phrase,
            "start": span_start,
            "end": span_end,
            "entity_type": "UNK",
            "mention_type": "UNK"
        })
        event_arguments.append({
            "entity_id": eid,
            "text": phrase,
            "role": role
        })

    # Standard roles
    for role in ["Agent", "Context", "Purpose", "Method", "Results", "Analysis", "Challenge", "Ethical", "Implications", "Contradictions"]:
        for phrase in event_dict.get("Arguments", {}).get(role, []):
            add_argument_span(phrase, role)

    # Object-specific handling
    obj = event_dict.get("Arguments", {}).get("Object", {})
    for phrase in obj.get("Primary Object", []):
        add_argument_span(phrase, "PrimaryObject")
    for phrase in obj.get("Secondary Object", []):
        add_argument_span(phrase, "SecondaryObject")

    # Optional: extract modifiers but comment them out
    # for phrase in obj.get("Primary Modifier", []):
    #     add_argument_span(phrase, "PrimaryModifier")
    # for phrase in obj.get("Secondary Modifier", []):
    #     add_argument_span(phrase, "SecondaryModifier")

    entry["entity_mentions"] = entity_mentions
    entry["event_mentions"] = [{
        "event_type": event_type,
        "id": trigger_id,
        "trigger": {
            "text": trigger_text,
            "start": trigger_start,
            "end": trigger_end
        },
        "arguments": event_arguments
    }]
    return entry


def extract_event_chunks(papers, tokenizer):
    event_entries = []
    for paper in papers:
        paper_id = paper["paper_code"]
        for idx, event in enumerate(paper["events"]):
            raw_text = event.get("Text", "")
            tokens, pieces, token_lens = tokenize_with_bert(raw_text, tokenizer)
            event_type = next(iter(event.keys() - {"Text", "Main Action", "Arguments"}), "Unknown")

            entry = {
                "doc_id": paper_id,
                "wnd_id": f"{paper_id}-{idx}",
                "entity_mentions": [],
                "relation_mentions": [],
                "event_mentions": [],
                "entity_coreference": [],
                "event_coreference": [],
                "tokens": tokens,
                "pieces": pieces,
                "token_lens": token_lens,
                "sentence": raw_text,
                "sentence_starts": [0]
            }

            entry = align_trigger_and_arguments(entry, event, event_type)
            event_entries.append(entry)
    return event_entries

# Run this locally:
# tokenizer = init_tokenizer()
# papers = load_annotated_data("ACL_annotation_23_11_annotated.json")
# result = extract_event_chunks(papers, tokenizer)
# with open("converted_test.json", "w") as f:
#     for entry in result:
#         f.write(json.dumps(entry) + "\n")
