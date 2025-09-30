#!/usr/bin/env python3
"""
build_text_only.py

Construct a texts_only.jsonl (the "text companion" used by reconstruct.py) from:
  1) Raw abstract .txt files (one per doc_id), and
  2) event_seg.jsonl produced earlier (wnd_id -> start_token/end_token over abstract tokens), and
  3) The annotation JSONL (structure w/ indices but no verbatim text).

For each window (line in annotation):
- Load its abstract by doc_id and tokenize consistently.
- Use event_seg to slice the abstract tokens [start_token, end_token] (inclusive) -> window tokens.
- Build:
    * top-level "tokens" (window tokens)
    * top-level "sentence" = " ".join(tokens)
    * entity_mentions_text: [{"id", "text"}] by slicing tokens with (start,end) from entity mention (end-exclusive)
    * event_mentions_text:
        - {"id", "event_type", "trigger_text", "arguments_text":[{"entity_id","role","text"}]}
        - trigger_text from ev.trigger (start,end) (end-exclusive) over window tokens
        - argument text mapped from the entity_id's entity mention text

Output per line (one JSON object for each annotation line):
{
  "doc_id": ...,
  "wnd_id": ...,
  "tokens": [...],
  "sentence": "...",
  "entity_mentions_text": [{"id": "...", "text": "..."}],
  "event_mentions_text": [
    {
      "id": "...",
      "event_type": "...",
      "trigger_text": "...",
      "arguments_text": [{"entity_id":"...","role":"...","text":"..."}]
    },
    ...
  ]
}

Usage:
  python data_scripts/shared/prepare_segmentation.py \
    --annotation SciEvent_data/DEGREE/processed/event_extraction_finetune_model.jsonl \
    --event_seg SciEvent_data/DEGREE/processed/event_seg.jsonl \
    --abstract_dir SciEvent_data/abstracts_texts \
    --output SciEvent_data/DEGREE/processed/texts_only.jsonl
"""

import argparse
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

from transformers import AutoTokenizer  # pip install transformers


# ---------- Helpers (embedded) ----------

def init_tokenizer(model_name: str = "facebook/bart-large",
                   cache_dir: str = "baselines/DEGREE/cache"):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    special_tokens = ['<Trigger>', '<sep>']
    tok.add_tokens(special_tokens)
    return tok

def tokenize_with_bert(text: str, tokenizer):
    words = text.strip().split()
    pieces = []
    token_lens = []
    for w in words:
        wp = tokenizer.tokenize(w)
        token_lens.append(len(wp))
        pieces.extend(wp)
    return words, pieces, token_lens

def normalize_piece(p: str) -> str:
    if p.startswith("##"): p = p[2:]
    if p.startswith("Ġ"):  p = p[1:]
    if p.startswith("▁"):  p = p[1:]
    return p

def abstract_tokenize(text: str) -> List[str]:
    # MUST match event_seg tokenization
    return re.findall(r"\S+", text)

def load_abstract(abstract_dir: str, doc_id: str) -> Optional[str]:
    p1 = os.path.join(abstract_dir, doc_id)
    p2 = p1 + ".txt"
    for p in (p1, p2):
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return None

def build_entity_texts_exact(entity_mentions: List[Dict[str, Any]],
                             tokens: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    n = len(tokens)
    if not isinstance(entity_mentions, list):
        return out
    for ent in entity_mentions:
        if not isinstance(ent, dict):
            continue
        eid = ent.get("id"); s = ent.get("start"); e = ent.get("end")
        if eid is None or not isinstance(s, int) or not isinstance(e, int):
            continue
        if 0 <= s <= e <= n:
            out[eid] = " ".join(tokens[s:e])
        else:
            out[eid] = ""
    return out

def build_event_texts_exact(events: List[Dict[str, Any]],
                            tokens: List[str],
                            ent_text_map: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(tokens)
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        ev_id = ev.get("id"); ev_type = ev.get("event_type")

        trig_txt = None
        trg = ev.get("trigger")
        if isinstance(trg, dict):
            s = trg.get("start"); e = trg.get("end")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= n:
                trig_txt = " ".join(tokens[s:e])

        args_text_list: List[Dict[str, Any]] = []
        for a in ev.get("arguments", []) or []:
            if not isinstance(a, dict):
                continue
            ent_id = a.get("entity_id"); role = a.get("role")
            txt = ent_text_map.get(ent_id, "")
            item: Dict[str, Any] = {"text": txt}
            if ent_id is not None: item["entity_id"] = ent_id
            if role   is not None: item["role"] = role
            args_text_list.append(item)

        ev_item: Dict[str, Any] = {}
        if ev_id   is not None: ev_item["id"] = ev_id
        if ev_type is not None: ev_item["event_type"] = ev_type
        if trig_txt is not None: ev_item["trigger_text"] = trig_txt
        if args_text_list:       ev_item["arguments_text"] = args_text_list
        out.append(ev_item)
    return out

# ----------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", "-s", required=True, help="Path to annotation JSONL (no verbatim text).")
    ap.add_argument("--event_seg", "-e", required=True, help="Path to event_seg.jsonl (doc_id,wnd_id -> start_token,end_token inclusive).")
    ap.add_argument("--abstract_dir", "-a", required=True, help="Directory of raw abstracts named by doc_id (optionally with .txt).")
    ap.add_argument("--output", "-o", required=True, help="Output texts_only.jsonl path.")
    ap.add_argument("--hf_model", default="facebook/bart-large", help="HF tokenizer name/path.")
    ap.add_argument("--hf_cache", default="baselines/DEGREE/cache", help="HF cache dir.")
    args = ap.parse_args()

    tokenizer = init_tokenizer(model_name=args.hf_model, cache_dir=args.hf_cache)

    # (doc_id, wnd_id) -> (start_token, end_token INCLUSIVE)
    seg_map: Dict[Tuple[str, str], Tuple[int, int]] = {}
    with open(args.event_seg, "r", encoding="utf-8") as fseg:
        for line in fseg:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id"); wnd_id = obj.get("wnd_id")
            st = obj.get("start_token"); en = obj.get("end_token")
            if isinstance(doc_id, str) and isinstance(wnd_id, str) and isinstance(st, int) and isinstance(en, int):
                seg_map[(doc_id, wnd_id)] = (st, en)

    abs_text_cache: Dict[str, str] = {}
    abs_tokens_cache: Dict[str, List[str]] = {}

    with open(args.annotation, "r", encoding="utf-8") as fsan, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fsan:
            line = line.strip()
            if not line:
                continue
            sobj = json.loads(line)
            doc_id = sobj.get("doc_id"); wnd_id = sobj.get("wnd_id")

            # Load abstract tokens
            if doc_id not in abs_tokens_cache:
                text = load_abstract(args.abstract_dir, doc_id) if doc_id else None
                abs_text_cache[doc_id] = text if text is not None else ""
                abs_tokens_cache[doc_id] = abstract_tokenize(abs_text_cache[doc_id]) if text else []

            abs_tokens = abs_tokens_cache.get(doc_id, [])
            st_en = seg_map.get((doc_id, wnd_id))

            # Construct ordered output
            out_obj = OrderedDict()
            if isinstance(doc_id, str): out_obj["doc_id"] = doc_id
            if isinstance(wnd_id, str): out_obj["wnd_id"] = wnd_id

            if not abs_tokens or st_en is None:
                out_obj["tokens"] = []
                out_obj["pieces"] = []
                out_obj["sentence"] = ""
                out_obj["entity_mentions_text"] = []
                out_obj["event_mentions_text"] = []
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                continue

            start_tok, end_tok_incl = st_en
            if start_tok < 0 or end_tok_incl < start_tok or end_tok_incl >= len(abs_tokens):
                window_tokens: List[str] = []
            else:
                window_tokens = abs_tokens[start_tok: end_tok_incl + 1]

            # Keep tokens EXACT
            out_obj["tokens"] = window_tokens

            # Compute pieces from sentence but insert BEFORE sentence
            sentence_str = " ".join(window_tokens)
            _, pieces, _ = tokenize_with_bert(sentence_str, tokenizer)
            pieces_norm = [normalize_piece(p) for p in pieces]
            out_obj["pieces"] = pieces_norm

            # Now sentence (after pieces)
            out_obj["sentence"] = sentence_str

            # EXACT texts from spans over THESE tokens
            ent_text_map = build_entity_texts_exact(sobj.get("entity_mentions", []), window_tokens)
            out_obj["entity_mentions_text"] = [{"id": k, "text": v} for k, v in ent_text_map.items()]
            out_obj["event_mentions_text"] = build_event_texts_exact(sobj.get("event_mentions", []),
                                                                     window_tokens,
                                                                     ent_text_map)

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()