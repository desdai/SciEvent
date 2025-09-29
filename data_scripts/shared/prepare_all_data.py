#!/usr/bin/env python3
"""
reconstruct.py

Reconstruct the ORIGINAL JSONL from:
  1) a annotation file (without verbatim text fields), and
  2) a texts-only file (only the removed verbatim text fields).

Goal: byte-for-byte STRUCTURE equivalence in terms of key ordering for:
- top-level keys,
- entity_mentions item keys,
- event_mentions trigger and argument item keys.

Assumptions (based on the provided schema/example):
Top-level canonical order typically is:
  ["doc_id","wnd_id","entity_mentions","relation_mentions","event_mentions",
   "entity_coreference","event_coreference","tokens","pieces","token_lens",
   "sentence","sentence_starts"]

However, since (1) the annotation file preserves insertion order of remaining keys and
(2) 'tokens'/'pieces'/'sentence' were removed, we re-insert them at the most likely
original positions:
  - "tokens" and "pieces" are inserted *before* "token_lens" (if present), else before "sentence_starts", else at end.
  - "sentence" is inserted *before* "sentence_starts" (if present), else at end.

For nested structures we rebuild dicts using canonical orders seen in the example:
- entity mention item: ["id","text","start","end","entity_type","mention_type", ...extras...]
- trigger: ["text","start","end", ...extras...]
- argument item: ["entity_id","text","role", ...extras...]
- event mention object: ["event_type","id","trigger","arguments", ...extras...]

If your original top-level order differs, consider extending save_texts.py to also store
a "top_key_order" per line and have this script honor it.

Usage:
  python data_scripts/shared/prepare_all_data.py \
    --annotation SciEvent_data/DEGREE/processed/annotation.jsonl \
    --texts      SciEvent_data/DEGREE/processed/texts_only.jsonl \
    --output     SciEvent_data/DEGREE/processed/all_data.json
"""

import argparse
import json
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Optional, Optional, Optional


def _index_of(keys: List[str], name: str) -> int:
    try:
        return keys.index(name)
    except ValueError:
        return -1


def _ordered_insert(base_pairs: List[Tuple[str, Any]],
                    inserts: List[Tuple[str, Any]],
                    before_key: str) -> List[Tuple[str, Any]]:
    """
    Insert (k,v) pairs just BEFORE the first occurrence of before_key.
    If before_key not found, append to the end.
    """
    out = []
    inserted = False
    for k, v in base_pairs:
        if not inserted and k == before_key:
            out.extend(inserts)
            inserted = True
        out.append((k, v))
    if not inserted:
        out.extend(inserts)
    return out


def _as_ordered(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    # Keep current insertion order of a dict (as read by json), expressed as list of pairs
    return list(d.items())


def _rebuild_entity(ent: Dict[str, Any], text_by_id: Dict[str, str]) -> OrderedDict:
    # Canonical order with fallbacks for extras
    # expected keys: id, (text), start, end, entity_type, mention_type, ...
    eid = ent.get("id")
    text = text_by_id.get(eid) if eid is not None else None

    extras_keys = [k for k in ent.keys()
                   if k not in {"id", "start", "end", "entity_type", "mention_type"}]

    out = OrderedDict()
    if "id" in ent: out["id"] = ent["id"]
    if text is not None: out["text"] = text
    if "start" in ent: out["start"] = ent["start"]
    if "end" in ent: out["end"] = ent["end"]
    if "entity_type" in ent: out["entity_type"] = ent["entity_type"]
    if "mention_type" in ent: out["mention_type"] = ent["mention_type"]
    # preserve any extras (in their current order)
    for k in extras_keys:
        out[k] = ent[k]
    return out


def _rebuild_trigger(trg: Dict[str, Any], trigger_text: Optional[str]) -> OrderedDict:
    # Canonical: text, start, end, then extras
    extras_keys = [k for k in trg.keys() if k not in {"text", "start", "end"}]

    out = OrderedDict()
    if trigger_text is not None:
        out["text"] = trigger_text
    if "start" in trg: out["start"] = trg["start"]
    if "end" in trg: out["end"] = trg["end"]
    for k in extras_keys:
        if k not in out:
            out[k] = trg[k]
    return out


def _rebuild_argument(arg: Dict[str, Any], arg_text: Optional[str]) -> OrderedDict:
    # Canonical: entity_id, text, role, then extras
    extras_keys = [k for k in arg.keys() if k not in {"entity_id", "role", "text"}]

    out = OrderedDict()
    if "entity_id" in arg: out["entity_id"] = arg["entity_id"]
    if arg_text is not None: out["text"] = arg_text
    if "role" in arg: out["role"] = arg["role"]
    for k in extras_keys:
        if k not in out:
            out[k] = arg[k]
    return out


def _match_arg_texts(annotation_args: List[Dict[str, Any]],
                     arg_texts: List[Dict[str, Any]]) -> List[Optional[str]]:
    """
    Greedy matching of texts to args by (entity_id, role), in order.
    Returns a list of texts aligned with annotation_args.
    """
    # Build buckets keyed by (entity_id, role) -> list of texts (in order)
    buckets: Dict[Tuple[Any, Any], List[str]] = defaultdict(list)
    for at in arg_texts or []:
        key = (at.get("entity_id"), at.get("role"))
        if at.get("text") is not None:
            buckets[key].append(at["text"])

    out_texts: List[Optional[str]] = []
    for a in annotation_args:
        key = (a.get("entity_id"), a.get("role"))
        lst = buckets.get(key, [])
        if lst:
            out_texts.append(lst.pop(0))
        else:
            out_texts.append(None)
    return out_texts


def _rebuild_event(ev: Dict[str, Any],
                   ev_text_info: Optional[Dict[str, Any]]) -> OrderedDict:
    # Canonical: event_type, id, trigger, arguments, then extras
    # Collect trigger/argument texts
    trig_text = None
    arg_texts_info: List[Dict[str, Any]] = []
    if ev_text_info:
        trig_text = ev_text_info.get("trigger_text")
        arg_texts_info = ev_text_info.get("arguments_text", []) or []

    # Trigger
    trigger = ev.get("trigger", {}) if isinstance(ev.get("trigger"), dict) else {}
    trigger_built = _rebuild_trigger(trigger, trig_text)

    # Arguments
    annotation_args = ev.get("arguments", []) if isinstance(ev.get("arguments"), list) else []
    matched_texts = _match_arg_texts(annotation_args, arg_texts_info)
    built_args: List[OrderedDict] = []
    for a, txt in zip(annotation_args, matched_texts):
        built_args.append(_rebuild_argument(a, txt))

    # Extras at event level (preserve order of any unknown keys)
    extras_keys = [k for k in ev.keys() if k not in {"event_type", "id", "trigger", "arguments"}]

    out = OrderedDict()
    if "event_type" in ev: out["event_type"] = ev["event_type"]
    if "id" in ev: out["id"] = ev["id"]
    out["trigger"] = trigger_built
    out["arguments"] = built_args
    for k in extras_keys:
        out[k] = ev[k]
    return out


def _rebuild_top_level(
    annotation_obj: Dict[str, Any],
    texts_obj: Optional[Dict[str, Any]]
) -> OrderedDict:
    """
    Rebuilds:
      - entity_mentions[].text
      - event_mentions[].trigger.text
      - event_mentions[].arguments[].text
      - reinserts top-level tokens/pieces/sentence at canonical positions
    while preserving order of existing annotation keys.
    """
    # Prepare text maps
    ent_text_map: Dict[str, str] = {}
    if texts_obj and isinstance(texts_obj.get("entity_mentions_text"), list):
        for item in texts_obj["entity_mentions_text"]:
            if isinstance(item, dict) and "id" in item and "text" in item:
                ent_text_map[item["id"]] = item["text"]

    ev_text_map: Dict[str, Dict[str, Any]] = {}
    if texts_obj and isinstance(texts_obj.get("event_mentions_text"), list):
        for ev in texts_obj["event_mentions_text"]:
            if isinstance(ev, dict) and "id" in ev:
                ev_text_map[ev["id"]] = ev

    # Build top-level in a staged way (preserve annotation key order)
    base_pairs = _as_ordered(annotation_obj)

    # First rebuild entity_mentions and event_mentions (so their value objects are fully rebuilt)
    new_pairs: List[Tuple[str, Any]] = []
    for k, v in base_pairs:
        if k == "entity_mentions" and isinstance(v, list):
            rebuilt_ents = [_rebuild_entity(e, ent_text_map) if isinstance(e, dict) else e for e in v]
            new_pairs.append((k, rebuilt_ents))
        elif k == "event_mentions" and isinstance(v, list):
            rebuilt_evs = []
            for ev in v:
                if not isinstance(ev, dict):
                    rebuilt_evs.append(ev)
                    continue
                ev_id = ev.get("id")
                ev_text_info = ev_text_map.get(ev_id, {})
                rebuilt_evs.append(_rebuild_event(ev, ev_text_info))
            new_pairs.append((k, rebuilt_evs))
        else:
            new_pairs.append((k, v))

    # Now re-insert top-level tokens/pieces/sentence using canonical positions
    inserts_tokens_pieces: List[Tuple[str, Any]] = []
    inserts_sentence: List[Tuple[str, Any]] = []

    if texts_obj:
        if "tokens" in texts_obj:
            inserts_tokens_pieces.append(("tokens", texts_obj["tokens"]))
        if "pieces" in texts_obj:
            inserts_tokens_pieces.append(("pieces", texts_obj["pieces"]))
        if "sentence" in texts_obj:
            inserts_sentence.append(("sentence", texts_obj["sentence"]))

    # Insert tokens/pieces before token_lens if possible; else before sentence_starts; else at end
    keys_now = [k for k, _ in new_pairs]
    if inserts_tokens_pieces:
        target = "token_lens" if "token_lens" in annotation_obj else (
                 "sentence_starts" if "sentence_starts" in annotation_obj else None)
        if target:
            new_pairs = _ordered_insert(new_pairs, inserts_tokens_pieces, target)
        else:
            new_pairs.extend(inserts_tokens_pieces)

    # Insert sentence before sentence_starts if possible; else at end
    if inserts_sentence:
        if "sentence_starts" in annotation_obj:
            new_pairs = _ordered_insert(new_pairs, inserts_sentence, "sentence_starts")
        else:
            new_pairs.extend(inserts_sentence)

    # Return as OrderedDict
    out = OrderedDict()
    for k, v in new_pairs:
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", "-s", required=True, help="Path to annotation JSONL")
    ap.add_argument("--texts", "-t", required=True, help="Path to texts-only JSONL")
    ap.add_argument("--output", "-o", required=True, help="Path to write reconstructed JSONL")
    args = ap.parse_args()

    # Load texts-only into a map keyed by (doc_id, wnd_id)
    text_map: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    with open(args.texts, "r", encoding="utf-8") as ft:
        for line in ft:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj.get("doc_id"), obj.get("wnd_id"))
            text_map[key] = obj

    # Stream annotation, reconstruct, and write
    with open(args.annotation, "r", encoding="utf-8") as fs, \
         open(args.output, "w", encoding="utf-8") as fo:
        for line in fs:
            line = line.strip()
            if not line:
                continue
            s_obj = json.loads(line)
            key = (s_obj.get("doc_id"), s_obj.get("wnd_id"))
            t_obj = text_map.get(key, None)

            rebuilt = _rebuild_top_level(s_obj, t_obj)
            fo.write(json.dumps(rebuilt, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
