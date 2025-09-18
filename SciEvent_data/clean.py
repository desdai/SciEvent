#!/usr/bin/env python3
"""
Batch-convert annotated JSON files:
- Rename "Main Action" -> "Action"
- In Arguments["Object"] (a dict), REMOVE only the two modifier fields:
    - "Primary Modifier"
    - "Secondary Modifier"
  Keep "Primary Object" and "Secondary Object" untouched.
- Ensure key order inside each event: ... "Text", "Action", "Arguments" ...
Usage:
  python convert_annotations.py --in_dir path/to/input --out_dir path/to/output
"""

import argparse
import json
from pathlib import Path


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []


def drop_object_modifiers(obj_dict):
    """
    Given an Object dict like:
      {
        "Primary Object": ["..."],
        "Primary Modifier": ["..."],
        "Secondary Object": [],
        "Secondary Modifier": []
      }
    Return the same dict but with the two *Modifier* keys removed.
    Ensure remaining object fields are lists.
    """
    if not isinstance(obj_dict, dict):
        return obj_dict  # leave non-dicts unchanged just in case

    new_obj = {}

    if "Primary Object" in obj_dict:
        new_obj["Primary Object"] = _as_list(obj_dict.get("Primary Object"))
    if "Secondary Object" in obj_dict:
        new_obj["Secondary Object"] = _as_list(obj_dict.get("Secondary Object"))

    # Explicitly drop modifiers if they exist
    # (no-op if they aren't present)
    # "Primary Modifier" and "Secondary Modifier" are intentionally omitted.

    return new_obj


def reorder_event_keys(ev, action_value):
    """
    Rebuild 'ev' dict so that:
      - 'Action' appears AFTER 'Text' and BEFORE 'Arguments' (when those exist)
      - All other keys keep their original relative order
    """
    original_items = list(ev.items())
    new_items = []
    inserted_action = False

    for k, v in original_items:
        if k in ("Main Action", "Action"):
            # skip original; we'll reinsert as "Action" at the right spot
            continue

        if k == "Text":
            new_items.append(("Text", v))
            if action_value is not None and not inserted_action:
                new_items.append(("Action", action_value))
                inserted_action = True
            continue

        if k == "Arguments":
            if action_value is not None and not inserted_action:
                new_items.append(("Action", action_value))
                inserted_action = True
            new_items.append(("Arguments", v))
            continue

        new_items.append((k, v))

    if action_value is not None and not inserted_action:
        # No Text/Arguments encountered; append at end
        new_items.append(("Action", action_value))

    ev.clear()
    ev.update(new_items)


def process_events(events):
    if not isinstance(events, list):
        return

    for ev in events:
        if not isinstance(ev, dict):
            continue

        # Capture action value from either key; prefer existing "Action"
        action_val = ev.get("Action")
        if action_val is None:
            action_val = ev.get("Main Action")

        # Transform Arguments.Object -> drop modifiers only
        args = ev.get("Arguments")
        if isinstance(args, dict):
            obj = args.get("Object")
            if isinstance(obj, dict):
                args["Object"] = drop_object_modifiers(obj)

        # Reorder keys so "Action" is after "Text" and before "Arguments"
        reorder_event_keys(ev, action_val)


def process_file(in_path: Path, out_path: Path):
    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Skipping {in_path}: failed to read/parse JSON ({e})")
        return

    papers = data.get("papers")
    if isinstance(papers, list):
        for paper in papers:
            if isinstance(paper, dict):
                process_events(paper.get("events"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Rename keys, drop Object modifiers, and enforce key order in annotated JSON files.")
    ap.add_argument("--in_dir", required=True, type=Path, help="Input directory containing JSON files")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory for converted JSON files")
    ap.add_argument("--glob", default="*.json", help="Glob pattern for input files (default: *.json)")
    args = ap.parse_args()

    files = sorted(args.in_dir.rglob(args.glob))
    if not files:
        print(f"[INFO] No files matched {args.glob} under {args.in_dir}")
        return

    for fp in files:
        rel = fp.relative_to(args.in_dir)
        out_fp = args.out_dir / rel
        process_file(fp, out_fp)
        print(f"[OK] {fp} -> {out_fp}")


if __name__ == "__main__":
    main()
# python SciEvent_data/clean.py --in_dir SciEvent_data/annotated --out_dir SciEvent_data/converted