#!/usr/bin/env python3
# prepare_true.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

KEY_MAP = {
    "Background/Introduction": "Background",
    "Methods/Approach": "Method",
    "Results/Findings": "Results",
    "Conclusions/Implications": "Implications",
    "Implications": "Implications",
}
SECTION_ORDER = ["Background", "Method", "Results", "Implications"]

def load_papers(obj: Any) -> List[Dict[str, Any]]:
    """Return a list of paper dicts from obj (array, dict with papers, or single paper)."""
    if isinstance(obj, list):
        return [p for p in obj if isinstance(p, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("papers"), list):
            return [p for p in obj["papers"] if isinstance(p, dict)]
        if "events" in obj:
            return [obj]
    return []

def extract_sections_from_paper(paper: Dict[str, Any]) -> str:
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_ORDER}
    for ev in paper.get("events", []):
        if not isinstance(ev, dict):
            continue
        section = None
        for k in KEY_MAP:
            if k in ev and ev.get(k) is not None:
                section = KEY_MAP[k]
                break
        if not section:
            continue
        text = ev.get("Text", "")
        text = text.strip() if isinstance(text, str) else ""
        if text:
            sections[section].append(text)

    paper_code = paper.get("paper_code") or paper.get("doc_id") or "UNKNOWN"
    lines: List[str] = [f"Paper Code: {paper_code}", ""]
    for sec in SECTION_ORDER:
        joined = " ".join(t for t in sections[sec] if t) if sections[sec] else "<NONE>"
        lines.append(f"[{sec}]: {joined}")
        lines.append("")
    return "\n".join(lines)

def extract_true_sections(obj: Any) -> str:
    blocks: List[str] = []
    for paper in load_papers(obj):
        blocks.append(extract_sections_from_paper(paper))
    return "\n".join(b.strip() for b in blocks if b).strip() + "\n"

def main():
    ap = argparse.ArgumentParser(description="Prepare TRUE text from cleaned JSONs (all outputs in one flat folder).")
    ap.add_argument("--in_folder", required=True, type=Path,
                    help="Root folder containing cleaned JSONs (subfolders like ACL, BIOINFO, CSCW, DH, JMIR).")
    ap.add_argument("--out_folder", required=True, type=Path,
                    help="Single folder where all .txt outputs will be written.")
    args = ap.parse_args()

    in_root = args.in_folder
    out_root = args.out_folder
    out_root.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_root.rglob("*.json"))
    if not json_files:
        print(f"[INFO] No JSON files found under {in_root}")
        return

    for fp in json_files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")
            continue

        text = extract_true_sections(obj)

        # Flatten: just use base filename, replace .json → .txt
        out_file = out_root / (fp.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print(f"[OK] {fp} -> {out_file}")

    print(f"[DONE] All outputs written to {out_root}")

if __name__ == "__main__":
    main()
