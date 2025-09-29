#!/usr/bin/env bash
# rebuild_all.sh
# One-shot: build texts_only (temp) + reconstruct all_data.json with your paths.

set -euo pipefail

# ---- Inputs ----
ABSTRACT_DIR="SciEvent_data/abstracts_texts"
ANNOTATION="SciEvent_data/annotated/annotation.jsonl"
EVENT_SEG="SciEvent_data/annotated/event_seg.jsonl"

# ---- Output ----
ALL_DATA_OUT="SciEvent_data/DEGREE/processed/all_data.json"

# Ensure output directory exists
mkdir -p "$(dirname "$ALL_DATA_OUT")"

# Temp file for texts_only.jsonl (auto-clean on exit)
TMP_TEXTS="$(mktemp -t texts_only.XXXXXX)"
cleanup() { rm -f "$TMP_TEXTS"; }
trap cleanup EXIT

echo "[1/2] Building texts_only (temporary)…"
python data_scripts/shared/prepare_segmentation.py \
  --annotation "$ANNOTATION" \
  --event_seg "$EVENT_SEG" \
  --abstract_dir "$ABSTRACT_DIR" \
  --output "$TMP_TEXTS"

echo "[2/2] Reconstructing all_data.json…"
python data_scripts/shared/prepare_all_data.py \
  --annotation "$ANNOTATION" \
  --texts      "$TMP_TEXTS" \
  --output     "$ALL_DATA_OUT"

echo "Done → $ALL_DATA_OUT"
