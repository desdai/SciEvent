#!/usr/bin/env bash
# wnd_id_rename.sh
# Convert DEGREE-format JSON to ONEIE-format JSON by renaming wnd_id -> sent_id

set -euo pipefail

SCRIPT="data_scripts/ONEIE/wnd_id_rename.py"

echo "[Step 1] Converting all_splits + human_subset..."
mkdir -p SciEvent_data/ONEIE/all_splits
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/train.json \
                 -output SciEvent_data/ONEIE/all_splits/train.oneie.json
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/dev.json \
                 -output SciEvent_data/ONEIE/all_splits/dev.oneie.json
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/test.json \
                 -output SciEvent_data/ONEIE/all_splits/test.oneie.json

echo "[Done] All DEGREE → ONEIE conversions finished."
