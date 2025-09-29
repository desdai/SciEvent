#!/usr/bin/env bash
# convert_to_eeqa.sh
# Convert DEGREE-format JSON to ONEIE-format JSON by renaming wnd_id -> sent_id

set -euo pipefail

SCRIPT="data_scripts/EEQA/convert_to_eeqa.py"

echo "[Step 1] Converting all_splits + human_subset..."
mkdir -p SciEvent_data/EEQA/all_splits
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/train.json \
                 -output SciEvent_data/EEQA/all_splits/train.eeqa.json
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/dev.json \
                 -output SciEvent_data/EEQA/all_splits/dev.eeqa.json
python "$SCRIPT" -input SciEvent_data/DEGREE/all_splits/test.json \
                 -output SciEvent_data/EEQA/all_splits/test.eeqa.json

echo "[Done] All DEGREE → ONEIE conversions finished."