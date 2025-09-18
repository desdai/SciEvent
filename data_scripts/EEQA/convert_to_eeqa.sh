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

# python "$SCRIPT" -input SciEvent_data/DEGREE/human_subset/test_subset.json \
#                  -output SciEvent_data/EEQA/human_subset/test_subset.eeqa.json

# echo "[Step 2] Converting ablation splits..."
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_acl/train_without_acl.json \
#                  -output SciEvent_data/EEQA/ablation/no_acl/train_without_acl.eeqa.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_bioinfo/train_without_bioinfo.json \
#                  -output  SciEvent_data/EEQA/ablation/no_bioinfo/train_without_bioinfo.eeqa.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_cscw/train_without_cscw.json \
#                  -output  SciEvent_data/EEQA/ablation/no_cscw/train_without_cscw.eeqa.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_dh/train_without_dh.json \
#                  -output  SciEvent_data/EEQA/ablation/no_dh/train_without_dh.eeqa.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_jmir/train_without_jmir.json \
#                  -output  SciEvent_data/EEQA/ablation/no_jmir/train_without_jmir.eeqa.json

echo "[Done] All DEGREE → ONEIE conversions finished."