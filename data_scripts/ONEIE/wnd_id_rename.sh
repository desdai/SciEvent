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

# mkdir -p SciEvent_data/ONEIE/human_subset
# python "$SCRIPT" -input SciEvent_data/DEGREE/human_subset/test_subset.json \
#                  -output SciEvent_data/ONEIE/human_subset/test_subset.oneie.json

# echo "[Step 2] Converting ablation splits..."
# mkdir -p SciEvent_data/ONEIE/ablation/no_acl
# mkdir -p SciEvent_data/ONEIE/ablation/no_bioinfo
# mkdir -p SciEvent_data/ONEIE/ablation/no_cscw
# mkdir -p SciEvent_data/ONEIE/ablation/no_dh
# mkdir -p SciEvent_data/ONEIE/ablation/no_jmir
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_acl/train_without_acl.json \
#                  -output SciEvent_data/ONEIE/ablation/no_acl/train_without_acl.oneie.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_bioinfo/train_without_bioinfo.json \
#                  -output SciEvent_data/ONEIE/ablation/no_bioinfo/train_without_bioinfo.oneie.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_cscw/train_without_cscw.json \
#                  -output SciEvent_data/ONEIE/ablation/no_cscw/train_without_cscw.oneie.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_dh/train_without_dh.json \
#                  -output SciEvent_data/ONEIE/ablation/no_dh/train_without_dh.oneie.json
# python "$SCRIPT" -input SciEvent_data/DEGREE/ablation/no_jmir/train_without_jmir.json \
#                  -output SciEvent_data/ONEIE/ablation/no_jmir/train_without_jmir.oneie.json

echo "[Done] All DEGREE → ONEIE conversions finished."
