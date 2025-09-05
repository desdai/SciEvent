export DYGIEFORMAT_PATH="./processed_data/ace05e_dygieppformat"
export OUTPUT_PATH="./processed_data/ace05e_bart"

echo "🔍 Checking input files in $DYGIEFORMAT_PATH..."
ls -l $DYGIEFORMAT_PATH
echo "--------------------------------------------"

mkdir -p $OUTPUT_PATH
echo "✅ Output directory ensured: $OUTPUT_PATH"
echo "--------------------------------------------"

echo "🔍 Running preprocessing for train.json..."
ls -l $DYGIEFORMAT_PATH/train.json  # Check if input exists
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/train.json -o $OUTPUT_PATH/train.w1.oneie.json -b facebook/bart-large -w 1
ls -l $OUTPUT_PATH/train.w1.oneie.json  # Check if output is created
echo "--------------------------------------------"

echo "🔍 Running preprocessing for dev.json..."
ls -l $DYGIEFORMAT_PATH/dev.json
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/dev.json -o $OUTPUT_PATH/dev.w1.oneie.json -b facebook/bart-large -w 1
ls -l $OUTPUT_PATH/dev.w1.oneie.json
echo "--------------------------------------------"

echo "🔍 Running preprocessing for test.json..."
ls -l $DYGIEFORMAT_PATH/test.json
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/test.json -o $OUTPUT_PATH/test.w1.oneie.json -b facebook/bart-large -w 1
ls -l $OUTPUT_PATH/test.w1.oneie.json
echo "--------------------------------------------"

export BASE_PATH="./processed_data/"
export SPLIT_PATH="./resource/low_resource_split/ace05e"

echo "🔍 Checking train.w1.oneie.json before dataset split..."
ls -l $BASE_PATH/ace05e_bart/train.w1.oneie.json
echo "--------------------------------------------"

python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/ace05e_bart/train.001.w1.oneie.json
ls -l $BASE_PATH/ace05e_bart/train.001.w1.oneie.json  # Check if split file is created
echo "✅ Finished processing train.001.w1.oneie.json"
echo "--------------------------------------------"

# Repeat for other splits
