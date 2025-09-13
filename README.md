# SciEvent (EMNLP-2025)

Raw annotated data is saved in ``SciEvent_data/raw`` folder, it consist of all raw annotations in JSON format. 
The SciEvent dataset contains 500 scientific abstracts across 5 venues of different domains:
- 100 abstracts from the Association for Computational Linguistics (ACL), Natural Language Processing
- 100 abstracts from the ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW), Social Computing
- 100 abstracts from Bioinformatics (BIOINFO), Computational Biology
- 120 abstracts from Digital Humanities Quarterly (DH), Digital Humanities
- 80 abstracts from the Journal of Medical Internet Research (JMIR), Medical Informatics

Each file start with the venue abbreviation for classification. Each file includes two abstracts,
and each abstract is annotated with:
- Unique paper code
- Event segments (Background, Methods, Results, Implications)
- Trigger tuples (Agent, Main Action, Object)
- Nine argument roles (Context, Purpose, Method, Results, Analysis, Challenge, Ethical, Implications, Contradictions)

Note that the blank entries means there is no such argument, we enforce such empty arguments to maintain structured format.

We also provide raw abstracts we used, they are in ``SciEvent_data/raw_abstracts`` folder. Unannotated JSON for each venue are also provided in ``SciEvent_data/raw/domain_specific_unannotated`` folder, you can load each of these JSON files on the Render interface for annotation.

# Tuning-Based Models:
Our chosen tuning-based baselines are [DEGREE](https://github.com/PlusLabNLP/DEGREE/tree/master), [OneIE's](https://blender.cs.illinois.edu/software/oneie/) and [EEQA's](https://github.com/xinyadu/eeqa/tree/master), which represented the state-of-the-art event extraction models at the time of this work. We adapt from [DEGREE's E2E (End2end)](https://github.com/PlusLabNLP/DEGREE/tree/master), [OneIE's](https://blender.cs.illinois.edu/software/oneie/) and [EEQA's](https://github.com/xinyadu/eeqa/tree/master) training and evaluation procedures with modifications. We deeply thank the contribution from the authors of these papers. 

## [DEGREE](https://github.com/PlusLabNLP/DEGREE/tree/master)

The following code assume your path at the repo's root ```./SciEvent```

Setup virtual env: ```conda env create -f degree.yml```

### Data Preprocessing

Please follow these code to preprocess SciEvent data for DEGREE format:

```bash
python data_scripts/DEGREE/prepare_for_DEGREE.py
# By default input path will be SciEvent_data/raw and output "all_data.json" will be save at SciEvent_data/DEGREE/processed
# OR you can alter the input and output directory with following: 
python data_scripts/DEGREE/prepare_for_DEGREE.py --input dir_to_your_raw_input_folder --output dir_to_your_output_folder

# Split "all_data.json" for training, validation and testing
python data_scripts/DEGREE/split_data.py
# Default output directory at SciEvent_data/DEGREE/all_splits
# Output names will be train.json, dev.json, test.json.
```
We also provid domain specific data and vocab file in ```SciEvent_data/DEGREE/processed```

To prepare for ablation study, we remove domain data from the training split with following code:
```bash
# Default input JSON file "SciEvent_data/DEGREE/all_splits/train.json"
# Default folder to save outputs "SciEvent_data/DEGREE/ablation"
python data_scripts/DEGREE/ablate_by_domain.py
```
### Training

Use following commands for training DEGREE on SciEVENT data:

Generate data for DEGREE
```bash
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent.json
```

Train DEGREE
```bash
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent.json
```

The model will be stored at ```baselines/DEGREE/output/degree_e2e_scievent/full_data/[timestamp]/best_model.mdl``` in default.

### Ablation training:

Very similar to above, but some input and output path are different on the config, so make sure the correct config is used:
```bash
# Ablate on ACL:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_acl.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_acl.json
```
```bash
# Ablate on BIOINFO:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_bioinfo.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_bioinfo.json
```
```bash
# Ablate on CSCW:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_cscw.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_cscw.json
```
```bash
# Ablate on DH:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_dh.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_dh.json
```
```bash
# Ablate on JMIR:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_jmir.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_scievent_no_jmir.json
```

### Evaluation
Evaluate with following code:
```bash
python baselines/DEGREE/degree/eval_scievent.py -c baselines/DEGREE/config/config_degree_scievent.json -e [path_to_your_mdl]
```
Of course, you can use any ablation model you trained above as well:
```bash
python baselines/DEGREE/degree/eval_scievent.py -c [path_to_your_ablation_config] -e [path_to_your_ablation_mdl]
```
To compare with human performance, we use same model as fully trained (not ablation) model, but only evalute on a subset of data
```bash
python baselines/DEGREE/degree/eval_scievent.py -c baselines/DEGREE/config/config_degree_scievent_subset.json -e [same_mdl_for_full_data_trainig]
```

## [OneIE](https://blender.cs.illinois.edu/software/oneie/)

The following code assume your path at the repo's root ```./SciEvent```

Setup virtual env: ```conda env create -f oneie.yml```

### Data Preprocessing
Please follow these code to preprocess SciEvent data for ONEIE format:

OneIE and DEGREE share similar input structure, we only need to rename ```wnd_id``` into ```sent_id```. Following will prepare all split and ablated training data.
```bash
bash data_scripts/ONEIE/wnd_id_rename.sh
```

### Training
Use following for training
```bash
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent.json
```
Or if you want to train ablated model(s):
```bash
# ONLY TRAIN ONE AT A TIME!!!!
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent_no_acl.json
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent_no_bioinfo.json
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent_no_cscw.json
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent_no_dh.json
python baselines/ONEIE/train.py -c baselines/ONEIE/config/config_oneie_scievent_no_jmir.json
```

### Evaluations
Full data:
```bash
python baselines/ONEIE/json_to_txt_for_pred.py --input_jsonl baselines/ONEIE/output/scievent/full_data/20250421_140654/result.test.json --output_dir baselines/ONEIE/txt/full_data
python baselines/ONEIE/predict.py -m baselines/ONEIE/output/scievent/full_data/20250421_140654/best.role.mdl -i baselines/ONEIE/txt/full_data -o baselines/ONEIE/output_json/full_data --format txt
python baselines/ONEIE/convert_oneie_to_degree_json.py --pred_dir baselines/ONEIE/output_json/full_data --output_file baselines/ONEIE/eval_data/full_data/oneie_preds_degree_format.json
python baselines/ONEIE/EM_overlap_eval.py --pred baselines/ONEIE/eval_data/full_data/oneie_preds_degree_format.json --gold SciEvent_data/ONEIE/all_splits/test.oneie.json
```

Ablation:
```bash
python baselines/ONEIE/json_to_txt_for_pred.py --input_jsonl [result.test.json_in_output_folder] --output_dir [txt_file_dir]
python baselines/ONEIE/predict.py -m [Ablation_mdl] -i [txt_file_dir] -o [output_json_dir] --format txt
python baselines/ONEIE/convert_oneie_to_degree_json.py --pred_dir [output_json_dir] --output_file [json_in_degree_format_output]
python baselines/ONEIE/EM_overlap_eval.py --pred [json_in_degree_format_output] --gold [ground_truth]

# use no ACL as example:
python baselines/ONEIE/json_to_txt_for_pred.py --input_jsonl baselines/ONEIE/output/scievent/no_acl/20250426_231744/result.test.json --output_dir baselines/ONEIE/txt/no_acl
python baselines/ONEIE/predict.py -m baselines/ONEIE/output/scievent/no_acl/20250426_231744/best.role.mdl -i baselines/ONEIE/txt/no_acl -o baselines/ONEIE/output_json/no_acl --format txt
python baselines/ONEIE/convert_oneie_to_degree_json.py --pred_dir baselines/ONEIE/output_json/no_acl --output_file baselines/ONEIE/eval_data/no_acl/oneie_preds_degree_format.json
python baselines/ONEIE/EM_overlap_eval.py --pred baselines/ONEIE/eval_data/no_acl/oneie_preds_degree_format.json --gold SciEvent_data/ONEIE/all_splits/test.oneie.json
```

Subset performance (to compare with human):
```bash
python data_scripts/ONEIE/filter.py -filter SciEvent_data/HUMAN/human_annotation.json -input baselines/ONEIE/txt/full_data -output baselines/ONEIE/txt/human_subset
python baselines/ONEIE/predict.py -m baselines/ONEIE/output/scievent/full_data/20250421_140654/best.role.mdl -i baselines/ONEIE/txt/human_subset -o baselines/ONEIE/output_json/human_subset --format txt
python baselines/ONEIE/convert_oneie_to_degree_json.py --pred_dir baselines/ONEIE/output_json/human_subset --output_file baselines/ONEIE/eval_data/human_subset/oneie_preds_degree_format.json
python baselines/ONEIE/EM_overlap_eval.py --pred baselines/ONEIE/eval_data/human_subset/oneie_preds_degree_format.json --gold SciEvent_data/ONEIE/human_subset/test_subset.oneie.json
```

## [EEQA](https://github.com/xinyadu/eeqa/tree/master)
The following code assume your path at the repo's root ```./SciEvent```

Setup virtual env: ```conda env create -f eeqa.yml```




### Data Preprocessing
Please follow these code to preprocess SciEvent data for ONEIE format:

We are also using DEGREE format and convert them into EEQA input format, splits and ablation data will be saved to ```SciEvent_data/EEQA``` in default

```bash
bash data_scripts/EEQA/convert_to_eeqa.sh
```

### Train and evaluate models

**Trigger Detection**

``` bash baselines/EEQA/code/script_trigger_qa.sh ```

Default output path: ```baselines/EEQA/scievent_trigger_qa_output/full_data```

Ablations:

Using acl ablation as example: 
``` bash baselines/EEQA/code/script_trigger_qa_no_acl.sh ```

Default output path: ```baselines/EEQA/scievent_trigger_qa_output/no_acl```

You can change acl to other venues to train other ablations.


**Argument Extraction**
SciEvent uses the best performing template and setting reported by EEQA and after preliminary experiment. 

Full data With dynamic threshold:
  
```bash baselines/EEQA/code/script_args_qa_thresh.sh```

Default output path: ```baselines/EEQA/scievent_args_qa_thresh_output/full_data```

Ablations:

```bash baselines/EEQA/code/script_args_qa_thresh_no_acl.sh```

Default output path: ```baselines/EEQA/scievent_args_qa_thresh_output/no_acl```


**Evaluations**

Full data:

```bash
python baselines/EEQA/scievent_eval/prepare_for_eval.py \
  -arg_raw baselines/EEQA/scievent_args_qa_thresh_output/full_data/best_args/arg_predictions.json \
  -trig_pred baselines/EEQA/scievent_trigger_qa_output/full_data/best/trigger_predictions.json \
  -gold SciEvent_data/EEQA/all_splits/test.eeqa.json \
  -final_out baselines/EEQA/scievent_eval/full_data/arg_predictions_final.json

python baselines/EEQA/scievent_eval/EM_overlap_eval.py --pred baselines/EEQA/scievent_eval/full_data/arg_predictions_final.json --gold SciEvent_data/EEQA/all_splits/test.eeqa.json
```

Ablations (use no_acl as example, please change for no_[venu_name] for other ablations):

```bash
python baselines/EEQA/scievent_eval/prepare_for_eval.py \
  -arg_raw baselines/EEQA/scievent_args_qa_thresh_output/no_acl/best_args/arg_predictions.json \
  -trig_pred baselines/EEQA/scievent_trigger_qa_output/no_acl/best/trigger_predictions.json \
  -gold SciEvent_data/EEQA/all_splits/test.eeqa.json \
  -final_out baselines/EEQA/scievent_eval/no_acl/arg_predictions_final.json

python baselines/EEQA/scievent_eval/EM_overlap_eval.py --pred baselines/EEQA/scievent_eval/no_acl/arg_predictions_final.json --gold SciEvent_data/EEQA/all_splits/test.eeqa.json
```

Subset data to compare with human performance:

```bash
# AFTER running the full data evaluation:
python baselines/EEQA/scievent_eval/prepare_for_subset.py

python baselines/EEQA/scievent_eval/EM_overlap_eval.py --pred baselines/EEQA/scievent_eval/human_subset/arg_predictions_subset.json --gold SciEvent_data/EEQA/human_subset/test_subset.eeqa.json
```



# LLMs & Human performance 
LLMs are given same input as human to ensure fair evaluation, and all LLMs share same input format and output format. 

## Prompting

This section provides comprehensive guidance on running the SciEvent Large Language Model (LLM) pipeline for scientific event extraction tasks.

The pipeline consists of two main tasks:
1. **Event Segmentation** - Breaking abstracts into semantic segments
2. **Event Extraction** - Extracting argument structures from events

Each task requires running a script to generate raw output, followed by a post-processing script to extract structured results.

---

### Installation

Install the required dependencies:

```bash
# Install dependencies using pip
pip install -r requirements.txt
```

---

### 1. Event Segmentation

**Step 1a: Generate Raw Segmentation Output**

Choose one approach:

**Open-source model:**

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``

```bash
python baselines/LLM/scripts/Event_segmentation.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompt-template Zero-Shot_Event_Segmentation \
  --output-base-dir baselines/LLM/output/Event_Segmentation \
  --input-dir SciEvent_data/raw/domain_specific_unannotated \
  --clean-cache
```


**OpenAI model:**
```bash
python baselines/LLM/scripts/Event_segmentation_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt-template Zero-Shot_Event_Segmentation \
  --output-base-dir baselines/LLM/output/Event_Segmentation \
  --input-dir SciEvent_data/raw/domain_specific_unannotated \
  --api-key "YOUR_API_KEY" \
  --max-concurrent 5
```

**Step 1b: Process Raw Segmentation Output**

After generating raw output, extract the structured segments:

Here the supported model names is the sub-directory of above model names, namely: ``"Qwen2.5-7B-Instruct"``, ``"Meta-Llama-3.1-8B-Instruct"``, ``"DeepSeek-R1-Distill-Llama-8B"`` or ``gpt-4.1``
```bash
python baselines/LLM/scripts/LLM_ES_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model-name Meta-Llama-3.1-8B-Instruct \
  --prompt-template Zero-Shot_Event_Segmentation \
  --base-dir ./baselines/LLM/output/Event_Segmentation \
  --input-dir ./SciEvent_data/raw/domain_specific_unannotated
```

### 2. Event Extraction

**Step 2a: Generate Raw Extraction Output**

Choose one approach:

**Standard Event Extraction (open-source):**

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``

Prompts choices are any prompt ending with ``_Event_Extraction.txt`` in this folder ``baselines/LLM/prompts``

```bash
python baselines/LLM/scripts/Event_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct"\
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir baselines/LLM/output/Event_Extraction \
  --input-dir SciEvent_data/raw/domain_specific_unannotated
```

**Standard Event Extraction (OpenAI):**
```bash
python baselines/LLM/scripts/Event_extraction_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir baselines/LLM/output/Event_Extraction \
  --input-dir SciEvent_data/raw/domain_specific_unannotated
  --api-key "YOUR_API_KEY" \
  --max-concurrent 5
```

#### Event Type Ablation

Following use same input and output folder as above.

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`` if ``MODEL_TYPE = huggingface``.

Only ``gpt-4.1``is experimented using ``MODEL_TYPE = openai``. If the API call supports, you can change into any other LLMs.

**Predicting Event Type and Extraction:**
```bash
  python baselines/LLM/scripts/Pred_Event_Type.py \
    --domains ACL BIOINFO CSCW DH JMIR \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --model-type huggingface \
    --prompt "Pred_Event_Type" \
    --clean-cache
```

**Providing True Event Type and Extraction:**
```bash
  python baselines/LLM/scripts/True_Event_type.py \
    --domains ACL BIOINFO CSCW DH JMIR \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --model-type huggingface \
    --prompt "True_Event_Type" \
    --clean-cache
```

**Step 2b: Process Raw Extraction Output**

All event extraction approaches generate raw text files that must be processed into structured JSON format:

```bash
python baselines/LLM/scripts/LLM_EE_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --prompt-template [PROMPT_TEMPLATE] \
  --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Where `[PROMPT_TEMPLATE]` should match the prompt template used in Step 2a:
- For Event_extraction.py or Event_extraction_openai.py: Use `"One-Shot_Event_Extraction"` or other extraction templates
- For Pred_Event_Type.py: Use `"Zero-shot_Pred_Event_Type"`
- For True_Event_type.py: Use `"Zero-shot_True_Event_Type"`
Default input folder: ``./SciEvent_data/raw/domain_specific_unannotated``, default base folder: ``./baselines/LLM/output/Event_Extraction``

### LLM Evaluation:

We collected the above output into data folder ``SciEvent_data/LLM`` for your easier access, the data is identical. We deliberately keep data in both LLM output path ``baselines/LLM/output`` and data storage path ``SciEvent_data/LLM`` to allow skipping prompting and directly running evaluation.

You can verify these by checking the JSON files under the same model and prompting method.

### 1. Event Segmentation

First run ``bash baselines/LLM/segmentation_eval/segmentation_prepare.sh `` to prepare input evaluation.

Then run ``bash baselines/LLM/segmentation_eval/segmentation_eval.sh`` to evaluate. 

Results are by default in ``baselines/LLM/LLM_results/Event_Segmentation``.


### 2. Event Extraction
Run these two files, to first preprocess the raw input:

```bash 
bash baselines/LLM/extraction_eval/prepare_for_eval.sh

python baselines/LLM/extraction_eval/filter_subset.py \
    --input SciEvent_data/LLM/Event_Extraction/human/human_eval/gold_event_level.json \
    --filter SciEvent_data/DEGREE/human_subset/test_subset.json \
    --output SciEvent_data/LLM/Event_Extraction/human/human_eval/filtered_gold_event_level.json

python baselines/LLM/extraction_eval/filter_subset.py \
    --input SciEvent_data/LLM/Event_Extraction/human/human_eval/pred_event_level.json \
    --filter SciEvent_data/DEGREE/human_subset/test_subset.json \
    --output SciEvent_data/LLM/Event_Extraction/human/human_eval/filtered_pred_event_level.json
```

and then evaluate:

```bash baselines/LLM/extraction_eval/EM_overlap_eval.sh```

results will be saved to default ```baselines/LLM/LLM_results/Event_Extraction```

```baselines/LLM/LLM_results/Event_Extraction/human.txt``` is the default path for human performance, this is compared to previous models' human_subset to demonstrate both finetuned and LLMs baselines's gap with human performance. LLMs' subset results are saved in ```baselines/LLM/LLM_results/Event_Extraction``` as well, ending with ```-subset.txt```
