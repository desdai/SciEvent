# SciEvent (EMNLP-2025)

The SciEvent dataset contains 500 scientific abstracts across 5 venues of different domains:
- 100 abstracts from the Association for Computational Linguistics (ACL), Natural Language Processing
- 100 abstracts from the ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW), Social Computing
- 100 abstracts from Bioinformatics (BIOINFO), Computational Biology
- 120 abstracts from Digital Humanities Quarterly (DH), Digital Humanities
- 80 abstracts from the Journal of Medical Internet Research (JMIR), Medical Informatics


We provide segementation and annotation in ``SciEvent_data/annotated`` folder. We also provide raw abstracts that are invovled in the experiments (except for CSCW's, due to license limitation) we used, they are in ``SciEvent_data/abstract_texts`` folder.

# Data Attribution:
We give attribution to all abstracts that contribute to building this work in the ``SciEvent_data/metadata/SciEvent_metadata.csv`` file. We acknolwedge the titles, authors, doi (if applicable), url and license.

Please note, for all following baseline models and LLMs, we cannot provide abstracts for CSCW due to license limitations, and if you want to reproduce exact results as the paper, please prepare these abstracts yourself. You can follow the "doc_id" and "url" in the ``SciEvent_data/metadata/cscw_used_abstracts.xlsx`` and download the corresponding abstracts. We highly recommend download these abstracts. If you encounter any trouble, please contact bofudong@iu.edu.

# Tuning-Based Models:
Our chosen tuning-based baselines are [DEGREE](https://github.com/PlusLabNLP/DEGREE/tree/master), [OneIE's](https://blender.cs.illinois.edu/software/oneie/) and [EEQA's](https://github.com/xinyadu/eeqa/tree/master), which represented the state-of-the-art event extraction models at the time of this work. We adapt from [DEGREE's E2E (End2end)](https://github.com/PlusLabNLP/DEGREE/tree/master), [OneIE's](https://blender.cs.illinois.edu/software/oneie/) and [EEQA's](https://github.com/xinyadu/eeqa/tree/master) training and evaluation procedures with modifications. We deeply thank the contribution from the authors of these papers. 

The following code assume your path at the Repository's root ```./SciEvent```

## [DEGREE](https://github.com/PlusLabNLP/DEGREE/tree/master)

Setup virtual env: ```conda env create -f degree.yml```

### Data Preprocessing

Please follow these code to preprocess SciEvent data for DEGREE format:

```bash
bash data_scripts/shared/prepare_data.sh
# By default output "all_data.json" will be save at SciEvent_data/DEGREE/processed

# Split "all_data.json" for training, validation and testing
python data_scripts/shared/split_data.py
# Default output directory at SciEvent_data/DEGREE/all_splits
# Output names will be "train.json", "dev.json", and "test.json".
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

If the loading of model errors, please make sure that ``use_cdn = False`` in the ``envs/DEGREE/lib/python3.8/site-packages/transformers/modeling_utils.py`` file. This will use back-up source to load model, since the cdn is no longer supported by HuggingFace. You will need to do the same for OneIE or EEQA environment if you experience same issue for their training.

### Evaluation
Evaluate with following code:
```bash
python baselines/DEGREE/degree/eval_scievent.py -c baselines/DEGREE/config/config_degree_scievent.json -e [path_to_your_mdl]
```

## [OneIE](https://blender.cs.illinois.edu/software/oneie/)

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

### Evaluations
Full data:
```bash
python baselines/ONEIE/json_to_txt_for_pred.py --input_jsonl baselines/ONEIE/output/scievent/full_data/[model_timestamp]/result.test.json --output_dir baselines/ONEIE/txt/full_data
python baselines/ONEIE/predict.py -m baselines/ONEIE/output/scievent/full_data/[model_timestamp]/best.role.mdl -i baselines/ONEIE/txt/full_data -o baselines/ONEIE/output_json/full_data --format txt
python baselines/ONEIE/convert_oneie_to_degree_json.py --pred_dir baselines/ONEIE/output_json/full_data --output_file baselines/ONEIE/eval_data/full_data/oneie_preds_degree_format.json
python baselines/ONEIE/EM_overlap_eval.py --pred baselines/ONEIE/eval_data/full_data/oneie_preds_degree_format.json --gold SciEvent_data/ONEIE/all_splits/test.oneie.json
```

## [EEQA](https://github.com/xinyadu/eeqa/tree/master)

Setup virtual env: ```conda env create -f eeqa.yml```


### Data Preprocessing
Please follow these code to preprocess SciEvent data for ONEIE format:

We are also using DEGREE format and convert them into EEQA input format, splits and ablation data will be saved to ```SciEvent_data/EEQA``` in default

```bash
bash data_scripts/EEQA/convert_to_eeqa.sh
```

### Train and evaluate models

**Trigger Detection:**

```bash
bash baselines/EEQA/code/script_trigger_qa.sh
```

Default output path: ```baselines/EEQA/scievent_trigger_qa_output/full_data```

**Argument Extraction:**

SciEvent uses the best performing template and setting reported by EEQA and after preliminary experiment. 

Full data With dynamic threshold:
  
```bash
bash baselines/EEQA/code/script_args_qa_thresh.sh
```

Default output path: ```baselines/EEQA/scievent_args_qa_thresh_output/full_data```

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


# LLMs  performance 

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
  --output-base-dir SciEvent_data/LLM/Event_Segmentation \
  --input-dir SciEvent_data/to_be_annotated \
  --clean-cache
```

**OpenAI model:**
```bash
python baselines/LLM/scripts/Event_segmentation_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt-template Zero-Shot_Event_Segmentation \
  --output-base-dir SciEvent_data/LLM/Event_Segmentation \
  --input-dir SciEvent_data/to_be_annotated \
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
  --base-dir SciEvent_data/LLM/Event_Segmentation \
  --input-dir SciEvent_data/to_be_annotated
```

### 2. Event Extraction

**Step 2a: Generate Raw Extraction Output**

Choose one approach:

**Standard Event Extraction (open-source):**

Model choices are: ``"Qwen/Qwen2.5-7B-Instruct"``, ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``, ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``

Prompts choices are `"Zero-Shot_Event_Extraction"`, `"One-Shot_Event_Extraction"`, `"Few-shot-5_Event_Extraction"` or `"Few-shot-2_Event_Extraction"` in this folder ``baselines/LLM/prompts``

```bash
python baselines/LLM/scripts/Event_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct"\
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir SciEvent_data/LLM/Event_Extraction \
  --input-dir SciEvent_data/to_be_annotated
```

**Standard Event Extraction (OpenAI):**
```bash
python baselines/LLM/scripts/Event_extraction_openai.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --model "gpt-4.1" \
  --prompt "Few-shot-2_Event_Extraction" \
  --output-base-dir SciEvent_data/LLM/Event_Extraction \
  --input-dir SciEvent_data/to_be_annotated
  --api-key "YOUR_API_KEY" \
  --max-concurrent 5
```

#### Event Type Ablation

Following use same input and output folder as above. This is to see how model perform if they are asked to consider the event type information when extracting. 

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

All event extraction approaches generate raw text files that are processed into structured JSON format:

```bash
python baselines/LLM/scripts/LLM_EE_extraction.py \
  --domains ACL BIOINFO CSCW DH JMIR \
  --prompt-template [PROMPT_TEMPLATE] \
  --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Where `[PROMPT_TEMPLATE]` should match the prompt template used in Step 2a:
- For Event_extraction.py or Event_extraction_openai.py: Use `"Zero-Shot_Event_Extraction"`, `"One-Shot_Event_Extraction"`, `"Few-shot-5_Event_Extraction"` or `"Few-shot-2_Event_Extraction"`
- For Pred_Event_Type.py: Use `"Zero-shot_Pred_Event_Type"`
- For True_Event_type.py: Use `"Zero-shot_True_Event_Type"`

Default input folder: ``SciEvent_data/to_be_annotated``, default base folder: ``./SciEvent_data/LLM/Event_Extraction``

### LLM Evaluation:

We collected the above output into data folder ``SciEvent_data/LLM`` for your easier access, the data is identical. We deliberately keep data in both LLM output path ``baselines/LLM/output`` and data storage path ``SciEvent_data/LLM`` to allow skipping prompting and directly running evaluation.

You can verify these by checking the JSON files under the same model and prompting method.

### 1. Event Segmentation

First prepare input evaluation: 
```bash
bash baselines/LLM/segmentation_eval/segmentation_prepare.sh
```

Then evaluate:
```bash
bash baselines/LLM/segmentation_eval/segmentation_eval.sh
```

Results are by default in ``baselines/LLM/LLM_results/Event_Segmentation``.


### 2. Event Extraction
First preprocess the raw input (we assume you run all prompts templates, but you can edit the shell scripts to remove certain prompts template, also please make sure the prompt template names used above mactch in these shell scripts):

```bash
bash baselines/LLM/extraction_eval/prepare_for_eval.sh
```

Then evaluate:

```bash
bash baselines/LLM/extraction_eval/EM_overlap_eval.sh
```

results will be saved to default ```baselines/LLM/LLM_results/Event_Extraction```

# Annotation Tool
Following is the link to our annotation tool: ``https://annotation-demo.onrender.com/`` you can annotate data by loading any of the unannotated JSON in this path ``SciEvent_data/to_be_annotated``, but you need to first creat an account to use it.
