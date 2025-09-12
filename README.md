# SciEvent
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

We also provide raw abstracts we used, they are in ``SciEvent_data/raw_abstracts`` folder.

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



## LLMs & Human performance 
LLMs are given same input as human to ensure fair evaluation, and all LLMs share same input format and output format, 

### Evaluation:
Run these two files, to first preprocess the raw input:

```bash 
bash baselines/LLM/evaluation_scripts/prepare_for_eval.sh

python baselines/LLM/evaluation_scripts/filter_subset.py \
    --input SciEvent_data/LLM/data/human/human_eval/gold_event_level.json \
    --filter SciEvent_data/DEGREE/human_subset/test_subset.json \
    --output SciEvent_data/LLM/data/human/human_eval/filtered_gold_event_level.json

python baselines/LLM/evaluation_scripts/filter_subset.py \
    --input SciEvent_data/LLM/data/human/human_eval/pred_event_level.json \
    --filter SciEvent_data/DEGREE/human_subset/test_subset.json \
    --output SciEvent_data/LLM/data/human/human_eval/filtered_pred_event_level.json
```

and then evaluate:

```bash baselines/LLM/evaluation_scripts/EM_overlap_eval.sh```

results will be saved to default ```baselines/LLM/LLM_results```

```baselines/LLM/LLM_results/human.txt``` is the default path for human performance, this is compared to previous models' human_subset to demonstrate both finetuned and LLMs baselines's gap with human performance. LLMs' subset results are saved in ```baselines/LLM/LLM_results``` as well, ending with ```-subset.txt```

# LLM Prompting

