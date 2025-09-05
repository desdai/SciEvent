# SciEvent
Raw data is saved in SciEvent_data/raw folder, it consist of all raw annotations in JSON format. 
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

## [DEGREE](https://github.com/PlusLabNLP/DEGREE/tree/master)

### Data Preprocessing
Please follow these code to preprocess SciEvent data for DEGREE format:

The following code assume your path at the repo's root ```./SciEvent```
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
### Model Training
We adapt from [DEGREE's](https://github.com/PlusLabNLP/DEGREE/tree/master) E2E (End2end) training procedure with modifications. We deeply thank the contribution from the authors of the paper. 

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
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_acl.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_acl.json
```
```bash
# Ablate on BIOINFO:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_bioinfo.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_bioinfo.json
```
```bash
# Ablate on CSCW:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_cscw.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_cscw.json
```
```bash
# Ablate on DH:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_dh.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_dh.json
```
```bash
# Ablate on JMIR:
# Generate data:
python baselines/DEGREE/degree/generate_data_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_jmir.json
# Train:
python baselines/DEGREE/degree/train_degree_scievent.py -c baselines/DEGREE/config/config_degree_no_jmir.json
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

## [OneIE](https://blender.cs.illinois.edu/software/oneie/)


## [EEQA](https://github.com/xinyadu/eeqa/tree/master)




## LLMs (All LLMs share same input format and output format)





# Model Finetuning

# LLM Prompting

