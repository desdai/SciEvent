in baselines/degree:
added: ./degree/config/config_degree_e2e_bofu.json
added: ./data/degree_e2e_bofu (to store data, input, target)
added: ./results (to store results)
added: ./results/....txt (to store f1 scores)

run this (successful, and gave zero pred, due to unseen event types):
python degree/eval_end2endEE.py \
  --e2e_config ./config/config_degree_e2e_bofu.json \
  --e2e_model ./pretrained_models/degree_ace05e_e2e/model.mdl
  --write_file ./results/my_predictions.json


need to create: generate_data_degree_e2e_my_data.py to store pickle files, json of input and targets and vocab in
need to create: in template_generate_ace.py, add our 4 event type classes under scienIE classes
need to create: config_degree_train_custom.json, path to input files, path to output dir, others follow exising config

DATA processing:
Input all_checked_data folder, json files
Run the main.py, get converted
run the split_data.py, get splited

THEN refer to readme.md of degree, for training.

Then use the eval_end2endEE.py for tesing, in there, get EM, simple overlap, scirex overlap, IoU, and 
also role-wise, argument-wise (for ablation)

Remember to change the data from the all_checked_data folder for 4-out-of-5-domain ablation