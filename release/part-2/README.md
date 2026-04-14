
# CSCI 2590: Homework 4 Part 2

> ### Please start early. 

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```bash
conda create -n nlp_hw4 python=3.11
conda activate nlp_hw4
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```bash
python evaluate.py
  --predicted_sql results/t5_ft_experiment_dev.sql
  --predicted_records records/t5_ft_experiment_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```

## Training / implementation flow

The Part 2 pipeline is now wired so the same preprocessing options are shared across:
- `load_data.py` for dataset construction
- `compute_stats.py` for the report tables
- `train_t5.py` for finetuning / generation

Baseline finetuning:
```bash
python train_t5.py --finetune --experiment_name baseline
```

Useful optional experiments:
```bash
python train_t5.py \
  --finetune \
  --experiment_name schema_prompt \
  --include_schema_in_input

python train_t5.py \
  --finetune \
  --experiment_name frozen_encoder \
  --freeze_encoder
```

To generate the Table 1 / Table 2 statistics for your report:
```bash
python compute_stats.py
```

If you want the stats to reflect a particular preprocessing configuration, pass the same flags you used in training:
```bash
python compute_stats.py --include_schema_in_input --lowercase_inputs
```

For curated sweeps, start with decode-only experiments on an existing checkpoint:
```bash
python sweep_experiments.py --preset decode_baseline --base_experiment baseline --skip_test_inference
```

Then try the stronger retraining sweep:
```bash
python sweep_experiments.py --preset record_f1_push --base_experiment baseline --skip_test_inference
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)
 
For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in 'data/test.nl' in the order they appear.
