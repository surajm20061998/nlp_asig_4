import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

from utils import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RECORDS_DIR = os.path.join(BASE_DIR, 'records')
DATA_DIR = os.path.join(BASE_DIR, 'data')


PRESETS = {
    'decode_baseline': [
        {
            'name_suffix': 'beam8_top',
            'skip_training': True,
            'args': {
                'num_beams': 8,
                'num_return_sequences': 1,
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'beam8_valid',
            'skip_training': True,
            'args': {
                'num_beams': 8,
                'num_return_sequences': 8,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'beam12_valid',
            'skip_training': True,
            'args': {
                'num_beams': 12,
                'num_return_sequences': 12,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.7,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'beam8_valid_sqlcanon',
            'skip_training': True,
            'args': {
                'num_beams': 8,
                'num_return_sequences': 8,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
                'canonicalize_sql': True,
            },
        },
    ],
    'record_f1_push': [
        {
            'name_suffix': 'lr5e5',
            'args': {
                'learning_rate': 5e-5,
                'batch_size': 4,
                'test_batch_size': 4,
                'max_n_epochs': 12,
                'patience_epochs': 3,
                'mixed_precision': 'bf16',
                'gradient_accumulation_steps': 4,
                'num_beams': 8,
                'num_return_sequences': 4,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'lr3e5',
            'args': {
                'learning_rate': 3e-5,
                'batch_size': 4,
                'test_batch_size': 4,
                'max_n_epochs': 12,
                'patience_epochs': 3,
                'mixed_precision': 'bf16',
                'gradient_accumulation_steps': 4,
                'num_beams': 8,
                'num_return_sequences': 4,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'tables_lr5e5',
            'args': {
                'learning_rate': 5e-5,
                'batch_size': 4,
                'test_batch_size': 4,
                'max_n_epochs': 12,
                'patience_epochs': 3,
                'mixed_precision': 'bf16',
                'gradient_accumulation_steps': 4,
                'schema_prompt_mode': 'tables',
                'num_beams': 8,
                'num_return_sequences': 4,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'tables_sqlcanon_lr5e5',
            'args': {
                'learning_rate': 5e-5,
                'batch_size': 4,
                'test_batch_size': 4,
                'max_n_epochs': 12,
                'patience_epochs': 3,
                'mixed_precision': 'bf16',
                'gradient_accumulation_steps': 4,
                'schema_prompt_mode': 'tables',
                'canonicalize_sql': True,
                'num_beams': 8,
                'num_return_sequences': 4,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
        {
            'name_suffix': 'tables_sqlcanon_lr3e5',
            'args': {
                'learning_rate': 3e-5,
                'batch_size': 4,
                'test_batch_size': 4,
                'max_n_epochs': 12,
                'patience_epochs': 3,
                'mixed_precision': 'bf16',
                'gradient_accumulation_steps': 4,
                'schema_prompt_mode': 'tables',
                'canonicalize_sql': True,
                'num_beams': 8,
                'num_return_sequences': 4,
                'beam_selection_strategy': 'first_valid',
                'length_penalty': 0.8,
                'repetition_penalty': 1.05,
                'no_repeat_ngram_size': 3,
            },
        },
    ],
}


def cli_args_from_dict(arg_dict):
    cli_args = []
    for key, value in arg_dict.items():
        flag = f'--{key}'
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


def evaluate_experiment(experiment_name, model_type='ft'):
    pred_sql = os.path.join(RESULTS_DIR, f't5_{model_type}_{experiment_name}_dev.sql')
    pred_records = os.path.join(RECORDS_DIR, f't5_{model_type}_{experiment_name}_dev.pkl')
    gold_sql = os.path.join(DATA_DIR, 'dev.sql')
    gold_records = os.path.join(RECORDS_DIR, 'ground_truth_dev.pkl')

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gold_sql,
        pred_sql,
        gold_records,
        pred_records,
    )
    error_rate = sum(bool(msg) for msg in error_msgs) / len(error_msgs)
    return {
        'sql_em': sql_em,
        'record_em': record_em,
        'record_f1': record_f1,
        'error_rate': error_rate,
    }


def get_output_paths(experiment_name, model_type='ft'):
    return (
        os.path.join(RESULTS_DIR, f't5_{model_type}_{experiment_name}_dev.sql'),
        os.path.join(RECORDS_DIR, f't5_{model_type}_{experiment_name}_dev.pkl'),
    )


def run_experiment(args, experiment_spec, summary_rows):
    experiment_name = experiment_spec.get('name') or f"{args.experiment_prefix}_{experiment_spec['name_suffix']}"
    train_args = dict(experiment_spec.get('args', {}))
    train_args.setdefault('batch_size', args.batch_size)
    train_args.setdefault('test_batch_size', args.test_batch_size)

    cmd = [sys.executable, 'train_t5.py', '--finetune', '--experiment_name', experiment_name]
    if experiment_spec.get('skip_training', False):
        cmd.append('--skip_training')
        cmd.extend(['--load_from_experiment_name', args.base_experiment])

    if args.skip_test_inference:
        cmd.append('--skip_test_inference')

    cmd.extend(cli_args_from_dict(train_args))

    pred_sql_path, pred_records_path = get_output_paths(experiment_name)
    already_exists = os.path.exists(pred_sql_path) and os.path.exists(pred_records_path)
    if args.skip_existing and already_exists:
        print(f'Skipping existing experiment: {experiment_name}')
    else:
        print()
        print(f'=== Running {experiment_name} ===')
        print('Command:', ' '.join(cmd))
        subprocess.run(cmd, cwd=BASE_DIR, check=True)

    metrics = evaluate_experiment(experiment_name)
    row = {
        'experiment_name': experiment_name,
        'record_f1': metrics['record_f1'],
        'record_em': metrics['record_em'],
        'sql_em': metrics['sql_em'],
        'error_rate': metrics['error_rate'],
        'config': json.dumps(train_args, sort_keys=True),
    }
    summary_rows.append(row)
    print(
        f"{experiment_name}: "
        f"Record F1={metrics['record_f1']:.4f}, "
        f"Record EM={metrics['record_em']:.4f}, "
        f"SQL EM={metrics['sql_em']:.4f}, "
        f"Error rate={metrics['error_rate']:.4f}"
    )


def write_summary(summary_dir, rows):
    rows = sorted(rows, key=lambda row: row['record_f1'], reverse=True)
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, 'summary.csv')
    json_path = os.path.join(summary_dir, 'summary.json')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)

    print()
    print(f'Summary written to {csv_path}')
    print('Top experiments:')
    for row in rows[:5]:
        print(
            f"  {row['experiment_name']}: "
            f"Record F1={row['record_f1']:.4f}, "
            f"Record EM={row['record_em']:.4f}, "
            f"SQL EM={row['sql_em']:.4f}, "
            f"Error rate={row['error_rate']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description='Run curated sweeps for the T5 Text-to-SQL pipeline')
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), required=True)
    parser.add_argument('--base_experiment', default='baseline')
    parser.add_argument('--experiment_prefix', default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--skip_test_inference', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    if args.experiment_prefix is None:
        args.experiment_prefix = f"{args.base_experiment}_{args.preset}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = os.path.join(BASE_DIR, 'sweep_results', f'{args.experiment_prefix}_{timestamp}')

    summary_rows = []
    for experiment_spec in PRESETS[args.preset]:
        run_experiment(args, experiment_spec, summary_rows)

    if summary_rows:
        write_summary(summary_dir, summary_rows)


if __name__ == '__main__':
    main()
