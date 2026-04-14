import argparse
import os
import pickle

from utils import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--model_type', default='ft', choices=['ft', 'scr'])
    parser.add_argument('--print_paths', action='store_true')
    args = parser.parse_args()

    pred_sql = os.path.join('results', f't5_{args.model_type}_{args.experiment_name}_dev.sql')
    pred_records = os.path.join('records', f't5_{args.model_type}_{args.experiment_name}_dev.pkl')
    gold_sql = os.path.join('data', 'dev.sql')
    gold_records = os.path.join('records', 'ground_truth_dev.pkl')

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gold_sql,
        pred_sql,
        gold_records,
        pred_records,
    )
    error_rate = sum(bool(msg) for msg in error_msgs) / len(error_msgs)

    print(f'Experiment: {args.experiment_name}')
    print(f'Model type: {args.model_type}')
    print(f'Dev SQL EM: {sql_em:.6f}')
    print(f'Dev Record EM: {record_em:.6f}')
    print(f'Dev Record F1: {record_f1:.6f}')
    print(f'Dev SQL error rate: {error_rate:.6f}')

    if args.print_paths:
        print()
        print(f'Predicted SQL path: {pred_sql}')
        print(f'Predicted records path: {pred_records}')


if __name__ == '__main__':
    main()
