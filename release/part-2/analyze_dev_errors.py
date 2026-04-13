import argparse
import pickle
from collections import Counter

from utils import read_queries
from load_data import load_lines


def normalize_sql(query):
    return ' '.join(query.strip().split())


def get_counter_stats(dev_nl, gold_sql, pred_sql, gold_records, pred_records):
    total = len(dev_nl)

    exact_sql_errors = 0
    empty_prediction = 0
    syntax_error = 0
    wrong_table_hint = 0
    wrong_aggregation_hint = 0
    partial_record_mismatch = 0

    examples = {
        'syntax_error': None,
        'wrong_table_hint': None,
        'wrong_aggregation_hint': None,
        'partial_record_mismatch': None,
    }

    for idx, (nl, gold_q, pred_q, gold_rec_pack, pred_rec_pack) in enumerate(
        zip(dev_nl, gold_sql, pred_sql, gold_records[0], pred_records[0])
    ):
        gold_q_norm = normalize_sql(gold_q)
        pred_q_norm = normalize_sql(pred_q)
        if gold_q_norm != pred_q_norm:
            exact_sql_errors += 1

        if pred_q_norm == '':
            empty_prediction += 1

        pred_error_msg = pred_records[1][idx]
        if pred_error_msg:
            syntax_error += 1
            if examples['syntax_error'] is None:
                examples['syntax_error'] = (idx, nl, gold_q, pred_q, pred_error_msg)

        if ('count' in gold_q_norm.lower() or 'max' in gold_q_norm.lower() or 'min' in gold_q_norm.lower()) and (
            'count' not in pred_q_norm.lower() and 'max' not in pred_q_norm.lower() and 'min' not in pred_q_norm.lower()
        ):
            wrong_aggregation_hint += 1
            if examples['wrong_aggregation_hint'] is None:
                examples['wrong_aggregation_hint'] = (idx, nl, gold_q, pred_q, pred_error_msg)

        gold_tokens = set(token.lower() for token in gold_q_norm.replace(',', ' ').split())
        pred_tokens = set(token.lower() for token in pred_q_norm.replace(',', ' ').split())
        table_like_gold = {tok for tok in gold_tokens if tok in {'flight', 'fare', 'airport', 'city', 'days', 'date_day', 'state'}}
        table_like_pred = {tok for tok in pred_tokens if tok in {'flight', 'fare', 'airport', 'city', 'days', 'date_day', 'state'}}
        if table_like_gold and table_like_gold != table_like_pred:
            wrong_table_hint += 1
            if examples['wrong_table_hint'] is None:
                examples['wrong_table_hint'] = (idx, nl, gold_q, pred_q, pred_error_msg)

        if set(gold_rec_pack) != set(pred_rec_pack) and len(set(gold_rec_pack).intersection(set(pred_rec_pack))) > 0:
            partial_record_mismatch += 1
            if examples['partial_record_mismatch'] is None:
                examples['partial_record_mismatch'] = (idx, nl, gold_q, pred_q, pred_error_msg)

    return {
        'total': total,
        'exact_sql_errors': exact_sql_errors,
        'empty_prediction': empty_prediction,
        'syntax_error': syntax_error,
        'wrong_table_hint': wrong_table_hint,
        'wrong_aggregation_hint': wrong_aggregation_hint,
        'partial_record_mismatch': partial_record_mismatch,
        'examples': examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_sql', required=True)
    parser.add_argument('--predicted_records', required=True)
    parser.add_argument('--development_sql', default='data/dev.sql')
    parser.add_argument('--development_nl', default='data/dev.nl')
    parser.add_argument('--development_records', default='records/ground_truth_dev.pkl')
    args = parser.parse_args()

    dev_nl = load_lines(args.development_nl)
    gold_sql = read_queries(args.development_sql)
    pred_sql = read_queries(args.predicted_sql)

    with open(args.development_records, 'rb') as f:
        gold_records = pickle.load(f)
    with open(args.predicted_records, 'rb') as f:
        pred_records = pickle.load(f)

    stats = get_counter_stats(dev_nl, gold_sql, pred_sql, gold_records, pred_records)

    print(f"Total dev examples: {stats['total']}")
    print(f"SQL mismatch count: {stats['exact_sql_errors']}/{stats['total']}")
    print(f"Empty prediction count: {stats['empty_prediction']}/{stats['total']}")
    print(f"Syntax/runtime SQL error count: {stats['syntax_error']}/{stats['total']}")
    print(f"Wrong-table hint count: {stats['wrong_table_hint']}/{stats['total']}")
    print(f"Wrong-aggregation hint count: {stats['wrong_aggregation_hint']}/{stats['total']}")
    print(f"Partial record mismatch count: {stats['partial_record_mismatch']}/{stats['total']}")
    print()

    for key, example in stats['examples'].items():
        if example is None:
            continue
        idx, nl, gold_q, pred_q, err = example
        print(f'[{key}] example #{idx}')
        print(f'NL: {nl}')
        print(f'Gold SQL: {gold_q}')
        print(f'Pred SQL: {pred_q}')
        if err:
            print(f'Error: {err}')
        print()


if __name__ == '__main__':
    main()
