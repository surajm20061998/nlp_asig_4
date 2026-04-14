import json
import os
import argparse

from load_data import (
    DEFAULT_DATA_DIR,
    TASK_PREFIX,
    build_data_config_from_args,
    get_processed_split_text,
    get_t5_tokenizer,
    load_lines,
)


def tokenize_lengths_and_vocab(tokenizer, lines):
    token_lengths = []
    vocab = set()

    for line in lines:
        token_ids = tokenizer.encode(line, add_special_tokens=True)
        token_lengths.append(len(token_ids))
        vocab.update(token_ids)

    mean_length = sum(token_lengths) / len(token_lengths)
    return mean_length, len(vocab)


def compute_split_stats(tokenizer, data_dir, split, data_config=None, preprocess=False):
    if preprocess:
        nl_lines, sql_lines = get_processed_split_text(split, data_dir, data_config)
    else:
        nl_lines = load_lines(os.path.join(data_dir, f'{split}.nl'))
        sql_lines = load_lines(os.path.join(data_dir, f'{split}.sql'))

    mean_nl_len, nl_vocab_size = tokenize_lengths_and_vocab(tokenizer, nl_lines)
    mean_sql_len, sql_vocab_size = tokenize_lengths_and_vocab(tokenizer, sql_lines)

    return {
        'num_examples': len(nl_lines),
        'mean_sentence_length': mean_nl_len,
        'mean_sql_query_length': mean_sql_len,
        'vocab_size_nl': nl_vocab_size,
        'vocab_size_sql': sql_vocab_size,
    }


def build_parser():
    parser = argparse.ArgumentParser(description='Compute train/dev statistics for the T5 Text-to-SQL pipeline')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model_name', type=str, default='google-t5/t5-small')
    parser.add_argument('--task_prefix', type=str, default=TASK_PREFIX)
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--lowercase_inputs', action='store_true')
    parser.add_argument('--include_schema_in_input', action='store_true')
    parser.add_argument('--schema_prompt_mode', choices=['none', 'tables', 'full'])
    parser.add_argument('--canonicalize_sql', action='store_true')
    parser.add_argument('--no_normalize_whitespace', dest='normalize_whitespace', action='store_false')
    parser.set_defaults(normalize_whitespace=True)
    return parser


def format_stat_value(value):
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def print_markdown_table(title, stats, include_num_examples=True):
    print(title)
    print('| Statistics Name | Train | Dev |')
    print('| --- | ---: | ---: |')
    keys = ['mean_sentence_length', 'mean_sql_query_length', 'vocab_size_nl', 'vocab_size_sql']
    if include_num_examples:
        keys = ['num_examples'] + keys

    for key in keys:
        if key not in stats['train']:
            continue
        print(
            f"| {key.replace('_', ' ').title()} | "
            f"{format_stat_value(stats['train'][key])} | "
            f"{format_stat_value(stats['dev'][key])} |"
        )
    print()


def main():
    args = build_parser().parse_args()
    tokenizer = get_t5_tokenizer(args.model_name)
    data_dir = args.data_dir
    data_config = build_data_config_from_args(args, data_folder=data_dir)

    before_stats = {
        'train': compute_split_stats(tokenizer, data_dir, 'train', preprocess=False),
        'dev': compute_split_stats(tokenizer, data_dir, 'dev', preprocess=False),
    }
    after_stats = {
        'model_name': tokenizer.name_or_path,
        'preprocessing': {
            'task_prefix': args.task_prefix,
            'normalize_whitespace': args.normalize_whitespace,
            'lowercase_inputs': args.lowercase_inputs,
            'include_schema_in_input': args.include_schema_in_input,
            'schema_prompt_mode': args.schema_prompt_mode or ('full' if args.include_schema_in_input else 'none'),
            'canonicalize_sql': args.canonicalize_sql,
            'max_input_length': args.max_input_length,
            'max_target_length': args.max_target_length,
        },
        'train': compute_split_stats(tokenizer, data_dir, 'train', data_config=data_config, preprocess=True),
        'dev': compute_split_stats(tokenizer, data_dir, 'dev', data_config=data_config, preprocess=True),
    }

    print('Before preprocessing:')
    print(json.dumps(before_stats, indent=2))
    print()
    print('After preprocessing:')
    print(json.dumps(after_stats, indent=2))
    print()
    print_markdown_table('Table 1 (before preprocessing)', before_stats, include_num_examples=True)
    print(f"Model name: {after_stats['model_name']}")
    print_markdown_table('Table 2 (after preprocessing)', after_stats, include_num_examples=False)


if __name__ == '__main__':
    main()
