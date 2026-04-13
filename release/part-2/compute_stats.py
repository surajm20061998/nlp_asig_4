import json
import os

from load_data import get_t5_tokenizer, load_lines, TASK_PREFIX


def tokenize_lengths_and_vocab(tokenizer, lines):
    token_lengths = []
    vocab = set()

    for line in lines:
        token_ids = tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=512)
        token_lengths.append(len(token_ids))
        vocab.update(token_ids)

    mean_length = sum(token_lengths) / len(token_lengths)
    return mean_length, len(vocab)


def compute_split_stats(tokenizer, data_dir, split, preprocess_inputs):
    nl_lines = load_lines(os.path.join(data_dir, f'{split}.nl'))
    sql_lines = load_lines(os.path.join(data_dir, f'{split}.sql'))

    if preprocess_inputs:
        nl_lines = [TASK_PREFIX + line for line in nl_lines]

    mean_nl_len, nl_vocab_size = tokenize_lengths_and_vocab(tokenizer, nl_lines)
    mean_sql_len, sql_vocab_size = tokenize_lengths_and_vocab(tokenizer, sql_lines)

    return {
        'num_examples': len(nl_lines),
        'mean_sentence_length': mean_nl_len,
        'mean_sql_query_length': mean_sql_len,
        'vocab_size_nl': nl_vocab_size,
        'vocab_size_sql': sql_vocab_size,
    }


def main():
    tokenizer = get_t5_tokenizer()
    data_dir = 'data'

    before_stats = {
        'train': compute_split_stats(tokenizer, data_dir, 'train', preprocess_inputs=False),
        'dev': compute_split_stats(tokenizer, data_dir, 'dev', preprocess_inputs=False),
    }
    after_stats = {
        'model_name': tokenizer.name_or_path,
        'train': compute_split_stats(tokenizer, data_dir, 'train', preprocess_inputs=True),
        'dev': compute_split_stats(tokenizer, data_dir, 'dev', preprocess_inputs=True),
    }

    print('Before preprocessing:')
    print(json.dumps(before_stats, indent=2))
    print()
    print('After preprocessing:')
    print(json.dumps(after_stats, indent=2))


if __name__ == '__main__':
    main()
