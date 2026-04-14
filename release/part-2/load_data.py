import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PAD_IDX = 0
TOKENIZER_NAME = 'google-t5/t5-small'
TASK_PREFIX = 'translate English to SQL: '
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')

_TOKENIZER = None


@dataclass(frozen=True)
class DataConfig:
    data_folder: str = DEFAULT_DATA_DIR
    tokenizer_name: str = TOKENIZER_NAME
    task_prefix: str = TASK_PREFIX
    max_input_length: int = 512
    max_target_length: int = 512
    normalize_whitespace: bool = True
    lowercase_inputs: bool = False
    include_schema_in_input: bool = False
    schema_prompt_mode: str = 'none'
    canonicalize_sql: bool = False


def get_t5_tokenizer(tokenizer_name=TOKENIZER_NAME):
    global _TOKENIZER
    if _TOKENIZER is None or _TOKENIZER.name_or_path != tokenizer_name:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return _TOKENIZER


def build_data_config_from_args(args, data_folder=DEFAULT_DATA_DIR):
    include_schema = getattr(args, 'include_schema_in_input', False)
    schema_prompt_mode = getattr(args, 'schema_prompt_mode', None)
    if schema_prompt_mode is None:
        schema_prompt_mode = 'full' if include_schema else 'none'

    return DataConfig(
        data_folder=data_folder,
        tokenizer_name=getattr(args, 'model_name', TOKENIZER_NAME),
        task_prefix=getattr(args, 'task_prefix', TASK_PREFIX),
        max_input_length=getattr(args, 'max_input_length', 512),
        max_target_length=getattr(args, 'max_target_length', 512),
        normalize_whitespace=getattr(args, 'normalize_whitespace', True),
        lowercase_inputs=getattr(args, 'lowercase_inputs', False),
        include_schema_in_input=include_schema,
        schema_prompt_mode=schema_prompt_mode,
        canonicalize_sql=getattr(args, 'canonicalize_sql', False),
    )


def normalize_text(text):
    return ' '.join(text.strip().split())


def _normalize_sql_nonquoted_segment(segment):
    segment = re.sub(r'([(),])', r' \1 ', segment)
    segment = re.sub(r'(<=|>=|!=|<>|=|<|>)', r' \1 ', segment)
    segment = re.sub(r'\s+', ' ', segment)
    return segment.strip()


def canonicalize_sql_query(query):
    parts = re.split(r"('(?:''|[^'])*')", query.strip())
    normalized_parts = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            normalized_parts.append(part)
        else:
            normalized_parts.append(_normalize_sql_nonquoted_segment(part))

    canonical_query = ' '.join(part for part in normalized_parts if part)
    return normalize_text(canonical_query)


@lru_cache(maxsize=16)
def build_schema_context(schema_path, mode):
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    if mode == 'tables':
        table_descriptions = list(schema['ents'].keys())
        return 'database tables: ' + ' , '.join(table_descriptions)

    table_descriptions = []
    for table_name, columns in schema['ents'].items():
        if mode == 'full':
            col_names = ', '.join(columns.keys())
            table_descriptions.append(f'{table_name} ({col_names})')
        else:
            table_descriptions.append(table_name)

    return 'database schema: ' + ' ; '.join(table_descriptions)


def preprocess_nl_query(query, data_config):
    processed_query = normalize_text(query) if data_config.normalize_whitespace else query.strip()
    if data_config.lowercase_inputs:
        processed_query = processed_query.lower()

    prompt_parts = []
    if data_config.task_prefix:
        prompt_parts.append(data_config.task_prefix.strip())
    if data_config.schema_prompt_mode != 'none':
        schema_path = os.path.join(data_config.data_folder, 'flight_database.schema')
        prompt_parts.append(build_schema_context(schema_path, data_config.schema_prompt_mode))
    prompt_parts.append(processed_query)

    return ' '.join(part for part in prompt_parts if part)


def preprocess_sql_query(query, data_config):
    processed_query = query.strip()
    if data_config.canonicalize_sql:
        return canonicalize_sql_query(processed_query)
    if data_config.normalize_whitespace:
        return normalize_text(processed_query)
    return processed_query


def get_processed_split_text(split, data_folder=DEFAULT_DATA_DIR, data_config=None):
    data_config = data_config or DataConfig(data_folder=data_folder)
    nl_queries = load_lines(os.path.join(data_folder, f'{split}.nl'))
    sql_queries = None if split == 'test' else load_lines(os.path.join(data_folder, f'{split}.sql'))

    processed_nl = [preprocess_nl_query(query, data_config) for query in nl_queries]
    processed_sql = None
    if sql_queries is not None:
        processed_sql = [preprocess_sql_query(query, data_config) for query in sql_queries]

    return processed_nl, processed_sql

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, data_config=None):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.is_test = split == 'test'
        self.data_config = data_config or DataConfig(data_folder=data_folder)
        self.tokenizer = get_t5_tokenizer(self.data_config.tokenizer_name)
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.examples = self.process_data(data_folder, split, self.tokenizer, self.data_config)

    def process_data(self, data_folder, split, tokenizer, data_config):
        nl_queries, sql_queries = get_processed_split_text(split, data_folder, data_config)
        if sql_queries is not None:
            assert len(nl_queries) == len(sql_queries), f'Mismatched sizes for {split}'

        examples = []
        for idx, nl_query in enumerate(nl_queries):
            encoder_ids = tokenizer.encode(
                nl_query,
                add_special_tokens=True,
                truncation=True,
                max_length=data_config.max_input_length,
            )
            encoder_ids = torch.tensor(encoder_ids, dtype=torch.long)
            initial_decoder_inputs = torch.tensor([self.decoder_start_token_id], dtype=torch.long)

            if split == 'test':
                examples.append((encoder_ids, initial_decoder_inputs))
                continue

            target_ids = tokenizer.encode(
                sql_queries[idx],
                add_special_tokens=True,
                truncation=True,
                max_length=data_config.max_target_length,
            )
            decoder_input_ids = [self.decoder_start_token_id] + target_ids[:-1]

            examples.append(
                (
                    encoder_ids,
                    torch.tensor(decoder_input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                    initial_decoder_inputs,
                )
            )

        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack([item[3] for item in batch], dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack([item[1] for item in batch], dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, data_folder=DEFAULT_DATA_DIR, data_config=None):
    dset = T5Dataset(data_folder, split, data_config=data_config)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, data_config=None, data_folder=DEFAULT_DATA_DIR):
    data_config = data_config or DataConfig(data_folder=data_folder)
    train_loader = get_dataloader(batch_size, "train", data_folder=data_folder, data_config=data_config)
    dev_loader = get_dataloader(test_batch_size, "dev", data_folder=data_folder, data_config=data_config)
    test_loader = get_dataloader(test_batch_size, "test", data_folder=data_folder, data_config=data_config)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
