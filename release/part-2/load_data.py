import os, random, re, string
from collections import Counter
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TOKENIZER_NAME = 'google-t5/t5-small'
TASK_PREFIX = 'translate English to SQL: '
DECODER_START_TOKEN = '<extra_id_0>'

_TOKENIZER = None


def get_t5_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
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
        self.tokenizer = get_t5_tokenizer()
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(DECODER_START_TOKEN)
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)

        sql_queries = None
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_path)
            assert len(nl_queries) == len(sql_queries), f'Mismatched sizes for {split}'

        examples = []
        for idx, nl_query in enumerate(nl_queries):
            encoder_text = TASK_PREFIX + nl_query
            encoder_ids = tokenizer.encode(
                encoder_text,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
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
                max_length=512,
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

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
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
