import os
import numpy as np
from transformers import T5TokenizerFast
from collections import Counter

def compute_statistics(split='train'):
    """
    Compute data statistics for Q4 before and after preprocessing.
    """
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    nl_path = os.path.join('data', f'{split}.nl')
    nl_queries = [line.strip() for line in open(nl_path, 'r').readlines()]
    
    if split != 'test':
        sql_path = os.path.join('data', f'{split}.sql')
        sql_queries = [line.strip() for line in open(sql_path, 'r').readlines()]
    else:
        sql_queries = None
    
    print(f"\n{'='*60}")
    print(f"Statistics for {split.upper()} set")
    print(f"{'='*60}\n")
    
    print("BEFORE PREPROCESSING (Raw Text):")
    print("-" * 60)
    print(f"Number of examples: {len(nl_queries)}")
    
    nl_word_lengths = [len(query.split()) for query in nl_queries]
    print(f"Mean sentence length (words): {np.mean(nl_word_lengths):.2f}")
    
    if sql_queries:
        sql_word_lengths = [len(query.split()) for query in sql_queries]
        print(f"Mean SQL query length (words): {np.mean(sql_word_lengths):.2f}")
        
        nl_vocab = set()
        for query in nl_queries:
            nl_vocab.update(query.lower().split())
        print(f"Vocabulary size (natural language): {len(nl_vocab)}")
        
        sql_vocab = set()
        for query in sql_queries:
            sql_vocab.update(query.split())
        print(f"Vocabulary size (SQL): {len(sql_vocab)}")
    
    print("\n" + "="*60)
    print("AFTER PREPROCESSING (T5 Tokenization):")
    print("-" * 60)
    print(f"Model name: google-t5/t5-small")
    
    nl_tokenized = tokenizer(nl_queries, add_special_tokens=True)
    nl_token_lengths = [len(ids) for ids in nl_tokenized['input_ids']]
    print(f"Mean sentence length (tokens): {np.mean(nl_token_lengths):.2f}")
    print(f"Max sentence length (tokens): {max(nl_token_lengths)}")
    print(f"Min sentence length (tokens): {min(nl_token_lengths)}")
    
    if sql_queries:
        sql_tokenized = tokenizer(sql_queries, add_special_tokens=True)
        sql_token_lengths = [len(ids) for ids in sql_tokenized['input_ids']]
        print(f"Mean SQL query length (tokens): {np.mean(sql_token_lengths):.2f}")
        print(f"Max SQL query length (tokens): {max(sql_token_lengths)}")
        print(f"Min SQL query length (tokens): {min(sql_token_lengths)}")
        
        all_nl_tokens = []
        for ids in nl_tokenized['input_ids']:
            all_nl_tokens.extend(ids)
        nl_unique_tokens = len(set(all_nl_tokens))
        print(f"Vocabulary size (natural language tokens): {nl_unique_tokens}")
        
        all_sql_tokens = []
        for ids in sql_tokenized['input_ids']:
            all_sql_tokens.extend(ids)
        sql_unique_tokens = len(set(all_sql_tokens))
        print(f"Vocabulary size (SQL tokens): {sql_unique_tokens}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATA STATISTICS FOR Q4")
    print("="*60)
    
    compute_statistics('train')
    compute_statistics('dev')
    
    print("\nNOTE: These statistics should be used to fill Table 1 and Table 2 in Q4.")
    print("Table 1 uses 'Before Preprocessing' statistics")
    print("Table 2 uses 'After Preprocessing' statistics")
