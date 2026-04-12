import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "e": "wsdfr",
    "i": "ujko",
    "o": "iklp",
    "u": "yhji",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "g": "tyfhvb",
    "h": "yugjbn",
    "k": "ijolm",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "p": "ol",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "y": "tghu",
}
PROTECTED_TOKENS = {"not", "no", "never", "n't"}


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    tokens = word_tokenize(example["text"])
    transformed_tokens = []

    for token in tokens:
        if not token.isalpha():
            transformed_tokens.append(token)
            continue

        lower_token = token.lower()
        new_token = lower_token

        if lower_token not in PROTECTED_TOKENS and len(lower_token) >= 4 and random.random() < 0.12:
            synonyms = set()
            for synset in wordnet.synsets(lower_token):
                for lemma in synset.lemmas():
                    candidate = lemma.name().replace("_", " ").lower()
                    if (
                        candidate.isalpha()
                        and candidate != lower_token
                        and len(candidate) >= 3
                    ):
                        synonyms.add(candidate)

            if synonyms:
                new_token = random.choice(sorted(synonyms))

        if lower_token not in PROTECTED_TOKENS and len(new_token) >= 4 and random.random() < 0.18:
            typo_positions = [
                idx for idx, char in enumerate(new_token)
                if char in KEYBOARD_NEIGHBORS
            ]
            if typo_positions:
                typo_idx = random.choice(typo_positions)
                replacement_choices = KEYBOARD_NEIGHBORS[new_token[typo_idx]]
                replacement = random.choice(replacement_choices)
                new_token = new_token[:typo_idx] + replacement + new_token[typo_idx + 1:]

        transformed_tokens.append(new_token)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
