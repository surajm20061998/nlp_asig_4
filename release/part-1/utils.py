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
VOWELS = set("aeiou")


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

    def synonym_replace(word):
        synonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower()
                if candidate.isalpha() and candidate != word and len(candidate) >= 3:
                    synonyms.add(candidate)

        if synonyms:
            return random.choice(sorted(synonyms))
        return word

    def keyboard_typo(word):
        typo_positions = [
            idx for idx, char in enumerate(word)
            if char in KEYBOARD_NEIGHBORS
        ]
        if not typo_positions:
            return word

        typo_idx = random.choice(typo_positions)
        replacement = random.choice(KEYBOARD_NEIGHBORS[word[typo_idx]])
        return word[:typo_idx] + replacement + word[typo_idx + 1:]

    def swap_adjacent(word):
        if len(word) < 4:
            return word

        swap_positions = list(range(1, len(word) - 2))
        if not swap_positions:
            return word

        swap_idx = random.choice(swap_positions)
        chars = list(word)
        chars[swap_idx], chars[swap_idx + 1] = chars[swap_idx + 1], chars[swap_idx]
        return "".join(chars)

    def drop_vowel(word):
        removable = [
            idx for idx, char in enumerate(word)
            if char in VOWELS and 0 < idx < len(word) - 1
        ]
        if not removable:
            return word

        drop_idx = random.choice(removable)
        candidate = word[:drop_idx] + word[drop_idx + 1:]
        return candidate if len(candidate) >= 3 else word

    def perturb_word(word):
        operations = []

        if any(char in KEYBOARD_NEIGHBORS for char in word):
            operations.append(keyboard_typo)
        if len(word) >= 4:
            operations.append(swap_adjacent)
        if any(char in VOWELS for char in word):
            operations.append(drop_vowel)

        if not operations:
            return word

        new_word = random.choice(operations)(word)
        if len(new_word) >= 5 and random.random() < 0.45:
            new_word = random.choice(operations)(new_word)
        return new_word

    tokens = word_tokenize(example["text"].lower())
    transformed_tokens = []

    for token in tokens:
        if not token.isalpha():
            transformed_tokens.append(token)
            continue

        if token in PROTECTED_TOKENS:
            transformed_tokens.append(token)
            continue

        new_token = token

        if len(new_token) >= 4 and random.random() < 0.2:
            new_token = synonym_replace(new_token)

        if len(new_token) >= 3 and random.random() < 0.55:
            new_token = perturb_word(new_token)

        transformed_tokens.append(new_token)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
