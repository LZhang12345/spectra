from __future__ import annotations

import re

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from spectra_probe import (
    NO_VERBALIZERS,
    YES_VERBALIZERS,
    answer_token_ids,
    generative_classification,
    likelihood_probe,
    single_token_ids,
)
from spectra_probe.prompts import CHOICE_QUESTIONS, PROBE_QUESTIONS, choice_prompt, probe_prompt

CATEGORIES = ("Comedy", "Drama", "Documentary", "Western")
CONTEXT = "Rated movies: Annie Hall 4/5; Hoop Dreams 5/5; Unforgiven 4/5."


class WordTokenizer:
    def __init__(self, texts, extra_tokens):
        words = sorted({w for text in texts for w in self.split(text)} | set(extra_tokens))
        self.vocab = {w: i for i, w in enumerate(["<pad>"] + words)}

    @staticmethod
    def split(text):
        return re.findall(r" ?[A-Za-z0-9'-]+|[^A-Za-z0-9\s]", text)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[w] for w in self.split(text)]


def main():
    torch.manual_seed(0)
    probe_q = PROBE_QUESTIONS["imperative"]["long_term"]
    choice_q = CHOICE_QUESTIONS["imperative"]["long_term"]
    corpus = [probe_prompt(CONTEXT, c, probe_q) for c in CATEGORIES]
    corpus.append(choice_prompt(CONTEXT, CATEGORIES, choice_q))
    answers = [v for word in YES_VERBALIZERS + NO_VERBALIZERS for v in (word, f" {word}")]
    tokenizer = WordTokenizer(corpus, answers + list("ABCD"))
    config = GPT2Config(vocab_size=len(tokenizer.vocab), n_positions=256, n_embd=32, n_layer=2, n_head=2, initializer_range=0.4)
    model = GPT2LMHeadModel(config).eval()

    def next_token_logits(prompts):
        rows = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([tokenizer.encode(prompt)])
                rows.append(model(ids).logits[0, -1].numpy())
        return np.stack(rows)

    yes_ids = single_token_ids(tokenizer, YES_VERBALIZERS)
    no_ids = single_token_ids(tokenizer, NO_VERBALIZERS)
    probe = likelihood_probe(next_token_logits, CONTEXT, CATEGORIES, probe_q, yes_ids, no_ids)
    choice = generative_classification(
        next_token_logits, CONTEXT, CATEGORIES, choice_q, answer_token_ids(tokenizer, CATEGORIES)
    )
    print("Randomly initialized model: the numbers only exercise the interface.")
    for name, readout in (("likelihood probing", probe), ("generative classification", choice)):
        cells = "  ".join(f"{c} {p:.3f}" for c, p in readout.as_dict().items())
        print(f"{name:<28}{cells}")


if __name__ == "__main__":
    main()
