from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import js_divergence_bits, ranking_to_vec, to_simplex
from .probing import (
    NO_VERBALIZERS,
    YES_VERBALIZERS,
    Readout,
    answer_token_ids,
    generative_classification,
    likelihood_probe,
    single_token_ids,
)
from .prompts import CHOICE_QUESTIONS, CHOICE_SUFFIX, PROBE_QUESTIONS, answer_letters
from .taxonomies import MOVIELENS_GENRES

POPULARITY_ORDER = (
    "Drama",
    "Comedy",
    "Action",
    "Thriller",
    "Adventure",
    "Romance",
    "Sci-Fi",
    "Crime",
    "Fantasy",
    "Children",
    "Mystery",
    "Horror",
    "Animation",
    "War",
    "IMAX",
    "Musical",
    "Documentary",
    "Western",
    "Film-Noir",
)

USER_WEIGHTS = {
    "Drama": 20.0,
    "Comedy": 12.0,
    "Action": 5.0,
    "Thriller": 6.0,
    "Adventure": 3.0,
    "Romance": 4.0,
    "Sci-Fi": 3.0,
    "Crime": 7.0,
    "Fantasy": 1.5,
    "Children": 0.8,
    "Mystery": 6.0,
    "Horror": 1.0,
    "Animation": 1.2,
    "War": 4.0,
    "IMAX": 0.5,
    "Musical": 1.5,
    "Documentary": 11.0,
    "Western": 3.5,
    "Film-Noir": 9.0,
}


class ToyTokenizer:
    def __init__(self, categories: Sequence[str]) -> None:
        tokens = []
        for word in YES_VERBALIZERS + NO_VERBALIZERS:
            tokens += [word, f" {word}"]
        tokens += answer_letters(categories)
        self.vocab = {token: i for i, token in enumerate(dict.fromkeys(tokens))}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]] if text in self.vocab else [-1, -1]


class ToyLM:
    def __init__(self, categories: Sequence[str], scores: Sequence[float], choice_scores: Sequence[float]) -> None:
        self.categories = tuple(categories)
        self.tokenizer = ToyTokenizer(categories)
        self.scores = dict(zip(self.categories, map(float, scores)))
        self.choice_scores = np.asarray(choice_scores, dtype=np.float64)
        self._yes = single_token_ids(self.tokenizer, YES_VERBALIZERS)
        self._no = single_token_ids(self.tokenizer, NO_VERBALIZERS)
        self._letters = answer_token_ids(self.tokenizer, self.categories)

    def _category_in(self, prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[-1]
        matches = [c for c in self.categories if c in question]
        return max(matches, key=len)

    def __call__(self, prompts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(prompts), len(self.tokenizer.vocab)), dtype=np.float64)
        for row, prompt in zip(out, prompts):
            if prompt.rstrip().endswith(CHOICE_SUFFIX):
                row[self._letters] = self.choice_scores
            else:
                score = self.scores[self._category_in(prompt)]
                row[self._yes] = 0.5 * score
                row[self._no] = -0.5 * score
        return out


@dataclass(frozen=True)
class ToyExample:
    categories: tuple[str, ...]
    truth: np.ndarray
    popularity: np.ndarray
    likelihood: Readout
    generative: Readout
    decoding_scores: np.ndarray

    def decoded_list(self, k: int) -> list[str]:
        order = np.lexsort((np.arange(len(self.decoding_scores)), -self.decoding_scores))[:k]
        return [self.categories[i] for i in order]

    def direct_generation(self, k: int) -> np.ndarray:
        return ranking_to_vec(self.decoded_list(k), self.categories)

    def js(self, prediction: Sequence[float]) -> float:
        return js_divergence_bits(prediction, self.truth)


def toy_example(noise: float = 0.25, frequency_bias: float = 1.0, seed: int = 7) -> ToyExample:
    categories = MOVIELENS_GENRES
    truth = to_simplex([USER_WEIGHTS[c] for c in categories])
    rank = {c: i for i, c in enumerate(POPULARITY_ORDER)}
    popularity = to_simplex([1.0 / (rank[c] + 1.0) ** 0.9 for c in categories])
    rng = np.random.default_rng(seed)
    probe_scores = np.log(truth) + noise * rng.standard_normal(len(categories))
    choice_scores = np.log(truth) + noise * rng.standard_normal(len(categories))
    lm = ToyLM(categories, probe_scores, choice_scores)
    context = "User history: a synthetic viewer with head and long-tail interests."
    yes_ids = single_token_ids(lm.tokenizer, YES_VERBALIZERS)
    no_ids = single_token_ids(lm.tokenizer, NO_VERBALIZERS)
    likelihood = likelihood_probe(
        lm, context, categories, PROBE_QUESTIONS["imperative"]["long_term"], yes_ids, no_ids
    )
    generative = generative_classification(
        lm,
        context,
        categories,
        CHOICE_QUESTIONS["imperative"]["long_term"],
        answer_token_ids(lm.tokenizer, categories),
    )
    decoding_scores = np.log(truth) + frequency_bias * np.log(popularity)
    return ToyExample(tuple(categories), truth, popularity, likelihood, generative, decoding_scores)
