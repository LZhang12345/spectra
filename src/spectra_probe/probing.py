from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .prompts import answer_letters, choice_prompt, probe_prompt

YES_VERBALIZERS = ("Yes", "Y", "y")
NO_VERBALIZERS = ("No", "N", "n")

NextTokenLogits = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class Readout:
    categories: tuple[str, ...]
    scores: np.ndarray
    distribution: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {c: float(p) for c, p in zip(self.categories, self.distribution)}

    def top(self, k: int) -> list[str]:
        order = np.lexsort((np.arange(len(self.scores)), -self.distribution))[:k]
        return [self.categories[i] for i in order]


@dataclass(frozen=True)
class HierarchicalReadout:
    coarse: Readout
    fine: dict[str, Readout]
    distribution: dict[str, float]
    unexpanded: dict[str, float]


def softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    arr = np.asarray(scores, dtype=np.float64) / temperature
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / exp.sum()


def single_token_ids(tokenizer, verbalizers: Sequence[str], leading_space: bool = True) -> list[int]:
    variants = list(verbalizers)
    if leading_space:
        variants += [f" {v}" for v in verbalizers]
    ids: list[int] = []
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(int(encoded[0]))
    if not ids:
        raise ValueError(f"no single-token verbalizer found among {variants}")
    return ids


def yes_no_score(logits: np.ndarray, yes_ids: Sequence[int], no_ids: Sequence[int]) -> float:
    row = np.asarray(logits, dtype=np.float64)
    return float(row[list(yes_ids)].mean() - row[list(no_ids)].mean())


def likelihood_probe(
    next_token_logits: NextTokenLogits,
    context: str,
    categories: Sequence[str],
    question: str,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    temperature: float = 1.0,
) -> Readout:
    prompts = [probe_prompt(context, category, question) for category in categories]
    logits = np.asarray(next_token_logits(prompts), dtype=np.float64)
    scores = np.array([yes_no_score(row, yes_ids, no_ids) for row in logits])
    return Readout(tuple(categories), scores, softmax(scores, temperature))


def generative_classification(
    next_token_logits: NextTokenLogits,
    context: str,
    categories: Sequence[str],
    question: str,
    answer_ids: Sequence[int],
    temperature: float = 1.0,
) -> Readout:
    if len(answer_ids) != len(categories):
        raise ValueError("one answer token is required per category")
    prompt = choice_prompt(context, categories, question)
    logits = np.asarray(next_token_logits([prompt]), dtype=np.float64)[0]
    scores = logits[list(answer_ids)]
    return Readout(tuple(categories), scores, softmax(scores, temperature))


def answer_token_ids(tokenizer, categories: Sequence[str], leading_space: bool = False) -> list[int]:
    ids = []
    for letter in answer_letters(categories):
        text = f" {letter}" if leading_space else letter
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"answer token {text!r} is not a single token")
        ids.append(int(encoded[0]))
    return ids


def hierarchical_probe(
    probe: Callable[[Sequence[str]], Readout],
    taxonomy: Mapping[str, Sequence[str]],
    branches: Sequence[str] | None = None,
    top_branches: int | None = None,
) -> HierarchicalReadout:
    coarse = probe(list(taxonomy))
    if branches is None:
        branches = coarse.top(top_branches) if top_branches is not None else list(taxonomy)
    coarse_mass = coarse.as_dict()
    fine: dict[str, Readout] = {}
    distribution: dict[str, float] = {}
    for branch in branches:
        children = list(taxonomy[branch])
        if not children:
            continue
        fine[branch] = probe(children)
        for child, conditional in fine[branch].as_dict().items():
            distribution[child] = coarse_mass[branch] * conditional
    unexpanded = {b: m for b, m in coarse_mass.items() if b not in fine}
    return HierarchicalReadout(coarse, fine, distribution, unexpanded)
