import numpy as np
import pytest

from spectra_probe import (
    NO_VERBALIZERS,
    YES_VERBALIZERS,
    answer_token_ids,
    generative_classification,
    hierarchical_probe,
    likelihood_probe,
    single_token_ids,
    softmax,
    yes_no_score,
)
from spectra_probe.probing import Readout
from spectra_probe.prompts import CHOICE_QUESTIONS, PROBE_QUESTIONS
from spectra_probe.toy import ToyLM

CATEGORIES = ("Drama", "Comedy", "Film-Noir", "Western")
SCORES = np.array([1.5, 0.4, -0.3, -1.1])
QUESTION = PROBE_QUESTIONS["imperative"]["long_term"]


def toy_lm():
    return ToyLM(CATEGORIES, SCORES, SCORES[::-1])


def test_softmax_is_a_distribution_and_respects_temperature():
    cold = softmax(SCORES, temperature=0.5)
    warm = softmax(SCORES, temperature=2.0)
    assert np.isclose(cold.sum(), 1.0) and np.isclose(warm.sum(), 1.0)
    assert cold.max() > warm.max()
    with pytest.raises(ValueError):
        softmax(SCORES, temperature=0.0)


def test_yes_no_score_is_mean_affirmative_minus_mean_negative():
    logits = np.array([2.0, 4.0, -1.0, -3.0, 9.0])
    assert yes_no_score(logits, [0, 1], [2, 3]) == pytest.approx(3.0 - (-2.0))


def test_likelihood_probe_recovers_the_scores():
    lm = toy_lm()
    yes_ids = single_token_ids(lm.tokenizer, YES_VERBALIZERS)
    no_ids = single_token_ids(lm.tokenizer, NO_VERBALIZERS)
    readout = likelihood_probe(lm, "history", CATEGORIES, QUESTION, yes_ids, no_ids)
    assert np.allclose(readout.scores, SCORES)
    assert np.allclose(readout.distribution, softmax(SCORES))
    assert readout.top(2) == ["Drama", "Comedy"]


def test_generative_classification_reads_answer_tokens():
    lm = toy_lm()
    ids = answer_token_ids(lm.tokenizer, CATEGORIES)
    readout = generative_classification(
        lm, "history", CATEGORIES, CHOICE_QUESTIONS["imperative"]["long_term"], ids
    )
    assert np.allclose(readout.scores, SCORES[::-1])
    assert readout.top(1) == ["Western"]


def test_single_token_ids_rejects_multi_token_verbalizers():
    lm = toy_lm()
    with pytest.raises(ValueError):
        single_token_ids(lm.tokenizer, ("Absolutely",))


def test_hierarchical_probe_factorizes_and_keeps_unexpanded_mass():
    taxonomy = {"Food": ["Pizza", "Sushi"], "Shopping": ["Books", "Shoes"], "Health": ["Dentist"]}
    table = {
        ("Food", "Shopping", "Health"): np.array([1.0, 0.0, -1.0]),
        ("Pizza", "Sushi"): np.array([0.0, 1.0]),
        ("Books", "Shoes"): np.array([2.0, 0.0]),
        ("Dentist",): np.array([0.0]),
    }

    def probe(categories):
        scores = table[tuple(categories)]
        return Readout(tuple(categories), scores, softmax(scores))

    full = hierarchical_probe(probe, taxonomy)
    assert sum(full.distribution.values()) == pytest.approx(1.0)
    coarse = softmax(table[("Food", "Shopping", "Health")])
    fine = softmax(table[("Pizza", "Sushi")])
    assert full.distribution["Sushi"] == pytest.approx(coarse[0] * fine[1])

    partial = hierarchical_probe(probe, taxonomy, top_branches=1)
    assert set(partial.fine) == {"Food"}
    assert sum(partial.distribution.values()) + sum(partial.unexpanded.values()) == pytest.approx(1.0)
