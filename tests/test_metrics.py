import numpy as np
import pytest

from spectra_probe import (
    bucket_mass,
    entropy_bits,
    global_buckets,
    js_divergence_bits,
    ndcg_at_k,
    ranking_to_vec,
    to_simplex,
    topk_exposure_entropy,
)

VOCAB = ("a", "b", "c", "d")


def test_to_simplex_handles_degenerate_input():
    assert np.allclose(to_simplex([0.0, 0.0]), [0.5, 0.5])
    assert np.all(np.isnan(to_simplex([0.0, 0.0], fallback="nan")))


def test_js_divergence_bounds():
    p = [0.7, 0.2, 0.1, 0.0]
    assert js_divergence_bits(p, p) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence_bits([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(js_divergence_bits([0, 0, 0, 0], p))


def test_ranking_to_vec_gives_zero_mass_to_omitted_categories():
    vec = ranking_to_vec(["b", "a", "b", "zzz"], VOCAB)
    assert np.allclose(vec, [0.5, 0.5, 0.0, 0.0])


def test_ndcg_is_one_for_the_true_ranking():
    truth = [0.5, 0.3, 0.15, 0.05]
    assert ndcg_at_k(truth, truth, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(truth[::-1], truth, k=3) < 1.0


def test_entropy_and_exposure_entropy():
    assert entropy_bits([0.25] * 4) == pytest.approx(2.0)
    same = np.tile([0.7, 0.2, 0.05, 0.05], (5, 1))
    assert topk_exposure_entropy(same, k=1) == pytest.approx(0.0)
    varied = np.eye(4)
    assert topk_exposure_entropy(varied, k=1) == pytest.approx(2.0)


def test_buckets_partition_the_categories():
    truth = np.array([[0.4, 0.3, 0.1, 0.1, 0.06, 0.04]])
    buckets = global_buckets(truth)
    assert sorted(buckets["head"] + buckets["mid"] + buckets["tail"]) == list(range(6))
    assert bucket_mass(truth[0], buckets["head"]) == pytest.approx(0.7)
