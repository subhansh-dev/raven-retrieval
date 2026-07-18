import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.significance import (
    paired_bootstrap_test, paired_t_test,
    bonferroni_correction, run_all_pairwise_tests
)


def test_bootstrap_known_difference():
    rng = np.random.RandomState(42)
    a = rng.normal(0.7, 0.1, 100)
    b = rng.normal(0.5, 0.1, 100)
    result = paired_bootstrap_test(a, b, n_resamples=5000)
    assert result["observed_diff"] > 0.1
    assert result["p_value"] < 0.01


def test_bootstrap_no_difference():
    rng = np.random.RandomState(42)
    a = rng.normal(0.5, 0.1, 100)
    b = a + rng.normal(0, 0.01, 100)
    result = paired_bootstrap_test(a, b, n_resamples=5000)
    assert abs(result["observed_diff"]) < 0.05
    assert result["p_value"] > 0.05


def test_t_test_agrees_with_bootstrap():
    rng = np.random.RandomState(42)
    a = rng.normal(0.7, 0.1, 100)
    b = rng.normal(0.5, 0.1, 100)
    bootstrap = paired_bootstrap_test(a, b, n_resamples=5000)
    t_test = paired_t_test(a, b)
    assert (bootstrap["p_value"] < 0.05) == (t_test["p_value"] < 0.05)


def test_bonferroni():
    p_values = [0.01, 0.03, 0.04, 0.06, 0.10, 0.50]
    result = bonferroni_correction(p_values, alpha=0.05)
    assert result["adjusted_alpha"] == 0.05 / 6
    assert result["adjusted_alpha"] == 0.05 / 6
    assert result["significant"][0] == False
    assert result["significant"][-1] == False


def test_pairwise_all():
    rng = np.random.RandomState(42)
    per_query = {
        "system_a": rng.normal(0.7, 0.1, 50).tolist(),
        "system_b": rng.normal(0.5, 0.1, 50).tolist(),
        "system_c": rng.normal(0.6, 0.1, 50).tolist(),
    }
    names = ["system_a", "system_b", "system_c"]
    comparisons = run_all_pairwise_tests(per_query, names, n_resamples=1000)
    assert len(comparisons) == 3
    for comp in comparisons:
        assert "bootstrap" in comp
        assert "t_test" in comp
        assert "bonferroni_significant" in comp


if __name__ == "__main__":
    test_bootstrap_known_difference()
    test_bootstrap_no_difference()
    test_t_test_agrees_with_bootstrap()
    test_bonferroni()
    test_pairwise_all()
    print("All significance tests passed.")
