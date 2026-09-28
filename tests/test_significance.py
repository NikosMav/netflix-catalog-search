"""Bootstrap and sign-test checks. No catalog and no model download."""

from __future__ import annotations

import math

from retrieval.significance import (
    cohens_kappa,
    paired_bootstrap_ci,
    quadratic_weighted_kappa,
    two_sided_sign_test,
)


def test_sign_test_known_binomial_tail():
    # 8 positive, 2 negative. Masses at least as small as C(10, 8) are
    # k in {0, 1, 2, 8, 9, 10}: 1+10+45+45+10+1 = 112, over 1024.
    diffs = [1.0] * 8 + [-1.0] * 2
    result = two_sided_sign_test(diffs)
    assert result["n_nonzero"] == 10
    assert result["n_ties"] == 0
    assert result["p_value"] == 112 / 1024


def test_sign_test_drops_ties_and_symmetric_case():
    result = two_sided_sign_test([1.0, -1.0, 0.0, 0.0])
    assert result["n_ties"] == 2
    assert result["n_nonzero"] == 2
    # Both outcomes are equally likely, so the two-sided p-value is 1.
    assert result["p_value"] == 1.0


def test_sign_test_all_ties_has_no_p_value():
    result = two_sided_sign_test([0.0, 0.0])
    assert result["n_nonzero"] == 0
    assert result["p_value"] is None


def test_bootstrap_constant_difference_is_degenerate():
    system = [0.4, 0.5, 0.6]
    baseline = [0.1, 0.2, 0.3]
    result = paired_bootstrap_ci(system, baseline, n_resamples=200, seed=20260928)
    assert math.isclose(result["mean_difference"], 0.3)
    assert math.isclose(result["ci_low"], result["mean_difference"])
    assert math.isclose(result["ci_high"], result["mean_difference"])
    assert result["n"] == 3
    assert result["n_resamples"] == 200


def test_bootstrap_seed_is_stable_and_interval_brackets_the_mean():
    system = [1.0, 0.0, 0.5, 0.2, 0.8]
    baseline = [0.2, 0.2, 0.2, 0.2, 0.2]
    first = paired_bootstrap_ci(system, baseline, n_resamples=1000, seed=7, level=0.95)
    second = paired_bootstrap_ci(system, baseline, n_resamples=1000, seed=7, level=0.95)
    assert first == second
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]
    other = paired_bootstrap_ci(system, baseline, n_resamples=1000, seed=8, level=0.95)
    assert other["ci_low"] != first["ci_low"] or other["ci_high"] != first["ci_high"]


def test_quadratic_weighted_kappa_perfect_and_one_step():
    perfect = quadratic_weighted_kappa([0, 3], [0, 3])
    assert perfect["agreement"] == 1.0
    assert perfect["kappa"] == 1.0
    # Weights (i-j)^2/9. Observed mass on (0,1) and (3,2).
    # Expected weighted disagreement is 5/9, observed is 2/9, kappa = 1 - 2/5.
    shifted = quadratic_weighted_kappa([0, 3], [1, 2])
    assert shifted["agreement"] == 0.0
    assert abs(shifted["kappa"] - 0.6) < 1e-12


def test_cohens_kappa_perfect_and_chance():
    perfect = cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0])
    assert perfect["agreement"] == 1.0
    assert perfect["kappa"] == 1.0
    # Agreement equals chance when each rater is constant on opposite classes? 
    # Both label half the items 1 but never agree: p_o=0, p_e=0.5, kappa=-1.
    opposed = cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1])
    assert opposed["agreement"] == 0.0
    assert opposed["kappa"] == -1.0
