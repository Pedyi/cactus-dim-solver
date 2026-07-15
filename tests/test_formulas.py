"""
test_formulas.py
================

Check the paper's identities and bounds against the exact oracle on random cactus
and class-C graphs.

Run:

    python -m pytest tests/ -v
    (or)  python tests/test_formulas.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import (
    dim_size,
    find_one_dim,
    random_cactus,
    random_class_C,
    compute_X,
    saturation_rhs,
    upper_bound_m_plus_l_over_3,
    X_via_dim,
    quadratic_lower_bound,
    cactus_spectral_lower_bound,
    num_cycle_blocks,
)

MAX_EDGES = 12  # keep the brute-force oracle fast


def _random_cacti(limit=200):
    for seed in range(limit):
        G = random_cactus(random.Random(seed).randint(1, 5), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > MAX_EDGES:
            continue
        yield seed, G


def test_saturation_identity():
    """Theorem 4.4: 3|M| = m - sum_{v in V(M)} (d(v) - 2) for every DIM M."""
    for seed, G in _random_cacti():
        M = find_one_dim(G)
        if M is None:
            continue
        assert 3 * len(M) == saturation_rhs(G, M), f"seed={seed}"


def test_upper_bound_m_plus_l_over_3():
    """Theorem 4.4: dim(G) <= (m + l)/3."""
    for seed, G in _random_cacti():
        d = dim_size(G)
        if d is None:
            continue
        assert d <= upper_bound_m_plus_l_over_3(G) + 1e-9, f"seed={seed}"


def test_size_identity_m_equals_n_minus_1_plus_c():
    """Lemma 5.6: for a connected cactus, m = n - 1 + c."""
    for seed, G in _random_cacti():
        n, m = G.number_of_nodes(), G.number_of_edges()
        c = num_cycle_blocks(G)
        assert m == n - 1 + c, f"seed={seed}: m={m}, n={n}, c={c}"


def test_refined_identity_on_class_C():
    """Theorem 4.9 / Corollary 4.10 on the class C: dim = m/3 - (2/3)X."""
    tested = 0
    for seed in range(300):
        G, cycles = random_class_C(seed)
        if G.number_of_edges() > 14:
            continue
        M = find_one_dim(G)
        if M is None:
            continue
        tested += 1
        m = G.number_of_edges()
        d = len(M)
        X_formula = X_via_dim(G)
        X_struct = compute_X(G, cycles, M)
        assert X_formula == X_struct, f"seed={seed}: {X_formula} != {X_struct}"
        assert X_formula % 3 == 0, f"seed={seed}: X={X_formula} not divisible by 3"
        assert abs(d - (m / 3.0 - (2.0 / 3.0) * X_formula)) < 1e-9, f"seed={seed}"
    assert tested > 0


def test_quadratic_lower_bound():
    """Theorem 5.6: dim(G) >= ceil(m / (2 lambda1^2 - 1))."""
    for seed, G in _random_cacti():
        d = dim_size(G)
        if d is None:
            continue
        assert d >= quadratic_lower_bound(G), f"seed={seed}"


def test_cactus_spectral_lower_bound():
    """Corollary 5.7: dim(G) >= ceil((n - 1 + c) / (2 lambda1^2 - 1))."""
    for seed, G in _random_cacti():
        d = dim_size(G)
        if d is None:
            continue
        assert d >= cactus_spectral_lower_bound(G), f"seed={seed}"


if __name__ == "__main__":
    test_saturation_identity()
    test_upper_bound_m_plus_l_over_3()
    test_size_identity_m_equals_n_minus_1_plus_c()
    test_refined_identity_on_class_C()
    test_quadratic_lower_bound()
    test_cactus_spectral_lower_bound()
    print("All formula tests passed.")
