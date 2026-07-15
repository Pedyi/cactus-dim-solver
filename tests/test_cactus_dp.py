"""
test_cactus_dp.py
=================

Validate the linear-time cactus dynamic program (dim_cactus.cactus_dp), i.e. the
implementation of Algorithm 1, against the brute-force oracle (dim_cactus.exact).

Covered: random cacti, random class-C graphs, every named family of the paper, the
two counterexamples, infeasible members of C, and the weighted case -- checking
both the optimum and the number of optima.

Run:

    python -m pytest tests/ -v
    (or)  python tests/test_cactus_dp.py
"""

import os
import random
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import (
    dim_size,
    count_min_dims,
    is_dim,
    random_cactus,
    random_class_C,
    counterexample_bridged,
    counterexample_glued,
    cycle_bouquet,
    infeasible_member_of_C,
    double_star,
    min_dim_and_count,
    max_weight_dim,
)

MAX_EDGES = 13  # keep the brute-force oracle fast


def _random_cacti(limit=400, max_edges=MAX_EDGES):
    for seed in range(limit):
        G = random_cactus(random.Random(seed).randint(1, 6), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > max_edges:
            continue
        yield seed, G


def test_dp_matches_oracle_on_random_cacti():
    """dim and the number of minimum DIMs agree with brute force."""
    tested = 0
    for seed, G in _random_cacti():
        tested += 1
        oracle_dim = dim_size(G)
        dp_dim, dp_count = min_dim_and_count(G)
        assert oracle_dim == dp_dim, (
            f"seed={seed}: oracle dim={oracle_dim}, dp dim={dp_dim}, "
            f"edges={sorted(G.edges())}"
        )
        if oracle_dim is not None:
            oracle_count = count_min_dims(G)
            assert oracle_count == dp_count, (
                f"seed={seed}: oracle count={oracle_count}, dp count={dp_count}, "
                f"edges={sorted(G.edges())}"
            )
    assert tested > 100, f"only {tested} graphs tested"


def test_dp_matches_oracle_on_class_C():
    """Same check restricted to the structural class C of the paper."""
    tested = 0
    for seed in range(300):
        G, _ = random_class_C(seed)
        if G.number_of_edges() > 14:
            continue
        tested += 1
        oracle_dim = dim_size(G)
        dp_dim, dp_count = min_dim_and_count(G)
        assert oracle_dim == dp_dim, f"seed={seed}"
        if oracle_dim is not None:
            assert count_min_dims(G) == dp_count, f"seed={seed}"
    assert tested > 0


def test_dp_named_families():
    cases = [
        (nx.path_graph(3), 1, 2),
        (nx.path_graph(4), 1, 1),
        (nx.path_graph(5), 2, 1),
        (nx.star_graph(3), 1, 3),
        (nx.star_graph(5), 1, 5),
        (nx.cycle_graph(3), 1, 3),
        (nx.cycle_graph(6), 2, 3),
        (nx.cycle_graph(9), 3, 3),
        (nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)]), 3, 1),
    ]
    for G, want_dim, want_count in cases:
        got_dim, got_count = min_dim_and_count(G)
        assert got_dim == want_dim, f"{sorted(G.edges())}: {got_dim} != {want_dim}"
        assert got_count == want_count, (
            f"{sorted(G.edges())}: count {got_count} != {want_count}"
        )


def test_dp_on_counterexamples():
    """The DP reproduces the paper's counterexample values."""
    Gb, _ = counterexample_bridged()
    assert min_dim_and_count(Gb)[0] == 6      # m/3 = 8, but dim = 6
    Gg, _ = counterexample_glued()
    assert min_dim_and_count(Gg)[0] == 3      # sigma/3 = 5, but dim = 3


def test_dp_detects_infeasibility():
    """Members of C with no DIM are reported as infeasible."""
    assert min_dim_and_count(infeasible_member_of_C(1)) == (None, 0)
    assert min_dim_and_count(infeasible_member_of_C(2)) == (None, 0)
    assert min_dim_and_count(infeasible_member_of_C(3))[0] == 6


def test_dp_bouquets():
    """dim(G_{k,s}) = s*k (Proposition 4.18)."""
    for k in (1, 2):
        for s in (1, 2, 3):
            assert min_dim_and_count(cycle_bouquet(k, s))[0] == s * k


def test_dp_double_stars():
    """The extremal family of the spectral section has dim = 1."""
    for k in (1, 2, 3):
        assert min_dim_and_count(double_star(k, k))[0] == 1


def _brute_max_weight(G, w):
    """Brute-force maximum-weight DIM and the number of optima."""
    E = list(G.edges())
    best = None
    count = 0
    for k in range(len(E) + 1):
        for M in combinations(E, k):
            if is_dim(G, M):
                value = sum(w(u, v) for u, v in M)
                if best is None or value > best:
                    best, count = value, 1
                elif value == best:
                    count += 1
    return best, count


def test_dp_weighted_case():
    """Maximum-Weight DIM and its multiplicity agree with brute force."""
    tested = 0
    for seed, G in _random_cacti(limit=150, max_edges=10):
        rng = random.Random(seed + 9999)
        weights = {frozenset(e): rng.randint(1, 9) for e in G.edges()}

        def w(u, v):
            return weights[frozenset((u, v))]

        tested += 1
        brute_value, brute_count = _brute_max_weight(G, w)
        dp_value, dp_count = max_weight_dim(G, weight=w)
        if brute_value is None:
            assert dp_value == float("-inf"), f"seed={seed}"
        else:
            assert abs(dp_value - brute_value) < 1e-9, (
                f"seed={seed}: dp={dp_value}, brute={brute_value}"
            )
            assert dp_count == brute_count, (
                f"seed={seed}: dp count={dp_count}, brute count={brute_count}"
            )
    assert tested > 0


if __name__ == "__main__":
    test_dp_matches_oracle_on_random_cacti()
    test_dp_matches_oracle_on_class_C()
    test_dp_named_families()
    test_dp_on_counterexamples()
    test_dp_detects_infeasibility()
    test_dp_bouquets()
    test_dp_double_stars()
    test_dp_weighted_case()
    print("All cactus DP tests passed.")
