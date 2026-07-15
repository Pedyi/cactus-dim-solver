"""
test_algorithm.py
=================

Validate the linear-time tree dynamic program (dim_cactus.tree_dp) against the
brute-force oracle (dim_cactus.exact) on random trees and named families, for
both the optimum size and the number of optimal solutions.

Run:

    python -m pytest tests/ -v
    (or)  python tests/test_algorithm.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import dim_size, count_min_dims, tree_min_dim_and_count


def _random_tree(n, seed):
    if n == 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    try:
        return nx.random_labeled_tree(n, seed=seed)
    except AttributeError:  # networkx < 3.2
        return nx.random_tree(n, seed=seed)


def test_tree_dp_matches_oracle_on_random_trees():
    mismatches = []
    for seed in range(300):
        n = random.Random(seed).randint(1, 12)
        T = _random_tree(n, seed)
        oracle_dim = dim_size(T)
        dp_dim, dp_count = tree_min_dim_and_count(T)
        assert oracle_dim == dp_dim, (
            f"seed={seed}: oracle dim={oracle_dim}, dp dim={dp_dim}"
        )
        if oracle_dim is not None:
            oracle_count = count_min_dims(T)
            assert oracle_count == dp_count, (
                f"seed={seed}: oracle count={oracle_count}, dp count={dp_count}"
            )
    assert not mismatches


def test_tree_dp_named_families():
    cases = [
        (nx.path_graph(3), 1, 2),      # P3: dim 1, two minimum DIMs
        (nx.path_graph(4), 1, 1),
        (nx.path_graph(5), 2, 1),
        (nx.star_graph(3), 1, 3),      # K_{1,3}: dim 1, three minimum DIMs
        (nx.star_graph(5), 1, 5),
        (nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)]), 3, 1),  # spider
    ]
    for T, want_dim, want_count in cases:
        got_dim, got_count = tree_min_dim_and_count(T)
        assert got_dim == want_dim, f"{T.edges()}: dim {got_dim} != {want_dim}"
        assert got_count == want_count, (
            f"{T.edges()}: count {got_count} != {want_count}"
        )


def test_tree_dp_weighted():
    """A weighted check: on P4 with a heavy middle edge, the optimum picks it."""
    T = nx.path_graph(4)  # 0-1-2-3
    from dim_cactus import tree_mwdim

    def w(u, v):
        return 10.0 if {u, v} == {1, 2} else 1.0

    weight, count = tree_mwdim(T, weight=w)
    assert weight == 10.0, f"expected 10.0, got {weight}"
    assert count == 1


if __name__ == "__main__":
    test_tree_dp_matches_oracle_on_random_trees()
    test_tree_dp_named_families()
    test_tree_dp_weighted()
    print("All algorithm tests passed.")
