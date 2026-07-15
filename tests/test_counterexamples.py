"""
test_counterexamples.py
=======================

Pin down the exact numbers reported in the paper for the two counterexamples
(Examples 4.21 and 4.22), the infeasible members of the class C (Example 4.23),
the bouquet formula (Proposition 4.18), and Proposition 4.1.

Run:

    python -m pytest tests/ -v
    (or)  python tests/test_counterexamples.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import (
    dim_size,
    find_one_dim,
    compute_X,
    X_via_dim,
    counterexample_bridged,
    counterexample_glued,
    infeasible_member_of_C,
    cycle_bouquet,
)


def test_example_4_21_bridged():
    """Example 4.21: n=21, m=24, m/3=8, but dim=6 and X=3."""
    G, cycles = counterexample_bridged()
    assert G.number_of_nodes() == 21
    assert G.number_of_edges() == 24
    assert G.number_of_edges() / 3 == 8
    assert dim_size(G) == 6
    assert X_via_dim(G) == 3
    M = find_one_dim(G)
    assert compute_X(G, cycles, M) == 3


def test_example_4_22_glued():
    """Example 4.22: n=12, m=sigma=15, sigma/3=5, but dim=3 and X=3."""
    G, cycles = counterexample_glued()
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 15
    assert G.number_of_edges() / 3 == 5
    assert dim_size(G) == 3
    assert X_via_dim(G) == 3
    M = find_one_dim(G)
    assert compute_X(G, cycles, M) == 3


def test_refined_identity_on_counterexamples():
    """dim = m/3 - (2/3)X holds on both counterexamples."""
    for builder in (counterexample_bridged, counterexample_glued):
        G, _ = builder()
        m = G.number_of_edges()
        d = dim_size(G)
        X = X_via_dim(G)
        assert d == m / 3 - (2 / 3) * X


def test_example_4_23_infeasible_members():
    """Membership in C does not guarantee a DIM: distances 1 and 2 are infeasible."""
    assert dim_size(infeasible_member_of_C(1)) is None
    assert dim_size(infeasible_member_of_C(2)) is None
    G3 = infeasible_member_of_C(3)
    assert G3.number_of_edges() == 18
    assert dim_size(G3) == 6  # = m/3


def test_proposition_4_1_no_nlc_formula():
    """C6 with pendants at w0,w1,w3 has dim 2; at w0,w2,w4 has dim 3."""
    G1 = nx.cycle_graph(6)
    for w in (0, 1, 3):
        G1.add_edge(w, f"p{w}")
    G2 = nx.cycle_graph(6)
    for w in (0, 2, 4):
        G2.add_edge(w, f"p{w}")
    # same (n, l, c)
    assert G1.number_of_nodes() == G2.number_of_nodes() == 9
    assert sum(1 for _, d in G1.degree() if d == 1) == 3
    assert sum(1 for _, d in G2.degree() if d == 1) == 3
    # different dim
    assert dim_size(G1) == 2
    assert dim_size(G2) == 3


def test_proposition_4_18_bouquets():
    """dim(G_{k,s}) = s*k for cycle bouquets."""
    for k in (1, 2):
        for s in (1, 2, 3):
            assert dim_size(cycle_bouquet(k, s)) == s * k


if __name__ == "__main__":
    test_example_4_21_bridged()
    test_example_4_22_glued()
    test_refined_identity_on_counterexamples()
    test_example_4_23_infeasible_members()
    test_proposition_4_1_no_nlc_formula()
    test_proposition_4_18_bouquets()
    print("All counterexample tests passed.")
