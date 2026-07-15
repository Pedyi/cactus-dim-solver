#!/usr/bin/env python3
"""
reproduce_claims.py
===================

Reproduce, from scratch, every combinatorial claim in the paper

    "Extremal and Spectral Results on Dominating Induced Matchings in
     Cactus and Unicyclic Graphs".

Run:

    python scripts/reproduce_claims.py

Each check prints PASS/FAIL. All checks use the brute-force oracle
(dim_cactus.exact) as ground truth.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import (
    dim_size,
    count_min_dims,
    find_one_dim,
    random_cactus,
    random_class_C,
    cycle_bouquet,
    counterexample_bridged,
    counterexample_glued,
    infeasible_member_of_C,
    compute_X,
    saturation_rhs,
    upper_bound_m_plus_l_over_3,
    X_via_dim,
    quadratic_lower_bound,
    cactus_spectral_lower_bound,
    spectral_radius,
    min_dim_and_count,
)

PASS, FAIL = "PASS", "FAIL"


def check(name, ok):
    print(f"[{PASS if ok else FAIL}] {name}")
    return ok


def main():
    ok_all = True

    # ---- Proposition 4.1: dim is not a function of (n, l, c) --------------- #
    # C6 with pendants at w0,w1,w3 -> dim 2 ; at w0,w2,w4 -> dim 3.
    G1 = nx.cycle_graph(6)
    for w in (0, 1, 3):
        G1.add_edge(w, f"p{w}")
    G2 = nx.cycle_graph(6)
    for w in (0, 2, 4):
        G2.add_edge(w, f"p{w}")
    ok = (dim_size(G1), dim_size(G2)) == (2, 3)
    ok_all &= check("Prop 4.1  dim not a function of (n,l,c)  [dim=2 vs 3]", ok)

    # ---- Theorem 4.4: saturation identity 3|M| = m - sum (d(v)-2) ---------- #
    viol = 0
    for seed in range(300):
        G = random_cactus(random.Random(seed).randint(1, 5), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > 12:
            continue
        M = find_one_dim(G)
        if M is None:
            continue
        if 3 * len(M) != saturation_rhs(G, M):
            viol += 1
    ok_all &= check(f"Thm 4.4   saturation identity  [{viol} violations]", viol == 0)

    # ---- Theorem 4.4: sharp upper bound dim <= (m + l)/3 ------------------- #
    viol = 0
    for seed in range(300):
        G = random_cactus(random.Random(seed).randint(1, 5), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > 12:
            continue
        d = dim_size(G)
        if d is None:
            continue
        if d > upper_bound_m_plus_l_over_3(G) + 1e-9:
            viol += 1
    ok_all &= check(f"Thm 4.4   upper bound (m+l)/3  [{viol} violations]", viol == 0)

    # ---- Theorem 4.9 & Cor 4.10: dim = m/3 - (2/3)X, X = (m-3dim)/2 -------- #
    viol = 0
    tested = 0
    for seed in range(400):
        G, cycles = random_class_C(seed)
        if G.number_of_edges() > 14:
            continue
        M = find_one_dim(G)
        if M is None:
            continue
        tested += 1
        d = len(M)
        X_formula = X_via_dim(G)
        X_struct = compute_X(G, cycles, M)
        m = G.number_of_edges()
        if d != (m / 3.0 - (2.0 / 3.0) * X_formula):
            viol += 1
        elif X_formula != X_struct:
            viol += 1
    ok_all &= check(
        f"Thm 4.9   dim = m/3 - (2/3)X and X=(m-3dim)/2  "
        f"[{tested} class-C graphs, {viol} violations]",
        viol == 0,
    )

    # ---- Examples 4.21 / 4.22: two counterexamples with X=3 ---------------- #
    Gb, cb = counterexample_bridged()
    okb = (Gb.number_of_nodes(), Gb.number_of_edges(), dim_size(Gb),
           X_via_dim(Gb)) == (21, 24, 6, 3)
    ok_all &= check("Ex 4.21   bridged: n=21,m=24,dim=6,X=3", okb)

    Gg, cg = counterexample_glued()
    okg = (Gg.number_of_nodes(), Gg.number_of_edges(), dim_size(Gg),
           X_via_dim(Gg)) == (12, 15, 3, 3)
    ok_all &= check("Ex 4.22   glued:   n=12,m=15,dim=3,X=3", okg)

    # ---- Example 4.23: infeasible members of C ----------------------------- #
    inf1 = dim_size(infeasible_member_of_C(1)) is None
    inf2 = dim_size(infeasible_member_of_C(2)) is None
    feas3 = dim_size(infeasible_member_of_C(3)) == 6
    ok_all &= check("Ex 4.23   infeasible at dist 1,2; dim=6 at dist 3",
                    inf1 and inf2 and feas3)

    # ---- Proposition 4.18: cycle bouquets dim(G_{k,s}) = s k ---------------- #
    viol = 0
    for k in (1, 2):
        for s in (1, 2, 3):
            G = cycle_bouquet(k, s)
            if dim_size(G) != s * k:
                viol += 1
    ok_all &= check(f"Prop 4.18 bouquets dim = s*k  [{viol} violations]", viol == 0)

    # ---- Theorem 5.6: quadratic lower bound dim >= ceil(m/(2 l1^2 - 1)) ----- #
    viol = 0
    for seed in range(300):
        G = random_cactus(random.Random(seed).randint(1, 5), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > 12:
            continue
        d = dim_size(G)
        if d is None:
            continue
        if d < quadratic_lower_bound(G):
            viol += 1
    ok_all &= check(f"Thm 5.6   quadratic lower bound  [{viol} violations]",
                    viol == 0)

    # ---- Corollary 5.7: cactus-specific spectral bound --------------------- #
    viol = 0
    for seed in range(300):
        G = random_cactus(random.Random(seed).randint(1, 5), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > 12:
            continue
        d = dim_size(G)
        if d is None:
            continue
        if d < cactus_spectral_lower_bound(G):
            viol += 1
    ok_all &= check(f"Cor 5.7   cactus spectral bound (m=n-1+c)  [{viol} violations]",
                    viol == 0)

    # ---- Theorem 6.3: Algorithm 1 agrees with the exact oracle -------------- #
    mismatches = 0
    tested = 0
    for seed in range(400):
        G = random_cactus(random.Random(seed).randint(1, 6), seed)
        if not nx.is_connected(G) or G.number_of_edges() == 0:
            continue
        if G.number_of_edges() > 13:
            continue
        tested += 1
        oracle = dim_size(G)
        dp, dp_count = min_dim_and_count(G)
        if oracle != dp:
            mismatches += 1
        elif oracle is not None and count_min_dims(G) != dp_count:
            mismatches += 1
    ok_all &= check(
        f"Thm 6.3   Algorithm 1 = oracle (dim and count)  "
        f"[{tested} cacti, {mismatches} mismatches]",
        mismatches == 0,
    )

    print()
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
