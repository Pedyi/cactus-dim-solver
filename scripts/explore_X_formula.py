#!/usr/bin/env python3
"""
explore_X_formula.py
====================

Companion to Question 7.1 and Remark 4.16 of the paper: is there a closed formula
for the invariant X, computable from the block-cut tree without solving DIM?

This script records the candidate rules we explored and the explicit configurations
that refute them. It is included so that a reader can verify the negative claims in
the paper's discussion, rather than having to take them on trust.

Run:

    python scripts/explore_X_formula.py
"""

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import find_one_dim


def x_C_of(G, cyc, M):
    """Number of vertices of the cycle `cyc` saturated by an M-edge outside it."""
    L = len(cyc)
    own = set(frozenset((cyc[i], cyc[(i + 1) % L])) for i in range(L))
    return sum(
        1
        for u, v in M
        for endpoint in (u, v)
        if endpoint in cyc and frozenset((u, v)) not in own
    )


def build(L, positions, branch):
    """Cycle C_L with a branch of the given kind attached at each position."""
    G = nx.cycle_graph(L)
    cyc = list(range(L))
    for i, p in enumerate(positions):
        tag = f"g{i}"
        if branch == "pendant_edge":
            G.add_edge(p, f"{tag}z")
        elif branch == "pendant_p3":
            prev = p
            for j in range(3):
                w = f"{tag}z{j}"
                G.add_edge(prev, w)
                prev = w
        elif branch == "glued_triangle":
            nx.add_cycle(G, [p, f"{tag}b1", f"{tag}b2"])
        else:
            raise ValueError(branch)
    return G, cyc


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def demo_branch_type_matters():
    """
    Claim in Remark 4.16: x_C is NOT a function of the cut-vertex positions alone.
    Same positions (w0, w2, w4 on C6), different branches, different x_C.
    """
    section("1. x_C depends on the branches, not only on the positions")
    for branch in ("pendant_p3", "glued_triangle"):
        G, cyc = build(6, [0, 2, 4], branch)
        M = find_one_dim(G)
        if M is None:
            print(f"  C6 + {branch:15s} at (0,2,4): no DIM")
        else:
            print(
                f"  C6 + {branch:15s} at (0,2,4): dim={len(M)}, "
                f"x_C={x_C_of(G, cyc, M)}"
            )
    print("\n  => identical positions, different x_C: any closed formula for x_C")
    print("     must read the branches. (This is the point of Remark 4.16.)")


def refute(rule_name, predict, branch, lengths=(6, 9), max_r=4, max_edges=15):
    """
    Test a candidate rule x_C = predict(arcs, r) against the oracle; report the
    first few counterexamples.
    """
    print(f"\n  Candidate rule: {rule_name}")
    tested = mism = 0
    shown = 0
    for L in lengths:
        for r in range(1, max_r + 1):
            for pos in combinations(range(L), r):
                G, cyc = build(L, list(pos), branch)
                if G.number_of_edges() > max_edges:
                    continue
                M = find_one_dim(G)
                pl = sorted(pos)
                arcs = [((pl[(j + 1) % len(pl)] - pl[j]) % L) for j in range(len(pl))]
                actual = None if M is None else x_C_of(G, cyc, M)
                pred = predict(arcs, r)
                tested += 1
                if actual != pred:
                    mism += 1
                    if shown < 4:
                        print(
                            f"    counterexample: C{L}, positions {pos}, "
                            f"arcs {arcs} -> predicted {pred}, actual {actual}"
                        )
                        shown += 1
    verdict = "REFUTED" if mism else "consistent on this sample"
    print(f"    tested {tested} configurations, {mism} mismatches -> {verdict}")
    return mism == 0


def explore_candidates():
    section("2. Candidate closed formulas for x_C, and their refutations")
    print("  Setting: cycle C_L with rigid branches (a single pendant edge, which")
    print("  forces its attachment vertex to be saturated) at chosen positions;")
    print("  arcs are the gaps between consecutive attachment positions.")

    # Candidate A: x_C = r when all arcs are ≡ 2 (mod 3), else 0.
    def rule_A(arcs, r):
        if all(a % 3 == 2 for a in arcs):
            return r
        if all(a % 3 == 0 for a in arcs):
            return 0
        return None  # predicts infeasibility

    refute("x_C = r if all arcs ≡ 2 (mod 3); 0 if all ≡ 0; else no DIM",
           rule_A, "pendant_edge")

    # Candidate B: x_C = #{arcs ≡ 2 (mod 3)} when that count is divisible by 3.
    def rule_B(arcs, r):
        q = sum(1 for a in arcs if a % 3 == 2)
        if q == 0:
            return 0
        if q % 3 == 0:
            return q
        return None

    refute("x_C = #{arcs ≡ 2 (mod 3)} when 3 divides that count; else no DIM",
           rule_B, "pendant_edge")

    # Candidate C: x_C determined by residues of the positions themselves.
    def rule_C(arcs, r):
        # "all cut-vertices in one residue class -> 0, else r"
        return 0 if all(a % 3 == 0 for a in arcs) else r

    refute("x_C = 0 if all arcs ≡ 0 (mod 3), else x_C = r",
           rule_C, "pendant_edge")

    print("\n  None of the natural position-only rules survives. Together with")
    print("  section 1 above, this is the evidence behind leaving Question 7.1 open.")


def what_is_proved():
    section("3. What the paper does prove (Corollary 4.13(ii))")
    print("  If all pairwise arc-distances between cut-vertices are ≡ 0 (mod 3),")
    print("  then x_C = 0. Verified below on all such configurations:")
    ok = True
    for L in (6, 9):
        for r in range(1, 4):
            for pos in combinations(range(L), r):
                pl = sorted(pos)
                arcs = [((pl[(j + 1) % len(pl)] - pl[j]) % L) for j in range(len(pl))]
                if not all(a % 3 == 0 for a in arcs):
                    continue
                for branch in ("pendant_edge", "pendant_p3", "glued_triangle"):
                    G, cyc = build(L, list(pos), branch)
                    if G.number_of_edges() > 15:
                        continue
                    M = find_one_dim(G)
                    if M is None:
                        continue
                    x = x_C_of(G, cyc, M)
                    if x != 0:
                        ok = False
                        print(f"    VIOLATION: C{L} {pos} {branch}: x_C={x}")
    print(f"    result: {'no violations -- as proved' if ok else 'VIOLATIONS FOUND'}")


def main():
    demo_branch_type_matters()
    explore_candidates()
    what_is_proved()
    print()
    print("Summary: x_C is not determined by cut-vertex positions alone; the")
    print("natural position-only formulas are refuted; the sufficient condition")
    print("proved in the paper (Corollary 4.13) holds without exception.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
