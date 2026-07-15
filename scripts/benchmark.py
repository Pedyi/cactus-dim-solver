#!/usr/bin/env python3
"""
benchmark.py
============

Demonstrate the linear-time behaviour of Algorithm 1 (dim_cactus.cactus_dp) on
random cactus graphs, supporting the O(n) claim of Theorem 6.3.

The table reports the time per vertex; if the algorithm is linear this figure stays
roughly constant as n grows over two orders of magnitude.

Run:

    python scripts/benchmark.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from dim_cactus import random_cactus, min_dim_and_count, num_cycle_blocks


def cycle_chain(t: int) -> nx.Graph:
    """
    A chain of t triangles joined by 3-edge bridges: a member of the class C on
    which dim = m/3 (Corollary 4.13(i)), so the DP does real work and returns a
    genuine optimum rather than reporting infeasibility.
    """
    G = nx.Graph()
    nx.add_cycle(G, ["c0_0", "c0_1", "c0_2"])
    for i in range(1, t):
        a, b = f"c{i-1}_0", f"c{i}_0"
        G.add_edge(a, f"br{i}_1")
        G.add_edge(f"br{i}_1", f"br{i}_2")
        G.add_edge(f"br{i}_2", b)
        nx.add_cycle(G, [b, f"c{i}_1", f"c{i}_2"])
    return G


def main():
    print("Linear-time behaviour of Algorithm 1")
    print()
    print("Part 1: feasible instances (chains of triangles, dim = m/3)")
    print()
    print(f"{'n':>8} {'m':>8} {'c':>6} {'dim':>7} {'time (s)':>10} {'us/vertex':>10}")
    print("-" * 55)
    feas = []
    for t in (50, 100, 200, 400, 800, 1600, 3200):
        G = cycle_chain(t)
        n, m = G.number_of_nodes(), G.number_of_edges()
        c = num_cycle_blocks(G)
        start = time.perf_counter()
        dim, _ = min_dim_and_count(G)
        elapsed = time.perf_counter() - start
        per_vertex = 1e6 * elapsed / n
        feas.append(per_vertex)
        dim_str = "none" if dim is None else str(dim)
        print(f"{n:8d} {m:8d} {c:6d} {dim_str:>7} {elapsed:10.4f} {per_vertex:10.2f}")
    if feas:
        print("-" * 55)
        print(f"time per vertex varies by a factor of {max(feas)/min(feas):.2f}")
    print()
    print("Part 2: general random cacti (most large random cacti admit no DIM;")
    print("the DP detects this in the same linear time)")
    print()
    print(f"{'n':>8} {'m':>8} {'c':>6} {'dim':>7} {'time (s)':>10} {'us/vertex':>10}")
    print("-" * 55)

    rows = []
    for steps in (200, 500, 1000, 2000, 4000, 8000, 16000):
        G = random_cactus(steps, seed=steps)
        if not nx.is_connected(G):
            continue
        n, m = G.number_of_nodes(), G.number_of_edges()
        c = num_cycle_blocks(G)

        start = time.perf_counter()
        dim, _ = min_dim_and_count(G)
        elapsed = time.perf_counter() - start

        per_vertex = 1e6 * elapsed / n
        rows.append(per_vertex)
        dim_str = "none" if dim is None else str(dim)
        print(f"{n:8d} {m:8d} {c:6d} {dim_str:>7} {elapsed:10.4f} {per_vertex:10.2f}")

    if rows:
        print("-" * 55)
        spread = max(rows) / min(rows)
        print(
            f"time per vertex varies by a factor of {spread:.2f} across the range; "
            "a constant\nfactor is the signature of linear scaling (a quadratic "
            "algorithm would grow\nproportionally to n)."
        )
    print()
    print("Note: m = n - 1 + c on every row, as guaranteed by Lemma 5.6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
