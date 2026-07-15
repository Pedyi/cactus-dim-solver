"""
exact.py
========

Exact (brute-force) dominating induced matching (DIM) routines used to verify
every combinatorial and spectral claim in the paper

    "Extremal and Spectral Results on Dominating Induced Matchings in
     Cactus and Unicyclic Graphs"  (Asadzadeh & Behmaram).

A DIM of a graph G is a subset M of edges such that

  * M is an induced matching: the subgraph induced by the endpoints of M is
    1-regular (no two edges of M share a vertex or are joined by an edge of G);
  * M dominates G: every edge of G not in M is adjacent to exactly one edge of M.

If G admits a DIM then all DIMs have the same size, denoted dim(G)
(Theorem 2.3 / Theorem 4.1 in the paper).

These routines are intentionally simple and readable rather than fast: their
purpose is to be an obviously-correct oracle against which the linear-time
dynamic program (dim_cactus/linear_dp.py) and all analytic formulas are checked.

License: MIT.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import networkx as nx

Edge = Tuple[object, object]


def is_induced_matching(G: nx.Graph, M: Iterable[Edge]) -> bool:
    """Return True iff M induces a 1-regular subgraph on its endpoints."""
    M = list(M)
    verts = set()
    for u, v in M:
        if u in verts or v in verts:
            return False
        verts.add(u)
        verts.add(v)
    H = G.subgraph(verts)
    return all(deg == 1 for _, deg in H.degree())


def dominates_exactly_once(G: nx.Graph, M: Iterable[Edge]) -> bool:
    """Return True iff every edge outside M is adjacent to exactly one M-edge."""
    M = list(M)
    Mset = set(frozenset(e) for e in M)
    for e in G.edges():
        if frozenset(e) in Mset:
            continue
        count = 0
        for me in M:
            if set(e) & set(me):
                count += 1
                if count > 1:
                    break
        if count != 1:
            return False
    return True


def is_dim(G: nx.Graph, M: Iterable[Edge]) -> bool:
    """Return True iff M is a dominating induced matching of G."""
    M = list(M)
    return is_induced_matching(G, M) and dominates_exactly_once(G, M)


def dim_size(G: nx.Graph) -> Optional[int]:
    """
    Return dim(G), the size of a minimum DIM, or None if G admits no DIM.

    Since all DIMs share a common cardinality when one exists, the first size k
    for which a DIM is found (enumerating edge subsets by increasing size) is
    exactly dim(G).
    """
    E = list(G.edges())
    m = len(E)
    for k in range(m + 1):
        for M in combinations(E, k):
            if is_dim(G, M):
                return k
    return None


def find_one_dim(G: nx.Graph) -> Optional[List[Edge]]:
    """Return one minimum-size DIM (as a list of edges), or None."""
    E = list(G.edges())
    m = len(E)
    for k in range(m + 1):
        for M in combinations(E, k):
            if is_dim(G, M):
                return list(M)
    return None


def all_min_dims(G: nx.Graph) -> List[Tuple[Edge, ...]]:
    """Return the list of all minimum-size DIMs of G (empty if none)."""
    E = list(G.edges())
    m = len(E)
    for k in range(m + 1):
        found = [M for M in combinations(E, k) if is_dim(G, M)]
        if found:
            return found
    return []


def count_min_dims(G: nx.Graph) -> int:
    """Return the number of minimum-size DIMs of G (0 if none)."""
    return len(all_min_dims(G))
