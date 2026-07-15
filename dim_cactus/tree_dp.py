"""
tree_dp.py
==========

Exact linear-time dynamic program for Maximum-Weight DIM with counting on TREES,
implementing the three-state edge-domination recurrence that forms the core of
Algorithm 1 in the paper (the cactus algorithm extends this with cycle-breaking;
see the paper, Section 6, and the note in ``README`` on the reference solvers).

States (rooted tree; e_u = (parent, u)):
    0  e_u not in M and not dominated from below  (needs domination from above)
    1  e_u not in M but u is matched to a child    (dominated from below)
    2  e_u in M

Recurrences:
    f[u][0] = prod_c f[c][1]
    f[u][2] = prod_c f[c][0]
    f[u][1] = sum_j (w(u,c_j),1) (x) f[c_j][2] (x) prod_{i != j} f[c_i][0]

Leaf: f[0]=f[2]=ONE, f[1]=ZERO. Root answer: f[root][0] (+) f[root][1].

Runs in O(n) time and O(n) space with an optimisation-counting semiring, and is
validated against the brute-force oracle over random trees in the test suite.

License: MIT.
"""

from __future__ import annotations

from math import inf
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

SR = Tuple[float, int]
ZERO: SR = (-inf, 0)
ONE: SR = (0.0, 1)


def _add(a: SR, b: SR) -> SR:
    if a[0] > b[0]:
        return a
    if b[0] > a[0]:
        return b
    return (a[0], a[1] + b[1])


def _mul(a: SR, b: SR) -> SR:
    return (a[0] + b[0], a[1] * b[1])


def _prod(vals) -> SR:
    acc = ONE
    for v in vals:
        acc = _mul(acc, v)
    return acc


def tree_mwdim(T: nx.Graph, weight: Optional[Callable] = None) -> SR:
    """
    Return (max_weight, count) of a Maximum-Weight DIM of the tree T, or
    ZERO = (-inf, 0) if T admits no DIM.
    """
    if T.number_of_nodes() == 0:
        return ZERO
    if not nx.is_tree(T):
        raise ValueError("tree_mwdim requires a tree")
    w = weight if weight is not None else (lambda u, v: 1.0)

    root = next(iter(T.nodes()))
    parent = {root: None}
    children: Dict = {root: []}
    order: List = []
    visited = {root}
    st = [(root, iter(sorted(T.neighbors(root), key=repr)))]
    while st:
        u, it = st[-1]
        adv = False
        for x in it:
            if x not in visited:
                visited.add(x)
                parent[x] = u
                children.setdefault(u, []).append(x)
                children.setdefault(x, [])
                st.append((x, iter(sorted(T.neighbors(x), key=repr))))
                adv = True
                break
        if not adv:
            order.append(u)
            st.pop()

    f: Dict = {}
    for u in order:  # post-order
        kids = children[u]
        f0 = _prod([f[c][1] for c in kids])
        f2 = _prod([f[c][0] for c in kids])
        # f1: choose exactly one child c_j matched to u
        d = len(kids)
        zeros = [f[c][0] for c in kids]
        P = [ONE] * (d + 1)
        Q = [ONE] * (d + 1)
        for i in range(d):
            P[i + 1] = _mul(P[i], zeros[i])
        for i in range(d - 1, -1, -1):
            Q[i] = _mul(zeros[i], Q[i + 1])
        f1 = ZERO
        for j, cj in enumerate(kids):
            loo = _mul(P[j], Q[j + 1])
            f1 = _add(f1, _mul((w(u, cj), 1), _mul(f[cj][2], loo)))
        f[u] = [f0, f1, f2]

    return _add(f[root][0], f[root][1])


def tree_min_dim_and_count(T: nx.Graph) -> Tuple[Optional[int], int]:
    """Return (dim(T), #minimum DIMs) via the linear DP with weight -1."""
    wt, cnt = tree_mwdim(T, weight=lambda u, v: -1.0)
    if wt == -inf:
        return None, 0
    return int(round(-wt)), cnt
