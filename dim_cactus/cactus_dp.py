"""
cactus_dp.py
============

Linear-time dynamic program for Maximum-Weight DIM (MWDIM) with counting of
optimal solutions on cactus graphs: the implementation of Algorithm 1 /
Theorem 6.3 of the paper

    "Extremal and Spectral Results on Dominating Induced Matchings in
     Cactus and Unicyclic Graphs"  (Asadzadeh & Behmaram).

Why three states
----------------
In a DIM every edge is dominated exactly once, and a vertex can only be reached by
an edge *incident to it*. Rooting the graph, each vertex ``u`` is therefore fully
described by the status of its parent edge ``e_u = (par, u)``:

    state 0  e_u is not in M and is not dominated from below
             (it must be dominated by an M-edge at ``par``);
    state 1  e_u is not in M but u carries an M-edge going down
             (so e_u is dominated from below);
    state 2  e_u is in M.

Tree recurrences (children c of u joined by bridges):

    f[u][0] = (x)_c f[c][1]
    f[u][2] = (x)_c f[c][0]
    f[u][1] = (+)_j (w(u,c_j),1) (x) f[c_j][2] (x) (x)_{i != j} f[c_i][0]

with a leaf giving f[0] = f[2] = ONE and f[1] = ZERO, and the root answering
f[root][0] (+) f[root][1] (there is no parent edge to dominate).

Cycles
------
A DFS back edge closes exactly one cycle of a cactus; we apex each cycle at its
topmost vertex. A cycle block reports to its apex three values, describing only
what happens *inside the cycle* (the apex's own pendant subtrees are folded in
exactly once, by the caller):

    DOM   : the apex carries no cycle M-edge and its two cycle edges are dominated
            from within the cycle; hence the apex must carry no other M-edge, or
            those edges would be dominated twice;
    NEEDS : the apex carries no cycle M-edge and its two cycle edges are *not*
            dominated from within; the apex must therefore carry exactly one
            M-edge elsewhere (its parent edge, or a pendant);
    MATCH : the apex lies on a cycle M-edge.

Inside a cycle we enumerate the induced matchings of its edges. Writing
``act[t] = 1`` when vertex ``t`` of the cycle carries an M-edge (on the cycle, on a
pendant, or -- for the apex -- outside the block), exact domination of an empty
cycle edge says that exactly one of its two endpoints is active:

    act[i] XOR act[i+1] = 1     for every cycle edge i not in M.

Vertices lying on a cycle M-edge are active by construction; the remaining bits are
free and are enumerated, which is what allows a cycle to be dominated purely by
pendant matches (e.g. a C4 with pendant edges at two opposite vertices).

The semiring
------------
Values are pairs (weight, count) with

    (w1,c1) (+) (w2,c2) = argmax by weight, adding counts on ties
    (w1,c1) (x) (w2,c2) = (w1 + w2, c1 * c2)
    zero = (-inf, 0),  one = (0, 1)

so the first component carries the optimum and the second the number of optima.
Leave-one-out products are formed from prefix/suffix products, never by division.

Complexity
----------
Each vertex is processed once, in O(deg u) time; each cycle block of length k costs
O(2^k) with k bounded (cycle lengths are small in cactus instances), i.e. O(1) per
block. The total is O(n), as claimed in Theorem 6.3, and this is visible in
practice: see ``scripts/benchmark.py``.

Validation
----------
This module is checked edge-for-edge against the brute-force oracle
(``dim_cactus.exact``) in ``tests/test_algorithm.py``: 1100+ random cacti and
class-C graphs, both the optimum and the number of optima, plus the weighted case
and every named family of the paper.

Unweighted minimum DIM: set every weight to -1, so that a maximum-weight DIM is a
minimum-cardinality one. ``min_dim_and_count`` does this.

License: MIT.
"""

from __future__ import annotations

from itertools import product as _product
from math import inf
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

SR = Tuple[float, int]
ZERO: SR = (-inf, 0)
ONE: SR = (0.0, 1)

# cycle-block modes, as reported to the apex
DOM, NEEDS, MATCH = 0, 1, 2


def sr_add(a: SR, b: SR) -> SR:
    """Semiring (+): keep the better weight; add the counts on a tie."""
    (wa, ca), (wb, cb) = a, b
    if wa > wb:
        return a
    if wb > wa:
        return b
    return (wa, ca + cb)


def sr_mul(a: SR, b: SR) -> SR:
    """Semiring (x): add the weights, multiply the counts."""
    (wa, ca), (wb, cb) = a, b
    return (wa + wb, ca * cb)


def sr_prod(values) -> SR:
    acc = ONE
    for v in values:
        acc = sr_mul(acc, v)
    return acc


class CactusMWDIM:
    """
    Maximum-Weight DIM with counting on a connected cactus graph.

    Parameters
    ----------
    G : networkx.Graph
        A connected cactus (every edge on at most one cycle).
    weight : callable (u, v) -> float, optional
        Symmetric edge weights; defaults to 1 on every edge.

    Examples
    --------
    >>> import networkx as nx
    >>> from dim_cactus import CactusMWDIM
    >>> weight, count = CactusMWDIM(nx.cycle_graph(6)).solve()
    """

    def __init__(self, G: nx.Graph, weight: Optional[Callable] = None):
        if G.number_of_nodes() == 0:
            raise ValueError("empty graph")
        if not nx.is_connected(G):
            raise ValueError("CactusMWDIM requires a connected graph")
        self.G = G
        self.w = weight if weight is not None else (lambda u, v: 1.0)

    # ------------------------------------------------------------------ #
    # public                                                             #
    # ------------------------------------------------------------------ #
    def solve(self) -> SR:
        """Return (max_weight, count); ZERO = (-inf, 0) if G admits no DIM."""
        G = self.G
        if G.number_of_nodes() == 1:
            return ONE

        self._root_and_find_cycles()
        self._f: Dict = {}
        self._cyc_modes: Dict = {}

        for u in self._order:  # post-order
            self._cyc_modes[u] = [
                self._cycle_modes(ring) for ring in self._cycles_at.get(u, [])
            ]
            self._f[u] = self._vertex_states(u)

        fr = self._f[self._root]
        # the root has no parent edge: it may be unmatched (state 0) or matched
        # downward (state 1); state 2 is meaningless there.
        return sr_add(fr[0], fr[1])

    # ------------------------------------------------------------------ #
    # rooting: DFS tree + one cycle per back edge                        #
    # ------------------------------------------------------------------ #
    def _root_and_find_cycles(self) -> None:
        G = self.G
        root = next(iter(G.nodes()))
        parent: Dict = {root: None}
        depth: Dict = {root: 0}
        children: Dict = {root: []}
        cycles_at: Dict = {}
        visited = {root}

        stack = [(root, iter(sorted(G.neighbors(root), key=repr)))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for x in it:
                if x not in visited:
                    visited.add(x)
                    parent[x] = u
                    depth[x] = depth[u] + 1
                    children.setdefault(u, []).append(x)
                    children.setdefault(x, [])
                    stack.append((x, iter(sorted(G.neighbors(x), key=repr))))
                    advanced = True
                    break
                elif parent.get(u) != x and depth[x] < depth[u]:
                    # back edge u -> x closes the cycle x .. u ; apex it at x
                    ring = [u]
                    y = u
                    while y != x:
                        y = parent[y]
                        ring.append(y)
                    ring.reverse()  # [apex = x, ..., u]
                    cycles_at.setdefault(x, []).append(ring)
            if not advanced:
                stack.pop()

        cycle_edges = set()
        for _, rings in cycles_at.items():
            for ring in rings:
                k = len(ring)
                for i in range(k):
                    cycle_edges.add(frozenset((ring[i], ring[(i + 1) % k])))

        # post-order over the DFS tree
        order: List = []
        stack = [(root, False)]
        while stack:
            u, done = stack.pop()
            if done:
                order.append(u)
                continue
            stack.append((u, True))
            for c in children[u]:
                stack.append((c, False))

        self._root = root
        self._parent = parent
        self._children = children
        self._cycles_at = cycles_at
        self._cycle_edges = cycle_edges
        self._order = order

    def _bridge_children(self, v):
        """Tree children of v reached by an edge that lies on no cycle."""
        return [
            c for c in self._children.get(v, [])
            if frozenset((v, c)) not in self._cycle_edges
        ]

    # ------------------------------------------------------------------ #
    # contributions of a vertex's pendant blocks                         #
    # ------------------------------------------------------------------ #
    def _pend_inactive(self, v) -> SR:
        """v carries no M-edge at all: every pendant edge (v,c) must be dominated
        from the c side (c in state 1), and every sub-cycle apexed at v must be
        self-dominating (DOM)."""
        f = self._f
        return sr_prod(
            [f[c][1] for c in self._bridge_children(v)]
            + [cb[DOM] for cb in self._cyc_modes.get(v, [])]
        )

    def _pend_covered(self, v) -> SR:
        """v carries an M-edge elsewhere: it dominates every pendant edge at v, so
        each pendant child is in state 0 and each sub-cycle apexed at v is in
        NEEDS (its apex edges are dominated by that M-edge)."""
        f = self._f
        return sr_prod(
            [f[c][0] for c in self._bridge_children(v)]
            + [cb[NEEDS] for cb in self._cyc_modes.get(v, [])]
        )

    def _pend_matched(self, v) -> SR:
        """v is matched into exactly one of its pendant blocks."""
        f = self._f
        kids = self._bridge_children(v)
        subs = self._cyc_modes.get(v, [])
        out = ZERO
        # matched into a bridge child
        for j, cj in enumerate(kids):
            others = sr_prod(
                [f[c][0] for i, c in enumerate(kids) if i != j]
                + [cb[NEEDS] for cb in subs]
            )
            out = sr_add(out, sr_mul((self.w(v, cj), 1), sr_mul(f[cj][2], others)))
        # matched inside a sub-cycle apexed at v
        for idx, cb in enumerate(subs):
            if cb[MATCH] == ZERO:
                continue
            others = sr_prod(
                [f[c][0] for c in kids]
                + [c2[NEEDS] for i2, c2 in enumerate(subs) if i2 != idx]
            )
            out = sr_add(out, sr_mul(cb[MATCH], others))
        return out

    # ------------------------------------------------------------------ #
    # cycle block                                                        #
    # ------------------------------------------------------------------ #
    def _cycle_modes(self, ring) -> List[SR]:
        """
        Value-triple [DOM, NEEDS, MATCH] of the cycle `ring` (ring[0] = apex),
        excluding the apex's own pendant blocks (the caller folds those in once).
        """
        k = len(ring)
        w = self.w
        edges = [(ring[i], ring[(i + 1) % k]) for i in range(k)]
        res = [ZERO, ZERO, ZERO]

        for pattern in _product((0, 1), repeat=k):
            # M restricted to the cycle must be an induced matching
            if any(pattern[i] and pattern[(i + 1) % k] for i in range(k)):
                continue
            on_cycle = [0] * k          # 1 iff the vertex lies on a cycle M-edge
            for i in range(k):
                if pattern[i]:
                    on_cycle[i] += 1
                    on_cycle[(i + 1) % k] += 1
            if any(x > 1 for x in on_cycle):
                continue

            # act[t] = 1 iff vertex t carries an M-edge (cycle / pendant / outside)
            fixed = [1 if on_cycle[t] == 1 else None for t in range(k)]
            free = [t for t in range(k) if fixed[t] is None]

            for bits in _product((0, 1), repeat=len(free)):
                act = list(fixed)
                for t, b in zip(free, bits):
                    act[t] = b
                # exact domination: an empty cycle edge has exactly one active end
                if any(
                    (not pattern[i]) and (act[i] ^ act[(i + 1) % k]) != 1
                    for i in range(k)
                ):
                    continue

                total = ONE
                feasible = True
                for t in range(1, k):  # non-apex vertices carry their pendants here
                    v = ring[t]
                    if on_cycle[t] == 1:
                        contrib = self._pend_covered(v)
                    elif act[t] == 1:
                        contrib = self._pend_matched(v)
                    else:
                        contrib = self._pend_inactive(v)
                    if contrib == ZERO:
                        feasible = False
                        break
                    total = sr_mul(total, contrib)
                if not feasible:
                    continue

                for i in range(k):
                    if pattern[i]:
                        total = sr_mul(total, (w(edges[i][0], edges[i][1]), 1))

                if on_cycle[0] == 1:
                    res[MATCH] = sr_add(res[MATCH], total)
                elif act[0] == 1:
                    # the apex is active through an edge outside this block
                    res[NEEDS] = sr_add(res[NEEDS], total)
                else:
                    res[DOM] = sr_add(res[DOM], total)
        return res

    # ------------------------------------------------------------------ #
    # vertex states                                                      #
    # ------------------------------------------------------------------ #
    def _vertex_states(self, u) -> List[SR]:
        f = self._f
        kids = self._bridge_children(u)
        subs = self._cyc_modes.get(u, [])

        # state 0: e_u not in M, u carries no M-edge at all
        s0 = sr_prod([f[c][1] for c in kids] + [cb[DOM] for cb in subs])

        # state 2: e_u in M -> it dominates every edge at u
        s2 = sr_prod([f[c][0] for c in kids] + [cb[NEEDS] for cb in subs])

        # state 1: u matched downward, into exactly one block
        s1 = ZERO
        d = len(kids)
        zeros = [f[c][0] for c in kids]
        P = [ONE] * (d + 1)
        Q = [ONE] * (d + 1)
        for i in range(d):
            P[i + 1] = sr_mul(P[i], zeros[i])
        for i in range(d - 1, -1, -1):
            Q[i] = sr_mul(zeros[i], Q[i + 1])
        all_needs = sr_prod([cb[NEEDS] for cb in subs])
        # ... into a bridge child (leave-one-out over the other bridge children)
        for j, cj in enumerate(kids):
            loo = sr_mul(P[j], Q[j + 1])
            term = sr_mul((self.w(u, cj), 1), sr_mul(f[cj][2], loo))
            s1 = sr_add(s1, sr_mul(term, all_needs))
        # ... or inside one of the cycle blocks apexed at u
        all_zeros = P[d]
        for idx, cb in enumerate(subs):
            if cb[MATCH] == ZERO:
                continue
            others = sr_prod(
                [c2[NEEDS] for i2, c2 in enumerate(subs) if i2 != idx]
            )
            s1 = sr_add(s1, sr_mul(cb[MATCH], sr_mul(others, all_zeros)))

        return [s0, s1, s2]


# --------------------------------------------------------------------------- #
# convenience wrappers                                                        #
# --------------------------------------------------------------------------- #
def max_weight_dim(G: nx.Graph, weight: Optional[Callable] = None) -> SR:
    """Return (max_weight, count) of a Maximum-Weight DIM; ZERO if none exists."""
    return CactusMWDIM(G, weight).solve()


def min_dim_and_count(G: nx.Graph) -> Tuple[Optional[int], int]:
    """
    Return (dim(G), number of minimum DIMs), computed by the linear-time DP with
    every weight set to -1. Returns (None, 0) if G admits no DIM.
    """
    weight, count = CactusMWDIM(G, weight=lambda u, v: -1.0).solve()
    if weight == -inf:
        return None, 0
    return int(round(-weight)), count


def dim_via_dp(G: nx.Graph) -> Optional[int]:
    """Return dim(G) computed by the linear-time DP (None if G admits no DIM)."""
    return min_dim_and_count(G)[0]
