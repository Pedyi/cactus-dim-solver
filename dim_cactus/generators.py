"""
generators.py
=============

Random cactus generators and named families used throughout the paper.

A cactus is a connected graph in which every edge lies on at most one simple
cycle; equivalently any two distinct cycles share at most one vertex. For a
connected cactus, m = n - 1 + c, where c is the number of cycle blocks
(Lemma 5.6 in the paper).

Two generators are provided:

  * random_cactus:        the generic generator of Section 3 (attach a bridge or
                          a cycle of length in {3,4,5,6} to a random vertex),
                          with a maximum-degree cap.
  * random_class_C:       a generator that stays inside the structural class C of
                          Definition 4.7 by tracking "phase-0" attachment points,
                          so that every cycle has length divisible by 3 and every
                          maximal bridge segment between special vertices has a
                          number of edges divisible by 3.

The named families (cycle, path, star, double_star, cycle_bouquet, and the two
counterexamples of Section 4.4) are also defined here.

License: MIT.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx


# --------------------------------------------------------------------------- #
# Generic random cactus (Section 3)                                           #
# --------------------------------------------------------------------------- #
def random_cactus(n_steps: int, seed: int, max_degree: int = 6) -> nx.Graph:
    """
    Build a random connected cactus by iteratively attaching, to a uniformly
    random existing vertex, either a bridge (probability 1/2) or a cycle of
    length uniform in {3,4,5,6} (probability 1/2). Attachments that would exceed
    `max_degree` at the chosen vertex are skipped.

    Parameters
    ----------
    n_steps : number of attachment steps.
    seed : PRNG seed for reproducibility.
    max_degree : cap on the maximum degree (the bounded-degree regime).

    Returns
    -------
    A networkx.Graph that is a connected cactus.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_node(0)
    node = 1
    for _ in range(n_steps):
        att = rng.choice(list(G.nodes()))
        if G.degree(att) >= max_degree:
            continue
        if rng.random() < 0.5:
            # attach a bridge
            G.add_edge(att, node)
            node += 1
        else:
            # attach a cycle of length L
            L = rng.choice([3, 4, 5, 6])
            prev = att
            for _ in range(L - 1):
                G.add_edge(prev, node)
                prev = node
                node += 1
            G.add_edge(prev, att)
    return G


# --------------------------------------------------------------------------- #
# Random member of the class C (Definition 4.7)                               #
# --------------------------------------------------------------------------- #
def random_class_C(seed: int) -> Tuple[nx.Graph, List[List[object]]]:
    """
    Build a random member of the class C, returning (G, cycles) where `cycles`
    is the list of vertex-cycles (each a list of vertices in cyclic order).

    Invariants maintained:
      * every cycle block has length divisible by 3 (lengths in {3,6});
      * new attachments are made only at "phase-0" vertices (cycle vertices, or
        tree vertices at a distance divisible by 3 from their anchor), so that
        every maximal bridge segment between special vertices has a number of
        edges divisible by 3.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    cycles: List[List[object]] = []

    t = rng.choice([1, 2])
    base = [f"c0_{i}" for i in range(3 * t)]
    nx.add_cycle(G, base)
    cycles.append(base)

    attach_ok = set(base)  # phase-0 attachment points
    counter = [0]

    def fresh() -> str:
        counter[0] += 1
        return f"z{counter[0]}"

    for _ in range(rng.randint(1, 5)):
        att = rng.choice(sorted(attach_ok))
        if rng.random() < 0.5:
            # pendant path of 3s edges
            s = rng.choice([1, 1, 2])
            prev = att
            for j in range(1, 3 * s + 1):
                w = fresh()
                G.add_edge(prev, w)
                prev = w
                if j % 3 == 0:
                    attach_ok.add(w)  # phase-0 position along the path
        else:
            # bridge path of 3r edges to a new cycle C_{3 t2}
            r = rng.choice([1, 1])
            t2 = rng.choice([1, 2])
            prev = att
            for _ in range(3 * r):
                w = fresh()
                G.add_edge(prev, w)
                prev = w
            new_cycle = [prev] + [fresh() for _ in range(3 * t2 - 1)]
            nx.add_cycle(G, new_cycle)
            cycles.append(new_cycle)
            attach_ok.update(new_cycle)
    return G, cycles


# --------------------------------------------------------------------------- #
# Named families                                                              #
# --------------------------------------------------------------------------- #
def cycle(k: int) -> nx.Graph:
    """The cycle C_k on vertices 0..k-1."""
    return nx.cycle_graph(k)


def path(n: int) -> nx.Graph:
    """The path P_n on n vertices (n-1 edges)."""
    return nx.path_graph(n)


def star(k: int) -> nx.Graph:
    """The star K_{1,k} with centre 0 and k leaves."""
    return nx.star_graph(k)


def double_star(a: int, b: int) -> nx.Graph:
    """
    The double star S(a,b): an edge u-v with a leaves on u and b leaves on v.
    The balanced double star S(k,k) is the extremal family of Remark 5.3.
    """
    G = nx.Graph()
    G.add_edge("u", "v")
    for i in range(a):
        G.add_edge("u", f"p{i}")
    for i in range(b):
        G.add_edge("v", f"q{i}")
    return G


def cycle_bouquet(k: int, s: int) -> nx.Graph:
    """
    The s-bouquet of C_{3k}: s cycles of length 3k sharing one common vertex 'v'
    (Proposition 4.18). Vertices on cycle i are v, a{i}_1, ..., a{i}_{3k-1}.
    """
    G = nx.Graph()
    for i in range(s):
        prev = "v"
        for j in range(1, 3 * k):
            node = f"a{i}_{j}"
            G.add_edge(prev, node)
            prev = node
        G.add_edge(prev, "v")
    return G


def counterexample_bridged() -> Tuple[nx.Graph, List[List[object]]]:
    """
    Example 4.21 (bridged member of C with X=3): a central C6 with, at each of
    w0, w2, w4, a 3-edge bridge path to a triangle.

    Returns (G, cycles). Here n=21, m=24, m/3=8, but dim(G)=6.
    """
    G = nx.Graph()
    base = [f"w{i}" for i in range(6)]
    nx.add_cycle(G, base)
    cycles = [base]
    for i in (0, 2, 4):
        G.add_edge(f"w{i}", f"x{i}1")
        G.add_edge(f"x{i}1", f"x{i}2")
        G.add_edge(f"x{i}2", f"y{i}")
        tri = [f"y{i}", f"b{i}1", f"b{i}2"]
        nx.add_cycle(G, tri)
        cycles.append(tri)
    return G, cycles


def counterexample_glued() -> Tuple[nx.Graph, List[List[object]]]:
    """
    Example 4.22 (bridgeless member of C: failure of sigma/3): a central C6 with
    a triangle glued (sharing a vertex) at each of w0, w2, w4.

    Returns (G, cycles). Here n=12, m=sigma=15, sigma/3=5, but dim(G)=3.
    """
    G = nx.Graph()
    base = [f"w{i}" for i in range(6)]
    nx.add_cycle(G, base)
    cycles = [base]
    for i in (0, 2, 4):
        tri = [f"w{i}", f"b{i}1", f"b{i}2"]
        nx.add_cycle(G, tri)
        cycles.append(tri)
    return G, cycles


def infeasible_member_of_C(distance: int) -> nx.Graph:
    """
    Example 4.23: a central C6 with a [3-edge bridge -> triangle] gadget attached
    at w0 and at w_{distance}. For distance in {1,2} the graph is in C but admits
    NO DIM; for distance == 3 a DIM exists with dim = m/3 = 6.
    """
    G = nx.cycle_graph(6)  # vertices 0..5

    def add_gadget(anchor: int, tag: str) -> None:
        G.add_edge(anchor, f"{tag}x1")
        G.add_edge(f"{tag}x1", f"{tag}x2")
        G.add_edge(f"{tag}x2", f"{tag}u")
        G.add_edge(f"{tag}u", f"{tag}b1")
        G.add_edge(f"{tag}b1", f"{tag}b2")
        G.add_edge(f"{tag}b2", f"{tag}u")

    add_gadget(0, "g0")
    add_gadget(distance, "g1")
    return G


# --------------------------------------------------------------------------- #
# Structural helpers                                                          #
# --------------------------------------------------------------------------- #
def num_cycle_blocks(G: nx.Graph) -> int:
    """
    Number of cycle blocks c of a connected cactus. For a cactus this equals the
    cyclomatic number m - n + 1 (Lemma 5.6), and also the number of biconnected
    components of size >= 3.
    """
    return sum(1 for comp in nx.biconnected_components(G) if len(comp) >= 3)


def cycle_edge_sets(cycles: List[List[object]]):
    """
    Given the list of vertex-cycles, return (cyc_edges, cyc_verts):
      cyc_edges : set of frozenset edges that lie on some cycle;
      cyc_verts : dict mapping each cycle vertex to the list of cycle indices
                  whose cycle contains it.
    """
    cyc_edges = set()
    cyc_verts: dict = {}
    for ci, cyc in enumerate(cycles):
        L = len(cyc)
        for i in range(L):
            cyc_edges.add(frozenset((cyc[i], cyc[(i + 1) % L])))
        for v in cyc:
            cyc_verts.setdefault(v, []).append(ci)
    return cyc_edges, cyc_verts


def compute_X(G: nx.Graph, cycles: List[List[object]], M) -> int:
    """
    Compute the invariant X = sum_C x_C directly from a DIM M and the cycle list,
    where x_C is the number of vertices of cycle C saturated by an edge of M that
    does not belong to E(C) (Lemma 4.8 / Theorem 4.9).
    """
    _, cyc_verts = cycle_edge_sets(cycles)
    # Precompute, per cycle, its own edge set.
    own_edges = []
    for cyc in cycles:
        L = len(cyc)
        own_edges.append(
            set(frozenset((cyc[i], cyc[(i + 1) % L])) for i in range(L))
        )
    X = 0
    for u, v in M:
        fe = frozenset((u, v))
        for endpoint in (u, v):
            if endpoint in cyc_verts:
                for ci in cyc_verts[endpoint]:
                    if fe not in own_edges[ci]:
                        X += 1
    return X
