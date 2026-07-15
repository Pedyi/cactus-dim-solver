#!/usr/bin/env python3
"""
make_figures.py
===============

Regenerate the two counterexample figures (Examples 4.21 and 4.22) used in the
paper. Outputs ce1.png and ce2.png in this directory.

Run:

    python figures/make_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from dim_cactus import find_one_dim, counterexample_bridged, counterexample_glued


def draw(G, M, fname, title, seed=3):
    pos = nx.spring_layout(G, seed=seed, k=0.9, iterations=200)
    plt.figure(figsize=(5.2, 5.2))
    Mset = set(frozenset(e) for e in M)
    matched = [e for e in G.edges() if frozenset(e) in Mset]
    other = [e for e in G.edges() if frozenset(e) not in Mset]
    nx.draw_networkx_edges(G, pos, edgelist=other, width=1.3, edge_color="#999999")
    nx.draw_networkx_edges(G, pos, edgelist=matched, width=3.4, edge_color="#c0392b")
    VM = set()
    for e in M:
        VM.update(e)
    nx.draw_networkx_nodes(G, pos, nodelist=[v for v in G if v in VM],
                           node_size=170, node_color="#c0392b")
    nx.draw_networkx_nodes(G, pos, nodelist=[v for v in G if v not in VM],
                           node_size=120, node_color="#2c3e50")
    plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    here = os.path.dirname(__file__)

    Gb, _ = counterexample_bridged()
    Mb = find_one_dim(Gb)
    draw(Gb, Mb, os.path.join(here, "ce1.png"),
         r"Example 4.21: $m=24$, $m/3=8$, but $\dim=6$ (red)", seed=3)

    Gg, _ = counterexample_glued()
    Mg = find_one_dim(Gg)
    draw(Gg, Mg, os.path.join(here, "ce2.png"),
         r"Example 4.22: $m=\sigma=15$, $\sigma/3=5$, but $\dim=3$ (red)", seed=7)

    print("Wrote ce1.png and ce2.png to", here)


if __name__ == "__main__":
    main()
