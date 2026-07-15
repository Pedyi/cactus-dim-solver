"""
formulas.py
===========

Directly computable versions of the paper's analytic quantities, used to check
the identities and bounds against the exact oracle.

Functions
---------
saturation_rhs(G, M)              : right-hand side of Theorem 4.4,
                                    m - sum_{v in V(M)} (d(v) - 2).
upper_bound_m_plus_l_over_3(G)    : the sharp upper bound (m + l)/3.
X_via_dim(G)                      : the invariant X = (m - 3 dim)/2 (Cor 4.10).
spectral_lower_bounded_degree(...) : Theorem 4/5 bounded-degree spectral bound.
quadratic_lower_bound(G)          : Theorem 5.6, ceil(m / (2 lambda1^2 - 1)).
cactus_spectral_lower_bound(G)    : Corollary 5.7, ceil((n-1+c)/(2 lambda1^2-1)).

All quantities are elementary; they exist so the verification scripts and tests
can compare "formula vs. exact dim" on many graphs.

License: MIT.
"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Iterable, Optional, Tuple

import networkx as nx
import numpy as np

from .exact import dim_size


def spectral_radius(G: nx.Graph) -> float:
    """Largest adjacency eigenvalue lambda_1(G)."""
    if G.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(G)
    return float(max(np.linalg.eigvalsh(A)))


def saturation_rhs(G: nx.Graph, M: Iterable[Tuple]) -> int:
    """
    Right-hand side of the saturation identity (Theorem 4.4):
        m - sum_{v in V(M)} (d(v) - 2).
    Equals 3|M| for any DIM M.
    """
    M = list(M)
    deg = dict(G.degree())
    VM = set()
    for u, v in M:
        VM.add(u)
        VM.add(v)
    return G.number_of_edges() - sum(deg[v] - 2 for v in VM)


def upper_bound_m_plus_l_over_3(G: nx.Graph) -> float:
    """The sharp upper bound (m + l)/3 of Theorem 4.4."""
    m = G.number_of_edges()
    l = sum(1 for _, d in G.degree() if d == 1)
    return (m + l) / 3.0


def X_via_dim(G: nx.Graph) -> Optional[int]:
    """
    The invariant X = (m - 3 dim(G))/2 (Corollary 4.10). Returns None if G admits
    no DIM. For G in the class C this equals sum_C x_C.
    """
    d = dim_size(G)
    if d is None:
        return None
    m = G.number_of_edges()
    val = m - 3 * d
    assert val % 2 == 0, "m - 3 dim must be even for a DIM-admitting graph"
    return val // 2


def spectral_lower_bounded_degree(n: int, lambda1: float, Delta0: int) -> float:
    """
    The bounded-degree spectral lower bound (paper, spectral section):
        dim(G) >= sqrt(Delta0)/(2 Delta0 - 1) * (n - 1)/lambda1.
    Returns the (real-valued) right-hand side.
    """
    return sqrt(Delta0) / (2 * Delta0 - 1) * (n - 1) / lambda1


def quadratic_lower_bound(G: nx.Graph) -> int:
    """
    The universal quadratic lower bound (Theorem 5.6):
        ceil(m / (2 lambda1^2 - 1)).
    """
    m = G.number_of_edges()
    lam = spectral_radius(G)
    return ceil(m / (2 * lam * lam - 1))


def cactus_spectral_lower_bound(G: nx.Graph) -> int:
    """
    The cactus-specific spectral lower bound (Corollary 5.7), using m = n - 1 + c:
        ceil((n - 1 + c) / (2 lambda1^2 - 1)).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    c = m - n + 1  # cyclomatic number = number of cycle blocks in a cactus
    lam = spectral_radius(G)
    return ceil((n - 1 + c) / (2 * lam * lam - 1))
