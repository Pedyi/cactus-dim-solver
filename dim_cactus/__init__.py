"""
dim_cactus
==========

Reference implementation accompanying the paper

    "Extremal and Spectral Results on Dominating Induced Matchings in
     Cactus and Unicyclic Graphs"  (P. Asadzadeh and A. Behmaram).

Modules
-------
exact       : brute-force DIM oracle (ground truth) -- dim, counting, membership.
generators  : random cactus / class-C generators and named families.
cactus_dp   : the linear-time O(n) DP with the optimisation-counting semiring on
              cactus graphs (Algorithm 1 / Theorem 6.3) -- weighted MWDIM and the
              number of optimal solutions.
tree_dp     : the same three-state recurrence restricted to trees (kept as a
              small, self-contained illustration of the core recurrence).
formulas    : the paper's closed-form / structural quantities (saturation
              identity, X, spectral bounds) as directly computable functions.

The exact oracle is the ground truth against which the linear-time DP and every
analytic claim are verified.
"""

from .exact import (
    is_dim,
    dim_size,
    find_one_dim,
    all_min_dims,
    count_min_dims,
)
from .generators import (
    random_cactus,
    random_class_C,
    cycle,
    path,
    star,
    double_star,
    cycle_bouquet,
    counterexample_bridged,
    counterexample_glued,
    infeasible_member_of_C,
    num_cycle_blocks,
    compute_X,
    cycle_edge_sets,
)
from .tree_dp import (
    tree_mwdim,
    tree_min_dim_and_count,
)
from .cactus_dp import (
    CactusMWDIM,
    max_weight_dim,
    min_dim_and_count,
    dim_via_dp,
)
from .formulas import (
    saturation_rhs,
    upper_bound_m_plus_l_over_3,
    X_via_dim,
    spectral_lower_bounded_degree,
    quadratic_lower_bound,
    cactus_spectral_lower_bound,
    spectral_radius,
)

__all__ = [
    "is_dim", "dim_size", "find_one_dim", "all_min_dims", "count_min_dims",
    "random_cactus", "random_class_C", "cycle", "path", "star", "double_star",
    "cycle_bouquet", "counterexample_bridged", "counterexample_glued",
    "infeasible_member_of_C", "num_cycle_blocks", "compute_X", "cycle_edge_sets",
    "tree_mwdim", "tree_min_dim_and_count",
    "CactusMWDIM", "max_weight_dim", "min_dim_and_count", "dim_via_dp",
    "saturation_rhs", "upper_bound_m_plus_l_over_3", "X_via_dim",
    "spectral_lower_bounded_degree", "quadratic_lower_bound",
    "cactus_spectral_lower_bound", "spectral_radius",
]

__version__ = "1.0.0"
