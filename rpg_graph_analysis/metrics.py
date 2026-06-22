"""Compatibility exports for graph-analysis metric helpers.
"""

from .edge_metrics import (  # noqa: F401
    compute_hamming,
    compute_shifted_similarity,
    digit_offsets,
    edge_arrays,
    histogram_rows,
    integer_histogram_rows,
    random_pairs,
    summary,
)
from .popularity import (  # noqa: F401
    bucket_for_frequency,
    popularity_rows,
    safe_pearson,
    safe_spearman,
    train_frequencies,
)
from .structural_metrics import (  # noqa: F401
    clustering_summary,
    component_summary,
    gini,
    import_igraph,
    indegree_histogram_rows,
    random_graph_clustering,
    random_indegree_summaries,
    reciprocity,
    undirected_graph,
)
