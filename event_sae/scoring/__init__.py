"""Sparse feature scoring and ranking with lazy public exports."""

__all__ = [
    "SparseTopKArtifact",
    "open_sparse_topk_artifact",
    "build_templates",
    "join_cluster_events",
    "score_cluster_features",
    "event_aligned_top_features_per_row",
    "event_aligned_suite_top_k",
    "window_mean_top_features_per_row",
    "window_mean_suite_top_k",
    "task_mean_top_features_per_task",
    "task_mean_suite_top_k",
    "alive_feature_ids",
    "random_alive_features",
]

_SCORE_MATRIX_EXPORTS = {
    "SparseTopKArtifact",
    "build_templates",
    "join_cluster_events",
    "open_sparse_topk_artifact",
    "score_cluster_features",
}
_RANKING_EXPORTS = {
    "alive_feature_ids",
    "event_aligned_suite_top_k",
    "event_aligned_top_features_per_row",
    "random_alive_features",
    "task_mean_suite_top_k",
    "task_mean_top_features_per_task",
    "window_mean_suite_top_k",
    "window_mean_top_features_per_row",
}


def __getattr__(name: str):
    if name in _SCORE_MATRIX_EXPORTS:
        from event_sae.scoring import score_matrix as _score_matrix

        return getattr(_score_matrix, name)
    if name in _RANKING_EXPORTS:
        from event_sae.scoring import rankings as _rankings

        return getattr(_rankings, name)
    raise AttributeError(f"module 'event_sae.scoring' has no attribute {name!r}")
