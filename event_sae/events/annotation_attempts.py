"""Non-destructive merge and audit for cluster-annotation attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from event_sae import sha256_file
from event_sae.events.io import load_jsonl


_IMMUTABLE_FIELDS = (
    "cluster_id",
    "task_description",
    "model",
    "prompt_version",
    "prompt_text",
    "prompt_sha256",
    "prompt_input_policy",
    "generation_config",
    "phase_labeler_provenance",
    "phase_scheme",
    "allowed_phase_labels",
    "representative_sample_ids",
    "representative_clip_paths",
    "representative_frame_paths",
    "annotation_media_layout",
    "representative_progress_percents",
    "episode_coverage",
    "annotation_min_episode_coverage",
)


def _stable(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_success(row: dict) -> bool:
    if row.get("api_error") is not None or row.get("parse_error") is not None:
        return False
    phrase = row.get("phrase")
    phase = row.get("phase")
    if not isinstance(phrase, str) or not phrase.strip():
        return False
    if not isinstance(phase, str) or not phase.strip():
        return False
    allowed = {str(value) for value in row.get("allowed_phase_labels", [])}
    return not allowed or phase.strip() in allowed


def merge_annotation_attempts(
    *,
    clusters_path: Path,
    attempt_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    min_episode_coverage: float,
) -> dict:
    """Freeze exactly one successful annotation row per selected cluster.

    Attempts remain immutable. Failed rows are retained only in their source
    attempt files, while the merge manifest records which attempt supplied each
    frozen row. Multiple successful attempts for one cluster fail closed so the
    merge cannot silently choose among model outputs.
    """

    clusters_path = Path(clusters_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    attempt_paths = [Path(path).resolve() for path in attempt_paths]
    if not 0.0 <= min_episode_coverage <= 1.0:
        raise ValueError("min_episode_coverage must be within [0, 1]")
    if not clusters_path.is_file():
        raise FileNotFoundError(clusters_path)
    if not attempt_paths:
        raise ValueError("At least one annotation attempt is required")
    if len(attempt_paths) != len(set(attempt_paths)):
        raise ValueError("Annotation attempt paths must be unique")
    for path in attempt_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite merge output: {path}")

    selected_clusters = [
        row
        for row in load_jsonl(clusters_path)
        if float(row["episode_coverage"]) >= min_episode_coverage
    ]
    expected_ids = [str(row["cluster_id"]) for row in selected_clusters]
    if not expected_ids:
        raise ValueError("No clusters meet min_episode_coverage")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Selected clusters contain duplicate cluster IDs")
    expected_set = set(expected_ids)

    candidates: dict[str, list[tuple[int, dict]]] = {
        cluster_id: [] for cluster_id in expected_ids
    }
    attempt_summaries: list[dict] = []
    reference_by_id: dict[str, dict] = {}
    for attempt_index, path in enumerate(attempt_paths):
        rows = load_jsonl(path)
        seen: set[str] = set()
        success_count = 0
        for row in rows:
            cluster_id = str(row["cluster_id"])
            if cluster_id in seen:
                raise ValueError(
                    f"Duplicate cluster_id={cluster_id!r} in attempt {path}"
                )
            seen.add(cluster_id)
            if cluster_id not in expected_set:
                raise ValueError(
                    f"Attempt {path} contains unselected cluster {cluster_id!r}"
                )
            reference = reference_by_id.setdefault(cluster_id, row)
            differing = [
                field
                for field in _IMMUTABLE_FIELDS
                if _stable(reference.get(field)) != _stable(row.get(field))
            ]
            if differing:
                raise ValueError(
                    f"Annotation provenance changed across attempts for "
                    f"{cluster_id}: {differing}"
                )
            candidates[cluster_id].append((attempt_index, row))
            success_count += int(_is_success(row))
        attempt_summaries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "successful_rows": success_count,
                "failed_rows": len(rows) - success_count,
            }
        )

    merged_rows: list[dict] = []
    selected_attempts: dict[str, int] = {}
    missing: list[str] = []
    duplicate_successes: list[str] = []
    for cluster_id in expected_ids:
        successes = [
            (attempt_index, row)
            for attempt_index, row in candidates[cluster_id]
            if _is_success(row)
        ]
        if not successes:
            missing.append(cluster_id)
            continue
        if len(successes) > 1:
            duplicate_successes.append(cluster_id)
            continue
        attempt_index, row = successes[0]
        merged_rows.append(row)
        selected_attempts[cluster_id] = attempt_index
    if missing:
        raise ValueError(
            f"Clusters without a successful annotation: {missing[:10]}"
        )
    if duplicate_successes:
        raise ValueError(
            "Clusters with multiple successful annotation attempts: "
            f"{duplicate_successes[:10]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "event_sae_annotation_attempt_merge_v1",
        "clusters_path": str(clusters_path),
        "clusters_sha256": sha256_file(clusters_path),
        "min_episode_coverage": float(min_episode_coverage),
        "expected_clusters": len(expected_ids),
        "attempts": attempt_summaries,
        "selected_attempt_index_by_cluster": selected_attempts,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_rows": len(merged_rows),
        "selection_rule": "exactly_one_successful_attempt_per_cluster",
        "passed": len(merged_rows) == len(expected_ids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    return manifest


def filter_cluster_rows_by_coverage(
    *,
    clusters_path: Path,
    rows_path: Path,
    output_path: Path,
    manifest_path: Path,
    min_episode_coverage: float,
) -> dict:
    """Filter cluster-keyed rows using coverage from the raw cluster catalog.

    ``rows_path`` may be the raw cluster catalog itself, a selected annotation
    media artifact, or a frozen annotation superset.  Its order is preserved,
    and it must contain every cluster selected at the requested threshold.
    """

    clusters_path = Path(clusters_path).resolve()
    rows_path = Path(rows_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not 0.0 <= min_episode_coverage <= 1.0:
        raise ValueError("min_episode_coverage must be within [0, 1]")
    for path in (clusters_path, rows_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite filter output: {path}")

    clusters = load_jsonl(clusters_path)
    rows = load_jsonl(rows_path)
    cluster_by_id: dict[str, dict] = {}
    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        if cluster_id in cluster_by_id:
            raise ValueError(f"Duplicate cluster_id in clusters: {cluster_id}")
        cluster_by_id[cluster_id] = cluster

    row_by_id: dict[str, dict] = {}
    for row in rows:
        cluster_id = str(row["cluster_id"])
        if cluster_id in row_by_id:
            raise ValueError(f"Duplicate cluster_id in rows: {cluster_id}")
        if cluster_id not in cluster_by_id:
            raise ValueError(f"Row references unknown cluster: {cluster_id}")
        row_by_id[cluster_id] = row

    selected_ids = [
        cluster_id
        for cluster_id, cluster in cluster_by_id.items()
        if float(cluster["episode_coverage"]) >= min_episode_coverage
    ]
    missing = [
        cluster_id for cluster_id in selected_ids if cluster_id not in row_by_id
    ]
    if missing:
        raise ValueError(
            f"Selected clusters lack source rows: {missing[:10]}"
        )
    selected_set = set(selected_ids)
    filtered = [
        row for row in rows if str(row["cluster_id"]) in selected_set
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in filtered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "event_sae_cluster_coverage_filter_v1",
        "clusters_path": str(clusters_path),
        "clusters_sha256": sha256_file(clusters_path),
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "min_episode_coverage": float(min_episode_coverage),
        "input_rows": len(rows),
        "output_rows": len(filtered),
        "selected_cluster_ids": selected_ids,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "passed": len(filtered) == len(selected_ids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
    return manifest
