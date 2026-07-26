import json
from pathlib import Path

import pytest

from event_sae.events.annotation_attempts import (
    filter_cluster_rows_by_coverage,
    merge_annotation_attempts,
)
from event_sae.events.io import load_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cluster(cluster_id: str, coverage: float = 0.5) -> dict:
    return {"cluster_id": cluster_id, "episode_coverage": coverage}


def _annotation(cluster_id: str, *, error: str | None = None) -> dict:
    return {
        "cluster_id": cluster_id,
        "task_description": "Open the drawer.",
        "model": "gemini",
        "prompt_version": "v11",
        "prompt_text": "prompt",
        "prompt_sha256": "abc",
        "prompt_input_policy": {},
        "generation_config": {"temperature": 0.0},
        "phase_labeler_provenance": {},
        "phase_scheme": "robocasa_action",
        "allowed_phase_labels": ["pull"],
        "representative_sample_ids": ["sample"],
        "representative_clip_paths": [""],
        "representative_frame_paths": [["frame.jpg"]],
        "annotation_media_layout": "layout",
        "representative_progress_percents": [],
        "episode_coverage": 0.5,
        "annotation_min_episode_coverage": 0.3,
        "phrase": None if error else "pulling",
        "phase": None if error else "pull",
        "api_error": error,
        "parse_error": None,
        "raw_response": None if error else "{}",
    }


def test_merge_annotation_attempts_keeps_failed_attempts_separate(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.jsonl"
    first = tmp_path / "attempt01.jsonl"
    retry = tmp_path / "attempt02.jsonl"
    output = tmp_path / "frozen.jsonl"
    manifest = tmp_path / "frozen.manifest.json"
    _write_jsonl(clusters, [_cluster("a"), _cluster("b"), _cluster("rare", 0.1)])
    _write_jsonl(first, [_annotation("a"), _annotation("b", error="timeout")])
    _write_jsonl(retry, [_annotation("b")])

    report = merge_annotation_attempts(
        clusters_path=clusters,
        attempt_paths=[first, retry],
        output_path=output,
        manifest_path=manifest,
        min_episode_coverage=0.3,
    )

    assert [row["cluster_id"] for row in load_jsonl(output)] == ["a", "b"]
    assert report["attempts"][0]["failed_rows"] == 1
    assert report["selected_attempt_index_by_cluster"] == {"a": 0, "b": 1}
    assert report["passed"] is True


def test_merge_annotation_attempts_rejects_missing_and_duplicate_success(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.jsonl"
    first = tmp_path / "attempt01.jsonl"
    retry = tmp_path / "attempt02.jsonl"
    _write_jsonl(clusters, [_cluster("a"), _cluster("b")])
    _write_jsonl(first, [_annotation("a")])
    with pytest.raises(ValueError, match="without a successful"):
        merge_annotation_attempts(
            clusters_path=clusters,
            attempt_paths=[first],
            output_path=tmp_path / "missing.jsonl",
            manifest_path=tmp_path / "missing.json",
            min_episode_coverage=0.3,
        )

    _write_jsonl(retry, [_annotation("a"), _annotation("b")])
    with pytest.raises(ValueError, match="multiple successful"):
        merge_annotation_attempts(
            clusters_path=clusters,
            attempt_paths=[first, retry],
            output_path=tmp_path / "duplicate.jsonl",
            manifest_path=tmp_path / "duplicate.json",
            min_episode_coverage=0.3,
        )


def test_merge_annotation_attempts_rejects_prompt_drift(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.jsonl"
    first = tmp_path / "attempt01.jsonl"
    retry = tmp_path / "attempt02.jsonl"
    _write_jsonl(clusters, [_cluster("a")])
    _write_jsonl(first, [_annotation("a", error="timeout")])
    changed = _annotation("a")
    changed["prompt_sha256"] = "changed"
    _write_jsonl(retry, [changed])

    with pytest.raises(ValueError, match="provenance changed"):
        merge_annotation_attempts(
            clusters_path=clusters,
            attempt_paths=[first, retry],
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "output.json",
            min_episode_coverage=0.3,
        )


def test_filter_cluster_rows_by_coverage_preserves_source_order(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.jsonl"
    rows = tmp_path / "annotations.jsonl"
    output = tmp_path / "filtered.jsonl"
    manifest = tmp_path / "filtered.manifest.json"
    _write_jsonl(
        clusters,
        [
            _cluster("a", 0.5),
            _cluster("b", 0.2),
            _cluster("c", 0.4),
        ],
    )
    _write_jsonl(
        rows,
        [
            {"cluster_id": "c", "value": 3},
            {"cluster_id": "a", "value": 1},
            {"cluster_id": "b", "value": 2},
        ],
    )

    report = filter_cluster_rows_by_coverage(
        clusters_path=clusters,
        rows_path=rows,
        output_path=output,
        manifest_path=manifest,
        min_episode_coverage=0.4,
    )

    assert [row["cluster_id"] for row in load_jsonl(output)] == ["c", "a"]
    assert report["output_rows"] == 2
    assert report["passed"] is True


def test_filter_cluster_rows_by_coverage_rejects_missing_selected_row(
    tmp_path: Path,
) -> None:
    clusters = tmp_path / "clusters.jsonl"
    rows = tmp_path / "annotations.jsonl"
    _write_jsonl(clusters, [_cluster("a", 0.5), _cluster("b", 0.4)])
    _write_jsonl(rows, [{"cluster_id": "a"}])

    with pytest.raises(ValueError, match="lack source rows"):
        filter_cluster_rows_by_coverage(
            clusters_path=clusters,
            rows_path=rows,
            output_path=tmp_path / "filtered.jsonl",
            manifest_path=tmp_path / "filtered.manifest.json",
            min_episode_coverage=0.4,
        )
