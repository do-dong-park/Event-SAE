"""Blind, two-stage human review for clustered event annotations.

This module owns the browser-safe review dataset, atomic review persistence,
annotation finalization, and the loopback-only review HTTP service. GR00T
Stage 4 and Oracle result exploration live in
:mod:`event_sae.groot.results_browser`.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import tempfile
import threading
from collections import defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from event_sae import resolve_groot_artifact_path
from event_sae import sha256_file as _sha256
from event_sae.events.cluster import _validate_cluster_bundle, build_task_vectors
from event_sae.events.io import load_jsonl
from event_sae.events.prompts import (
    PAPER_PHASE_DESCRIPTIONS,
    ROBOCASA_ACTION_PHASE_DESCRIPTIONS,
)


REVIEW_STAGES = {"blind", "adjudication"}


REVIEW_DOCUMENT_FORMAT = "event_sae_stage3_human_cluster_reviews_v2"


PHASE_DESCRIPTIONS = {
    **PAPER_PHASE_DESCRIPTIONS,
    **ROBOCASA_ACTION_PHASE_DESCRIPTIONS,
}


def _phase_descriptions(annotation: dict) -> dict[str, str]:
    labels = [str(value) for value in annotation["allowed_phase_labels"]]
    artifact_descriptions = annotation.get("phase_descriptions") or {}
    return {
        label: str(
            artifact_descriptions.get(
                label,
                PHASE_DESCRIPTIONS.get(
                    label,
                    "Use the shared visual state at the temporal center.",
                ),
            )
        )
        for label in labels
    }


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "task"


def render_contact_sheets(
    clusters_path: Path,
    output_dir: Path,
    *,
    canonical_only: bool = True,
    tile_size: int = 128,
) -> list[Path]:
    """Render one chronological representative-frame sheet per task."""

    clusters_path = resolve_groot_artifact_path(clusters_path).resolve()
    if not clusters_path.is_file():
        raise FileNotFoundError(f"clusters.jsonl not found: {clusters_path}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    clusters = load_jsonl(clusters_path)
    if canonical_only:
        clusters = [
            cluster
            for cluster in clusters
            if cluster.get("meets_min_coverage", False)
        ]
    if not clusters:
        raise ValueError("No clusters matched the rendering criteria")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for cluster in clusters:
        frame_groups = cluster.get("representative_frame_paths", [])
        if not frame_groups or not frame_groups[0]:
            raise ValueError(
                f"Cluster has no representative frames: {cluster['cluster_id']}"
            )
        by_task[cluster["task_description"]].append(cluster)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty contact-sheet directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    label_width = 360
    row_height = tile_size + 28
    written: list[Path] = []

    for task_description, task_clusters in sorted(by_task.items()):
        task_clusters.sort(key=lambda cluster: cluster["cluster_id"])
        max_frames = max(
            len(cluster["representative_frame_paths"][0])
            for cluster in task_clusters
        )
        canvas = Image.new(
            "RGB",
            (
                label_width + max_frames * tile_size,
                42 + len(task_clusters) * row_height,
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), task_description, fill="black")

        for row_idx, cluster in enumerate(task_clusters):
            y = 42 + row_idx * row_height
            short_cluster_id = cluster["cluster_id"].rsplit(
                "_cluster_", maxsplit=1
            )[-1]
            label = (
                f"cluster_{short_cluster_id}  n={cluster['num_members']}  "
                f"coverage={float(cluster['episode_coverage']):.2f}  "
                "progress="
                f"{float(cluster['cluster_mean_progress_percent']):.2f}"
            )
            draw.text((8, y + 4), label, fill="black")
            draw.line(
                (0, y + row_height - 1, canvas.width, y + row_height - 1),
                fill="#cccccc",
            )
            for frame_idx, frame_path_value in enumerate(
                cluster["representative_frame_paths"][0]
            ):
                frame_path = resolve_groot_artifact_path(frame_path_value)
                if not frame_path.is_file():
                    raise FileNotFoundError(
                        "Representative frame not found for "
                        f"{cluster['cluster_id']}: {frame_path}"
                    )
                with Image.open(frame_path) as image:
                    tile = image.convert("RGB").resize(
                        (tile_size, tile_size),
                        Image.Resampling.LANCZOS,
                    )
                canvas.paste(tile, (label_width + frame_idx * tile_size, y))

        output_path = output_dir / f"{_slugify(task_description)}.jpg"
        canvas.save(output_path, format="JPEG", quality=92)
        written.append(output_path)

    manifest = {
        "format": "event_sae_cluster_contact_sheets_v1",
        "clusters_path": str(clusters_path),
        "canonical_only": canonical_only,
        "num_clusters": len(clusters),
        "num_tasks": len(by_task),
        "sheets": [str(path) for path in written],
    }
    (output_dir / "contact_sheet_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return written


def finalize_reviewed_annotations(
    *,
    annotations_path: Path,
    output_path: Path,
    expected_clusters: int | None = None,
    reviews_path: Path | None,
    assume_approved: bool,
) -> list[dict]:
    """Materialize reviewed annotations while retaining review provenance.

    When ``expected_clusters`` is ``None``, the validated annotation artifact
    defines the cluster count instead of relying on a version-specific CLI
    constant.
    """

    annotations_path = resolve_groot_artifact_path(
        annotations_path
    ).resolve()
    output_path = resolve_groot_artifact_path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        )
    if assume_approved == (reviews_path is not None):
        raise ValueError("Choose exactly one of reviews_path or assume_approved")

    annotations = load_jsonl(annotations_path)
    by_id: dict[str, dict] = {}
    for row in annotations:
        cluster_id = str(row["cluster_id"])
        if cluster_id in by_id:
            raise ValueError(f"Duplicate annotation cluster_id: {cluster_id}")
        if row.get("api_error") is not None or row.get("parse_error") is not None:
            raise ValueError(
                f"Annotation contains an API/parse error: {cluster_id}"
            )
        phrase = str(row.get("phrase", "")).strip()
        phase = str(row.get("phase", "")).strip()
        if not phrase or not phase:
            raise ValueError(
                f"Annotation has an empty phrase/phase: {cluster_id}"
            )
        allowed = set(row.get("allowed_phase_labels") or [])
        if allowed and phase not in allowed:
            raise ValueError(
                f"Annotation phase is outside its vocabulary: {cluster_id}"
            )
        by_id[cluster_id] = row
    if not by_id:
        raise ValueError("Annotation artifact is empty")
    if expected_clusters is not None and len(by_id) != expected_clusters:
        raise ValueError(
            f"Expected {expected_clusters} clusters, got {len(by_id)}"
        )

    review_by_id: dict[str, dict] = {}
    if reviews_path is not None:
        reviews_path = resolve_groot_artifact_path(reviews_path).resolve()
        document = json.loads(reviews_path.read_text(encoding="utf-8"))
        if document.get("annotations_sha256") != _sha256(annotations_path):
            raise ValueError(
                "Review document points to a different annotation file"
            )
        for review in document.get("reviews", []):
            cluster_id = str(review["cluster_id"])
            if cluster_id in review_by_id:
                raise ValueError(f"Duplicate review cluster_id: {cluster_id}")
            review_by_id[cluster_id] = review
        if set(review_by_id) != set(by_id):
            raise ValueError("Every annotation must have exactly one review")
        review_mode = "human_review"
    else:
        review_mode = "user_authorized_assumed_review"

    finalized: list[dict] = []
    finalized_at = datetime.now(timezone.utc).isoformat()
    source_annotations_sha256 = _sha256(annotations_path)
    for cluster_id in sorted(by_id):
        row = dict(by_id[cluster_id])
        if reviews_path is not None:
            review = review_by_id[cluster_id]
            if review.get("review_stage", "adjudicated") != "adjudicated":
                raise ValueError(
                    f"Review has not completed model adjudication: {cluster_id}"
                )
            verdict = str(review.get("verdict", ""))
            if verdict not in {"approved", "corrected"}:
                raise ValueError(
                    f"Unresolved review verdict for {cluster_id}: {verdict}"
                )
            row["phrase"] = str(review["human_phrase"])
            row["phase"] = str(review["human_phase"])
            row["review_verdict"] = verdict
            row["reviewer"] = str(review.get("reviewer", "human"))
            row["reviewed_at"] = review.get("reviewed_at")
            row["mixed_cluster"] = bool(review.get("mixed_cluster", False))
            row["visually_insufficient"] = bool(
                review.get("visually_insufficient", False)
            )
            row["phrase_phase_consistent"] = review.get(
                "phrase_phase_consistent"
            )
            row["phase_corrected"] = review.get("phase_corrected")
        else:
            row["review_verdict"] = "assumed_approved"
            row["reviewer"] = "user_authorized_assumed_review"
            row["reviewed_at"] = None
        row["review_mode"] = review_mode
        row["actual_human_review_completed"] = reviews_path is not None
        row["finalized_at"] = finalized_at
        row["source_annotations_sha256"] = source_annotations_sha256
        finalized.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in finalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return finalized


def _normalize_projection(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    center = (coordinates.max(axis=0) + coordinates.min(axis=0)) / 2.0
    half_span = (coordinates.max(axis=0) - coordinates.min(axis=0)) / 2.0
    half_span = np.where(half_span > 1e-12, half_span, 1.0)
    return np.clip((coordinates - center) / half_span, -1.0, 1.0)


def _project_task(
    records: list[dict],
    *,
    method: str,
    random_state: int,
    block_normalization: str,
) -> np.ndarray:
    vectors = build_task_vectors(
        records,
        vision_weight=1.0,
        state_weight=0.5,
        progress_weight=0.4,
        block_normalization=block_normalization,
    )
    if len(records) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if method == "pca" or len(records) < 4:
        coordinates = PCA(n_components=2).fit_transform(vectors)
    elif method == "tsne":
        perplexity = min(30.0, max(2.0, (len(records) - 1) / 3.0))
        coordinates = TSNE(
            n_components=2,
            metric="cosine",
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(vectors)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return _normalize_projection(coordinates)


def build_blind_cluster_review_dataset(
    *,
    event_features_path: Path,
    clusters_path: Path,
    assignments_path: Path,
    annotations_path: Path,
    representative_media_clusters_path: Path | None = None,
    condition_id: str = "default",
    condition_label: str = "Cluster review",
    media_layout: str = "unspecified",
    condition_metadata: dict[str, str] | None = None,
    projection_method: str = "tsne",
    random_state: int = 0,
    validate_media: bool = True,
    block_normalization: str = "balanced",
) -> tuple[dict, dict[str, tuple[Path, ...]], dict[str, dict]]:
    """Join clustering artifacts into a browser-safe review payload."""

    paths = {
        "event_features": resolve_groot_artifact_path(
            event_features_path
        ).resolve(),
        "clusters": resolve_groot_artifact_path(clusters_path).resolve(),
        "assignments": resolve_groot_artifact_path(
            assignments_path
        ).resolve(),
        "annotations": resolve_groot_artifact_path(
            annotations_path
        ).resolve(),
    }
    if representative_media_clusters_path is not None:
        paths["representative_media_clusters"] = resolve_groot_artifact_path(
            representative_media_clusters_path
        ).resolve()
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    features = load_jsonl(paths["event_features"])
    clusters = load_jsonl(paths["clusters"])
    assignments = load_jsonl(paths["assignments"])
    annotations = load_jsonl(paths["annotations"])
    representative_media_clusters = (
        load_jsonl(paths["representative_media_clusters"])
        if "representative_media_clusters" in paths
        else None
    )

    bundle = _validate_cluster_bundle(
        clusters=clusters,
        assignments=assignments,
        features=features,
    )
    features_by_id = bundle.features_by_sample_id
    assert features_by_id is not None
    assignments_by_id = bundle.assignments_by_sample_id
    clusters_by_id = bundle.clusters_by_id
    annotations_by_id = {str(row["cluster_id"]): row for row in annotations}

    if len(annotations_by_id) != len(annotations):
        raise ValueError("Annotations contain duplicate cluster IDs")
    if not set(annotations_by_id).issubset(clusters_by_id):
        raise ValueError("Annotations reference unknown clusters")
    if any(row.get("api_error") or row.get("parse_error") for row in annotations):
        raise ValueError("Annotations contain API or parse errors")
    annotation_coverage_thresholds = {
        (
            None
            if row.get("annotation_min_episode_coverage") is None
            else float(row["annotation_min_episode_coverage"])
        )
        for row in annotations
    }
    if len(annotation_coverage_thresholds) != 1:
        raise ValueError(
            "Annotations must use one consistent minimum episode coverage"
        )
    annotation_min_episode_coverage = annotation_coverage_thresholds.pop()

    representative_ids = {
        str(sample_id)
        for cluster in clusters
        for sample_id in cluster["representative_sample_ids"]
    }
    base_media_paths: dict[str, tuple[Path, ...]] = {}
    task_records: dict[str, list[dict]] = defaultdict(list)
    for record in features:
        sample_id = str(record["sample_id"])
        frame_paths = tuple(
            resolve_groot_artifact_path(value).resolve()
            for value in record["frame_paths"]
        )
        if len(frame_paths) != 5:
            raise ValueError(
                f"Expected five frames for sample {sample_id}, got {len(frame_paths)}"
            )
        if validate_media:
            missing = [str(path) for path in frame_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing media for sample {sample_id}: {missing[0]}"
                )
        base_media_paths[sample_id] = frame_paths
        task_records[str(record["task_description"])].append(record)

    media_paths = base_media_paths
    if representative_media_clusters is not None:
        media_clusters_by_id = {
            str(row["cluster_id"]): row for row in representative_media_clusters
        }
        if len(media_clusters_by_id) != len(representative_media_clusters):
            raise ValueError(
                "Representative media clusters contain duplicate cluster IDs"
            )
        if set(media_clusters_by_id) != set(annotations_by_id):
            raise ValueError(
                "Representative media cluster IDs must exactly match annotation targets"
            )
        media_paths = {}
        for cluster_id, media_cluster in media_clusters_by_id.items():
            base_cluster = clusters_by_id[cluster_id]
            sample_ids = [
                str(value) for value in media_cluster["representative_sample_ids"]
            ]
            expected_ids = [
                str(value) for value in base_cluster["representative_sample_ids"]
            ]
            if sample_ids != expected_ids:
                raise ValueError(f"Representative sample mismatch: {cluster_id}")
            annotation_ids = [
                str(value)
                for value in annotations_by_id[cluster_id].get(
                    "representative_sample_ids", sample_ids
                )
            ]
            if sample_ids != annotation_ids:
                raise ValueError(f"Annotation representative mismatch: {cluster_id}")
            frame_groups = media_cluster.get("representative_frame_paths", [])
            if len(frame_groups) != len(sample_ids):
                raise ValueError(
                    f"Representative media count mismatch: {cluster_id}"
                )
            for sample_id, values in zip(sample_ids, frame_groups, strict=True):
                if sample_id not in features_by_id:
                    raise ValueError(f"Unknown representative sample: {sample_id}")
                frame_paths = tuple(
                    resolve_groot_artifact_path(value).resolve()
                    for value in values
                )
                if len(frame_paths) != 5:
                    raise ValueError(
                        f"Expected five representative frames for {sample_id}, "
                        f"got {len(frame_paths)}"
                    )
                if validate_media:
                    missing = [
                        str(path) for path in frame_paths if not path.is_file()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"Missing representative media for {sample_id}: "
                            f"{missing[0]}"
                        )
                if sample_id in media_paths:
                    raise ValueError(f"Duplicate representative media: {sample_id}")
                media_paths[sample_id] = frame_paths

    projection_by_id: dict[str, list[float]] = {}
    for task_description, records in sorted(task_records.items()):
        records.sort(
            key=lambda row: (
                int(row["episode_num"]),
                int(row["waypoint_step"]),
                int(row["waypoint_rank"]),
            )
        )
        coordinates = _project_task(
            records,
            method=projection_method,
            random_state=random_state,
            block_normalization=block_normalization,
        )
        for record, coordinate in zip(records, coordinates, strict=True):
            projection_by_id[str(record["sample_id"])] = [
                round(float(coordinate[0]), 6),
                round(float(coordinate[1]), 6),
            ]

    browser_id_by_sample = {
        sample_id: f"clip_{index:06d}"
        for index, sample_id in enumerate(sorted(features_by_id))
    }
    browser_media_paths = {
        browser_id_by_sample[sample_id]: paths
        for sample_id, paths in media_paths.items()
    }
    browser_samples: dict[str, dict] = {}
    for sample_id, record in features_by_id.items():
        browser_id = browser_id_by_sample[sample_id]
        assignment = assignments_by_id[sample_id]
        sample_media_paths = media_paths.get(sample_id, ())
        browser_samples[browser_id] = {
            "sample_id": browser_id,
            "task_description": str(record["task_description"]),
            "cluster_id": str(assignment["cluster_id"]),
            "projection": projection_by_id[sample_id],
            "num_frames": len(sample_media_paths),
            "media_available": bool(sample_media_paths),
            "is_representative": sample_id in representative_ids,
        }

    browser_clusters = []
    for cluster in sorted(
        clusters,
        key=lambda row: (str(row["task_description"]), str(row["cluster_id"])),
    ):
        cluster_id = str(cluster["cluster_id"])
        member_sample_ids = [str(value) for value in cluster["member_sample_ids"]]
        representative_sample_ids = [
            str(value) for value in cluster["representative_sample_ids"]
        ]
        representative_set = set(representative_sample_ids)
        ordered_sample_ids = representative_sample_ids + [
            sample_id
            for sample_id in member_sample_ids
            if sample_id not in representative_set
        ]
        playable_sample_ids = [
            browser_id_by_sample[sample_id]
            for sample_id in ordered_sample_ids
            if sample_id in media_paths
        ]
        annotation = annotations_by_id.get(cluster_id)
        review_protocol = None
        if annotation is not None:
            review_protocol = {
                "allowed_phase_labels": list(annotation["allowed_phase_labels"]),
                "phase_descriptions": _phase_descriptions(annotation),
            }
        browser_clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": int(cluster["cluster_label"]),
                "task_description": str(cluster["task_description"]),
                "num_members": int(cluster["num_members"]),
                "episode_coverage": float(cluster["episode_coverage"]),
                "total_task_episodes": int(cluster["total_task_episodes"]),
                "cluster_mean_progress_percent": float(
                    cluster["cluster_mean_progress_percent"]
                ),
                "member_sample_ids": [
                    browser_id_by_sample[sample_id]
                    for sample_id in member_sample_ids
                ],
                "representative_sample_ids": [
                    browser_id_by_sample[sample_id]
                    for sample_id in representative_sample_ids
                ],
                "playable_sample_ids": playable_sample_ids,
                "annotation_target": annotation is not None,
                "review_protocol": review_protocol,
            }
        )

    task_payload = []
    for task_description in sorted(task_records):
        task_cluster_rows = [
            cluster
            for cluster in browser_clusters
            if cluster["task_description"] == task_description
        ]
        task_payload.append(
            {
                "task_description": task_description,
                "num_samples": len(task_records[task_description]),
                "num_clusters": len(task_cluster_rows),
                "num_playable_samples": sum(
                    browser_samples[
                        browser_id_by_sample[str(row["sample_id"])]
                    ]["media_available"]
                    for row in task_records[task_description]
                ),
                "num_annotated_clusters": sum(
                    cluster["annotation_target"]
                    for cluster in task_cluster_rows
                ),
            }
        )

    condition = {
        "id": condition_id,
        "label": condition_label,
        **(condition_metadata or {}),
    }
    payload = {
        "format": "event_sae_stage3_human_review_payload_v3",
        "meta": {
            "version_id": condition_id,
            "version_label": condition_label,
            "condition": condition,
            "media_layout": media_layout,
            "media_scope": (
                "representative-only"
                if representative_media_clusters is not None
                else "all-samples"
            ),
            "projection_method": projection_method,
            "projection_scope": (
                f"task-local {block_normalization} C0 descriptor "
                "(vision=1.0, state=0.5, progress=0.4)"
            ),
            "projection_warning": (
                "The 2D view is an approximate projection and does not preserve every "
                "cosine distance used by clustering."
            ),
            "success_omitted": True,
            "blind_temporal_metadata_omitted": True,
            "review_mode": "blind_two_stage",
            "num_samples": len(browser_samples),
            "num_clusters": len(browser_clusters),
            "num_playable_samples": len(media_paths),
            "num_annotated_clusters": len(annotations_by_id),
            "annotation_min_episode_coverage": (
                annotation_min_episode_coverage
            ),
            "source_sha256": {
                label: _sha256(path) for label, path in paths.items()
            },
        },
        "tasks": task_payload,
        "clusters": browser_clusters,
        "samples": browser_samples,
    }
    return payload, browser_media_paths, annotations_by_id


class ClusterReviewStore:
    """Atomic persistence for blind assessment and model adjudication."""

    def __init__(
        self,
        path: Path,
        *,
        annotations_by_id: dict[str, dict],
        annotations_path: Path,
    ) -> None:
        self.path = Path(path).resolve()
        self.annotations_by_id = annotations_by_id
        self.annotations_path = Path(annotations_path).resolve()
        self.annotations_sha256 = _sha256(self.annotations_path)
        self._lock = threading.Lock()
        self._reviews: dict[str, dict] = {}
        self._updated_at = datetime.now(timezone.utc).isoformat()
        if self.path.exists():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            document_format = document.get("format")
            if document_format != REVIEW_DOCUMENT_FORMAT:
                raise ValueError(
                    f"Unsupported review document format: {document_format}"
                )
            if document.get("annotations_sha256") != self.annotations_sha256:
                raise ValueError(
                    "Existing reviews were created for a different annotation file"
                )
            review_rows = list(document.get("reviews", []))
            review_ids = [str(row["cluster_id"]) for row in review_rows]
            if len(review_ids) != len(set(review_ids)):
                raise ValueError("Existing reviews contain duplicate cluster IDs")
            unknown_ids = set(review_ids) - set(self.annotations_by_id)
            if unknown_ids:
                first_unknown_id = sorted(unknown_ids)[0]
                raise ValueError(
                    f"Existing reviews reference unknown cluster: {first_unknown_id}"
                )
            self._reviews = {
                cluster_id: self._validate_loaded_review(row)
                for cluster_id, row in zip(review_ids, review_rows, strict=True)
            }
            loaded_updated_at = document.get("updated_at")
            if loaded_updated_at is not None:
                self._updated_at = str(loaded_updated_at)

    @staticmethod
    def _validate_loaded_review(row: dict) -> dict:
        review = dict(row)
        if review.get("review_stage") not in {
            "blind_recorded",
            "adjudicated",
        }:
            raise ValueError(
                "Existing review has an invalid review_stage: "
                f"{review.get('review_stage')}"
            )
        return review

    def _document(
        self,
        *,
        reviews: dict[str, dict] | None = None,
        updated_at: str | None = None,
    ) -> dict:
        active_reviews = self._reviews if reviews is None else reviews
        return {
            "format": REVIEW_DOCUMENT_FORMAT,
            "annotations_path": str(self.annotations_path),
            "annotations_sha256": self.annotations_sha256,
            "review_mode": "blind_two_stage",
            "updated_at": self._updated_at if updated_at is None else updated_at,
            "reviews": [
                active_reviews[cluster_id] for cluster_id in sorted(active_reviews)
            ],
        }

    def document(self) -> dict:
        with self._lock:
            return self._document()

    def reveal_annotation(self, cluster_id: str) -> dict:
        """Return a model annotation only after blind assessment was persisted."""

        with self._lock:
            if cluster_id not in self.annotations_by_id:
                raise KeyError(cluster_id)
            review = self._reviews.get(cluster_id)
            if review is None:
                raise PermissionError(
                    "Lock the independent blind assessment before revealing Gemini"
                )
            annotation = self.annotations_by_id[cluster_id]
            return {
                "cluster_id": cluster_id,
                "phrase": str(annotation["phrase"]),
                "phase": str(annotation["phase"]),
                "model": str(annotation["model"]),
                "prompt_version": str(annotation["prompt_version"]),
                "allowed_phase_labels": list(
                    annotation["allowed_phase_labels"]
                ),
            }

    def revealed_annotations(self) -> dict[str, dict]:
        with self._lock:
            cluster_ids = tuple(self._reviews)
        return {
            cluster_id: self.reveal_annotation(cluster_id)
            for cluster_id in cluster_ids
        }

    @staticmethod
    def _request_bool(
        request: dict,
        key: str,
        *,
        default: bool | None = None,
    ) -> bool:
        value = request.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a JSON boolean")
        return value

    def _persist(self, record: dict) -> dict:
        cluster_id = str(record["cluster_id"])
        saved_at = str(record.get("reviewed_at") or record["blind_reviewed_at"])
        next_reviews = dict(self._reviews)
        next_reviews[cluster_id] = record
        next_document = self._document(
            reviews=next_reviews,
            updated_at=saved_at,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(
                    json.dumps(next_document, indent=2, ensure_ascii=False)
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        self._reviews = next_reviews
        self._updated_at = saved_at
        return record

    def _save_blind(self, request: dict) -> dict:
        cluster_id = str(request.get("cluster_id", "")).strip()
        reviewer = str(request.get("reviewer", "")).strip() or "human"
        notes = str(request.get("notes", "")).strip()
        if cluster_id not in self.annotations_by_id:
            raise ValueError(f"Cluster is not an annotation target: {cluster_id}")
        if len(reviewer) > 200 or len(notes) > 10000:
            raise ValueError("Reviewer or notes field is too long")

        annotation = self.annotations_by_id[cluster_id]
        human_phrase_value = str(request.get("human_phrase", "")).strip()
        human_phase_value = str(request.get("human_phase", "")).strip()
        mixed_cluster = self._request_bool(
            request,
            "mixed_cluster",
            default=False,
        )
        visually_insufficient = self._request_bool(
            request,
            "visually_insufficient",
            default=False,
        )
        unresolved = mixed_cluster or visually_insufficient
        if not unresolved and not human_phrase_value:
            raise ValueError("Blind assessment requires a non-empty human phrase")
        if not unresolved and not human_phase_value:
            raise ValueError("Blind assessment requires a human phase")
        if (
            human_phase_value
            and human_phase_value not in set(annotation["allowed_phase_labels"])
        ):
            raise ValueError("Human phase is outside the allowed vocabulary")
        if len(human_phrase_value) > 1000:
            raise ValueError("Human phrase is too long")

        with self._lock:
            if cluster_id in self._reviews:
                raise ValueError(
                    "Independent blind assessment is already locked for this cluster"
                )
            saved_at = datetime.now(timezone.utc).isoformat()
            record = {
                "cluster_id": cluster_id,
                "task_description": str(annotation["task_description"]),
                "review_stage": "blind_recorded",
                "reviewer": reviewer,
                "human_phrase": human_phrase_value or None,
                "human_phase": human_phase_value or None,
                "mixed_cluster": mixed_cluster,
                "visually_insufficient": visually_insufficient,
                "notes": notes,
                "blind_reviewed_at": saved_at,
                "phrase_phase_consistent": None,
                "phase_corrected": None,
                "verdict": None,
                "reviewed_at": None,
            }
            return self._persist(record)

    def _save_adjudication(self, request: dict) -> dict:
        cluster_id = str(request.get("cluster_id", "")).strip()
        adjudication_notes = str(
            request.get("adjudication_notes", "")
        ).strip()
        if len(adjudication_notes) > 10000:
            raise ValueError("Adjudication notes field is too long")
        phrase_phase_consistent = self._request_bool(
            request,
            "phrase_phase_consistent",
        )
        with self._lock:
            existing = self._reviews.get(cluster_id)
            if existing is None:
                raise ValueError(
                    "Blind assessment must be locked before adjudication"
                )
            annotation = self.annotations_by_id[cluster_id]
            human_phase = existing.get("human_phase")
            phase_corrected = (
                None
                if human_phase is None
                else str(human_phase) != str(annotation["phase"])
            )
            if existing["mixed_cluster"] or existing["visually_insufficient"]:
                verdict = "ambiguous"
            elif phase_corrected or not phrase_phase_consistent:
                verdict = "corrected"
            else:
                verdict = "approved"
            saved_at = datetime.now(timezone.utc).isoformat()
            record = {
                **existing,
                "review_stage": "adjudicated",
                "model_phrase": str(annotation["phrase"]),
                "model_phase": str(annotation["phase"]),
                "phrase_phase_consistent": phrase_phase_consistent,
                "phase_corrected": phase_corrected,
                "adjudication_notes": adjudication_notes,
                "verdict": verdict,
                "reviewed_at": saved_at,
            }
            return self._persist(record)

    def save(self, request: dict) -> dict:
        stage = str(request.get("stage", "")).strip()
        if stage not in REVIEW_STAGES:
            raise ValueError(f"Invalid review stage: {stage}")
        if stage == "blind":
            return self._save_blind(request)
        return self._save_adjudication(request)


class ClusterReviewService:
    def __init__(
        self,
        *,
        payload: dict,
        media_paths: dict[str, tuple[Path, ...]],
        review_store: ClusterReviewStore,
        ui_path: Path,
    ) -> None:
        self.payload = payload
        self.media_paths = media_paths
        self.review_store = review_store
        self.ui = Path(ui_path).read_bytes()

    def data(self) -> dict:
        return {
            **self.payload,
            "review_document": self.review_store.document(),
            "revealed_annotations": self.review_store.revealed_annotations(),
        }

    def reviews(self) -> dict:
        return self.review_store.document()


def build_cluster_review_service(
    *,
    event_features_path: Path,
    clusters_path: Path,
    assignments_path: Path,
    annotations_path: Path,
    reviews_path: Path,
    media_clusters_path: Path | None,
    ui_path: Path,
    projection_method: str,
    condition_id: str,
    condition_label: str,
    media_layout: str,
    block_normalization: str = "balanced",
    condition_metadata: dict[str, str] | None = None,
) -> ClusterReviewService:
    """Build one condition-scoped blind review application."""

    payload, media_paths, annotations = build_blind_cluster_review_dataset(
        event_features_path=event_features_path,
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        annotations_path=annotations_path,
        representative_media_clusters_path=media_clusters_path,
        condition_id=condition_id,
        condition_label=condition_label,
        media_layout=media_layout,
        condition_metadata=condition_metadata,
        projection_method=projection_method,
        block_normalization=block_normalization,
    )
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id=annotations,
        annotations_path=annotations_path,
    )
    return ClusterReviewService(
        payload=payload,
        media_paths=media_paths,
        review_store=store,
        ui_path=ui_path,
    )


def make_cluster_review_http_handler(
    application: ClusterReviewService,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EventSAEReview/3.0"

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                "connect-src 'self'",
            )
            self.end_headers()

        def _send_bytes(
            self,
            data: bytes,
            *,
            status: int = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
        ) -> None:
            self._headers(int(status), content_type, len(data))
            self.wfile.write(data)

        def _send_json(self, value: dict, *, status: int = HTTPStatus.OK) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send_bytes(
                data,
                status=status,
                content_type="application/json; charset=utf-8",
            )

        def _error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(
                    application.ui,
                    content_type="text/html; charset=utf-8",
                )
                return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/data":
                self._send_json(application.data())
                return
            if parsed.path == "/api/reviews":
                self._send_json(application.reviews())
                return
            if parsed.path == "/api/annotation":
                cluster_id = query.get("cluster_id", [""])[0]
                try:
                    annotation = application.review_store.reveal_annotation(
                        cluster_id
                    )
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, "Unknown cluster")
                    return
                except PermissionError as exc:
                    self._error(HTTPStatus.FORBIDDEN, str(exc))
                    return
                self._send_json({"annotation": annotation})
                return
            if parsed.path == "/api/frame":
                sample_id = query.get("sample_id", [""])[0]
                try:
                    frame_index = int(query.get("index", ["-1"])[0])
                except ValueError:
                    self._error(HTTPStatus.BAD_REQUEST, "Invalid frame index")
                    return
                paths = application.media_paths.get(sample_id)
                if paths is None or not 0 <= frame_index < len(paths):
                    self._error(HTTPStatus.NOT_FOUND, "Unknown sample or frame")
                    return
                frame_path = paths[frame_index]
                data = frame_path.read_bytes()
                content_type = (
                    mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
                )
                self._send_bytes(data, content_type=content_type)
                return
            if parsed.path == "/healthz":
                self._send_json(
                    {
                        "status": "ok",
                        "version": application.payload["meta"]["version_id"],
                    }
                )
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/review":
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid content length")
                return
            if not 0 < length <= 65536:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid request size")
                return
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("JSON body must be an object")
                record = application.review_store.save(request)
            except (json.JSONDecodeError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            annotation = application.review_store.reveal_annotation(
                str(record["cluster_id"])
            )
            self._send_json(
                {
                    "review": record,
                    "annotation": annotation,
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            if self.path != "/healthz":
                super().log_message(format, *args)

    return Handler


__all__ = [
    "ClusterReviewService",
    "ClusterReviewStore",
    "build_blind_cluster_review_dataset",
    "build_cluster_review_service",
    "finalize_reviewed_annotations",
    "make_cluster_review_http_handler",
    "render_contact_sheets",
]
