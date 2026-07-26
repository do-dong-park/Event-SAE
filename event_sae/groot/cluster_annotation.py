"""Centroid-five cluster annotation, consensus, and artifact materialization.

This module derives a content-addressed experiment from centroid-nearest media
catalogs and requires five fresh responses for each condition/cluster pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from google.genai import types

from event_sae import sha256_file
from event_sae.events.representative_annotation import (
    AnnotationImagePart,
    REQUEST_EXPOSURE_POLICY,
    RESPONSE_MIME_TYPE,
    RESPONSE_SCHEMA_ID,
    SINGLE_VIEW_LEFT_LAYOUT,
    SEPARATE_MULTIVIEW_LAYOUT,
    build_representative_request_parts,
    build_representative_response_schema,
    hash_annotation_media,
    hash_canonical_record,
    hash_representative_set,
    parse_representative_response,
    redact_sensitive_error,
    resolve_annotation_media_layout,
    serialize_usage_metadata,
    validate_representative_annotation,
    validate_representative_media,
)
from event_sae.events.cluster import build_task_vectors
from event_sae.events.io import load_jsonl
from event_sae.events.prompts import (
    ROBOCASA_PHASE_LABELER_PROVENANCE,
    ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
    build_representative_clip_annotation_prompt,
    resolve_representative_phase_vocabulary,
    validate_clean_visual_annotation_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT
    / "logs/groot_n15/experiments/anchor_view_controlled_ablation_v1"
)
CENTROID_DIVERSITY_REPRESENTATIVE_ROOT = (
    EXPERIMENT_ROOT / "v12r1/representatives"
)
LEGACY_MEDIA_ROOT = EXPERIMENT_ROOT / "annotation_media"
FROZEN_INTERACTIVE_ROOT = EXPERIMENT_ROOT / "annotations_v12r3"
HISTORICAL_ANNOTATION_ROOTS = (
    EXPERIMENT_ROOT / "annotations_batch_generate_content_v1",
    EXPERIMENT_ROOT / "annotations_batch_singleton_normalized_v2",
    EXPERIMENT_ROOT / "annotations_batch_centroid_nearest_five",
    EXPERIMENT_ROOT / "annotations_batch_centroid_nearest_five_majority_3_of_5",
    EXPERIMENT_ROOT / "annotations_batch_centroid_nearest_five_unique_plurality",
)
DEFAULT_OUTPUT_ROOT = (
    EXPERIMENT_ROOT / "annotations_batch_centroid_nearest_five_v2"
)
ADAPTIVE_OUTPUT_ROOT = (
    EXPERIMENT_ROOT / "annotations_v12_adaptive_plurality"
)
CONTINUATION_OUTPUT_ROOT = (
    EXPERIMENT_ROOT / "annotations_v12_adaptive_plurality_rank17"
)
SOURCE_CENTROID_ROOT = (
    EXPERIMENT_ROOT / "annotations_batch_centroid_nearest_five"
)
SOURCE_PLURALITY_ROOT = (
    EXPERIMENT_ROOT
    / "annotations_batch_centroid_nearest_five_unique_plurality"
)
SOURCE_RUN_CONTRACT_SHA256 = (
    "2a203be108f3568c98b654de9245957582ffa530e0f605b00a2b2279d13c8e2b"
)
CONTINUATION_SOURCE_RUN_CONTRACT_SHA256 = (
    "20cbde0fcb3301ddb09b064e9db6566f26a19eaeada389b98d98ce23c2803b5b"
)

RUN_FORMAT = "event_sae_centroid_nearest_five_batch_run_v2"
SUMMARY_FORMAT = "event_sae_centroid_nearest_five_summary_v2"
OUTPUT_FORMAT = "event_sae_centroid_nearest_five_annotation_v2"
ADAPTIVE_RUN_FORMAT = "event_sae_v12_adaptive_plurality_run_v1"
ADAPTIVE_SUMMARY_FORMAT = "event_sae_v12_adaptive_plurality_summary_v1"
ADAPTIVE_OUTPUT_FORMAT = "event_sae_v12_adaptive_plurality_annotation_v1"
CONTINUATION_RUN_FORMAT = (
    "event_sae_v12_centroid_plurality_continuation_run_v1"
)
CONTINUATION_SUMMARY_FORMAT = (
    "event_sae_v12_adaptive_plurality_rank17_summary_v1"
)
CONTINUATION_OUTPUT_FORMAT = (
    "event_sae_v12_adaptive_plurality_rank17_annotation_v1"
)
ATTEMPT_FORMAT = "event_sae_cluster_annotation_attempt_v2"
SELECTION_STRATEGY = "centroid_nearest_five_unique_episode"
SELECTION_ROLES = ("centroid",) * 5
REPRESENTATIVE_INDICES = (1, 2, 3, 4, 5)
ADAPTIVE_SELECTION_STRATEGY = (
    "centroid_nearest_nine_unique_episode"
)
ADAPTIVE_REPRESENTATIVE_INDICES = tuple(range(1, 10))
ADAPTIVE_REQUEST_INDICES = tuple(range(6, 10))
ADAPTIVE_CHECKPOINTS = ((6, 7), (8, 9))
CONTINUATION_SELECTION_STRATEGY = (
    "centroid_nearest_seventeen_unique_episode"
)
CONTINUATION_REPRESENTATIVE_INDICES = tuple(range(1, 18))
CONTINUATION_SOURCE_INDICES = tuple(range(1, 10))
CONTINUATION_REQUEST_INDICES = tuple(range(10, 18))
CONTINUATION_CHECKPOINTS = (
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
)
CONTINUATION_TARGET_CONDITION_ID = (
    "e3_rel_pos_gripper_cluster_multiview_label_multiview"
)
CONTINUATION_TARGET_CLUSTER_ID = (
    "pick_the_beer_from_the_counter_and_place_it_in_the_cabinet_cluster_10"
)
MIN_EPISODE_COVERAGE = 0.3
CONSENSUS_VOTES = 4
MODEL = "gemini-3.1-pro-preview"
EXPECTED_MODEL_VERSION = "gemini-3.1-pro-preview"
TEMPERATURE = 0.0
MAX_REQUEST_ATTEMPTS = 3
MAX_SERIALIZED_CHUNK_BYTES = 15_000_000
NORMALIZATION_POLICY_ID = "object_or_singleton_object_list_v1"

ADAPTER_FILES = (
    "event_sae/groot/cluster_annotation.py",
    "event_sae/groot/gemini_batch.py",
    "scripts/groot/annotate_clusters.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256_file(Path(path))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _fsync_directory(directory: Path) -> None:
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(_jsonl_bytes(rows).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _package_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in ("google-genai", "Pillow")
    }


@dataclass(frozen=True)
class AdaptiveExtensionConfig:
    source_rank: int
    maximum_rank: int
    checkpoints: tuple[tuple[int, ...], ...]
    selection_strategy: str
    source_run_contract_sha256: str
    source_provider_response_count: int
    output_format: str
    summary_format: str
    audit_format: str
    source_annotation_field: str
    extension_field: str

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(range(1, self.source_rank + 1))

    @property
    def available_indices(self) -> tuple[int, ...]:
        return tuple(range(1, self.maximum_rank + 1))

    @property
    def fresh_indices(self) -> tuple[int, ...]:
        return tuple(range(self.source_rank + 1, self.maximum_rank + 1))


@dataclass(frozen=True)
class ConditionContext:
    experiment_id: str
    condition_id: str
    clusters_path: Path
    output_path: Path
    annotation_view: str
    media_layout: str
    view_order: tuple[str, ...]
    source_rows: list[dict[str, Any]]
    source_clusters: dict[str, dict[str, Any]]
    selection_contract: dict[str, Any]
    targeted_rows: list[dict[str, Any]]
    target_ids: list[str]
    base_annotations: dict[str, dict[str, Any]] | None = None
    adaptive_config: AdaptiveExtensionConfig | None = None


@dataclass(frozen=True)
class NormalizedResponse:
    source_text: str
    normalized_text: str
    source_shape: str
    parsed: Mapping[str, Any]
    source_text_sha256: str
    normalized_text_sha256: str
    changed: bool


def normalize_response_text(
    raw_text: str,
    *,
    task_description: str,
) -> NormalizedResponse:
    """Accept an object or unwrap exactly one singleton object list."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Provider response text must be non-empty")
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"json_decode_error: {exc}") from exc
    if isinstance(decoded, dict):
        source_shape = "object"
        normalized_text = raw_text
        changed = False
    elif (
        isinstance(decoded, list)
        and len(decoded) == 1
        and isinstance(decoded[0], dict)
    ):
        source_shape = "singleton_object_list"
        normalized_text = json.dumps(
            decoded[0],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        changed = True
    else:
        raise ValueError(
            "unsupported_top_level_json_shape:"
            f"{type(decoded).__name__}"
        )
    parsed, parse_error = parse_representative_response(
        normalized_text,
        task_description=task_description,
    )
    if parsed is None or parse_error is not None:
        raise ValueError(
            f"normalized_response_failed_parser:{parse_error}"
        )
    return NormalizedResponse(
        source_text=raw_text,
        normalized_text=normalized_text,
        source_shape=source_shape,
        parsed=dict(parsed),
        source_text_sha256=sha256_bytes(raw_text.encode("utf-8")),
        normalized_text_sha256=sha256_bytes(
            normalized_text.encode("utf-8")
        ),
        changed=changed,
    )


def normalization_provenance(
    response: NormalizedResponse,
) -> dict[str, Any]:
    return {
        "policy_id": NORMALIZATION_POLICY_ID,
        "source_shape": response.source_shape,
        "source_text_sha256": response.source_text_sha256,
        "normalized_text_sha256": response.normalized_text_sha256,
        "changed": response.changed,
    }


def annotation_request_id(
    condition_id: str,
    cluster_id: str,
    representative_index: int,
) -> str:
    return (
        f"{condition_id}|{cluster_id}|"
        f"representative-{representative_index}"
    )


def _logical_hash(logical_id: str) -> str:
    return sha256_bytes(logical_id.encode("utf-8"))


def _request_key(logical_id: str, attempt: int) -> str:
    return f"req-{_logical_hash(logical_id)[:24]}-a{attempt}"


def serialize_inlined_request(request: types.InlinedRequest) -> dict[str, Any]:
    return request.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


def serialize_batch_payload(
    requests: Sequence[types.InlinedRequest],
) -> bytes:
    return _canonical_json_bytes(
        {
            "inlined_requests": [
                serialize_inlined_request(request)
                for request in requests
            ]
        }
    )


def _attempt_directory(output_root: Path, logical_id: str) -> Path:
    return (
        Path(output_root).resolve()
        / "responses"
        / _logical_hash(logical_id)
    )


def _attempt_path(
    output_root: Path,
    logical_id: str,
    attempt: int,
) -> Path:
    return (
        _attempt_directory(output_root, logical_id)
        / f"attempt-{attempt:03d}.json"
    )


def _load_attempts(
    output_root: Path,
    logical_id: str,
) -> list[dict[str, Any]]:
    directory = _attempt_directory(output_root, logical_id)
    paths = (
        sorted(directory.glob("attempt-*.json"))
        if directory.exists()
        else []
    )
    records = [load_json_object(path) for path in paths]
    if any(record.get("format") != ATTEMPT_FORMAT for record in records):
        raise ValueError(f"Invalid response attempt format: {logical_id}")
    attempts = [int(record["attempt"]) for record in records]
    if attempts != list(range(1, len(records) + 1)):
        raise ValueError(
            f"Non-contiguous response attempts for {logical_id}: {attempts}"
        )
    successes = [
        record for record in records
        if record.get("status") == "success"
    ]
    if len(successes) > 1 or (
        successes and successes[0] is not records[-1]
    ):
        raise ValueError(f"Invalid success sequence for {logical_id}")
    return records


def _successful_attempt(
    output_root: Path,
    logical_id: str,
) -> dict[str, Any] | None:
    return next(
        (
            record
            for record in _load_attempts(output_root, logical_id)
            if record.get("status") == "success"
        ),
        None,
    )


def _write_attempt(
    path: Path,
    record: Mapping[str, Any],
) -> None:
    if path.exists():
        existing = load_json_object(path)
        comparable_existing = dict(existing)
        comparable_record = dict(record)
        comparable_existing.pop("collected_at_utc", None)
        comparable_record.pop("collected_at_utc", None)
        if comparable_existing != comparable_record:
            raise ValueError(f"Response attempt drifted at {path}")
        return
    write_json_exclusive(path, record)


@dataclass(frozen=True)
class ConditionSpec:
    experiment_id: str
    condition_id: str
    partition: str
    media_layout: str


CONDITIONS = (
    ConditionSpec(
        experiment_id="E0",
        condition_id="e0_rel_pos_cluster_left_label_left",
        partition="p0",
        media_layout=SINGLE_VIEW_LEFT_LAYOUT,
    ),
    ConditionSpec(
        experiment_id="E1",
        condition_id="e1_rel_pos_cluster_left_label_multiview",
        partition="p0",
        media_layout=SEPARATE_MULTIVIEW_LAYOUT,
    ),
    ConditionSpec(
        experiment_id="E2",
        condition_id="e2_rel_pos_cluster_multiview_label_multiview",
        partition="p1",
        media_layout=SEPARATE_MULTIVIEW_LAYOUT,
    ),
    ConditionSpec(
        experiment_id="E3",
        condition_id=(
            "e3_rel_pos_gripper_cluster_multiview_label_multiview"
        ),
        partition="p2",
        media_layout=SEPARATE_MULTIVIEW_LAYOUT,
    ),
    ConditionSpec(
        experiment_id="E4",
        condition_id=(
            "e4_abs_pos_gripper_cluster_multiview_label_multiview"
        ),
        partition="p3",
        media_layout=SEPARATE_MULTIVIEW_LAYOUT,
    ),
)


def _validate_output_root(output_root: Path) -> Path:
    output_root = Path(output_root).resolve()
    temporary_root = Path("/tmp").resolve()
    if (
        output_root
        not in {
            DEFAULT_OUTPUT_ROOT.resolve(),
            ADAPTIVE_OUTPUT_ROOT.resolve(),
            CONTINUATION_OUTPUT_ROOT.resolve(),
        }
        and temporary_root not in output_root.parents
    ):
        raise ValueError(
            "Cluster annotation output must use a designated experiment "
            f"root or a temporary test root: {output_root}"
        )
    protected = (
        FROZEN_INTERACTIVE_ROOT.resolve(),
        *(root.resolve() for root in HISTORICAL_ANNOTATION_ROOTS),
        LEGACY_MEDIA_ROOT.resolve(),
        CENTROID_DIVERSITY_REPRESENTATIVE_ROOT.resolve(),
    )
    if output_root != ADAPTIVE_OUTPUT_ROOT.resolve():
        protected = (*protected, ADAPTIVE_OUTPUT_ROOT.resolve())
    for root in protected:
        if (
            output_root == root
            or output_root in root.parents
            or root in output_root.parents
        ):
            raise ValueError(
                "Centroid annotation output must be disjoint from protected "
                f"artifacts: output={output_root}, protected={root}"
            )
    return output_root


@contextmanager
def exclusive_run_lock(output_root: Path) -> Iterator[None]:
    """Lock only after the centroid-aware output policy accepts the root."""

    output_root = _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another centroid annotation writer holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stable_record_key(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(record["episode_num"]),
        int(record.get("waypoint_step", -1)),
        int(record.get("waypoint_rank", -1)),
        str(record["sample_id"]),
    )


def _normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 1e-12, norms, 1.0)


def select_centroid_representatives(
    member_records: Sequence[Mapping[str, Any]],
    member_vectors: np.ndarray,
    *,
    count: int = 5,
) -> tuple[list[Mapping[str, Any]], list[float], list[int]]:
    """Select stable centroid-nearest representatives from unique episodes."""

    records = list(member_records)
    vectors = np.asarray(member_vectors, dtype=np.float64)
    if count < 1:
        raise ValueError("count must be positive")
    if not records:
        raise ValueError("member_records must not be empty")
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError(
            "member_vectors must be a 2D matrix aligned with member_records"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("member_vectors must contain only finite values")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("member sample IDs must be unique")
    if len({int(record["episode_num"]) for record in records}) < count:
        raise ValueError(
            f"centroid selection requires at least {count} unique episodes"
        )

    normalized = _normalized_rows(vectors)
    centroid = _normalized_rows(normalized.mean(axis=0, keepdims=True))
    distances = (
        1.0
        - np.clip(normalized @ centroid.T, -1.0, 1.0).reshape(-1)
    )
    ordered_indices = sorted(
        range(len(records)),
        key=lambda index: (
            float(distances[index]),
            *_stable_record_key(records[index]),
        ),
    )
    raw_rank = {
        index: rank
        for rank, index in enumerate(ordered_indices, start=1)
    }
    selected_indices: list[int] = []
    selected_episodes: set[int] = set()
    for index in ordered_indices:
        episode_num = int(records[index]["episode_num"])
        if episode_num in selected_episodes:
            continue
        selected_indices.append(index)
        selected_episodes.add(episode_num)
        if len(selected_indices) == count:
            break
    if len(selected_indices) != count:
        raise RuntimeError("centroid selection could not satisfy uniqueness")
    return (
        [records[index] for index in selected_indices],
        [float(distances[index]) for index in selected_indices],
        [raw_rank[index] for index in selected_indices],
    )


def _vector_index(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    selection_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in feature_rows:
        by_task[str(record["task_description"])].append(record)
    vector_by_id: dict[str, np.ndarray] = {}
    for task_records in by_task.values():
        ordered = sorted(task_records, key=_stable_record_key)
        vectors = build_task_vectors(
            list(ordered),
            vision_weight=float(selection_config["vision_weight"]),
            state_weight=float(selection_config["state_weight"]),
            progress_weight=float(selection_config["progress_weight"]),
            block_normalization=str(
                selection_config["block_normalization"]
            ),
        )
        for record, vector in zip(ordered, vectors, strict=True):
            vector_by_id[str(record["sample_id"])] = vector
    return vector_by_id


def _media_catalog_paths(partition: str) -> tuple[Path, Path]:
    directory = LEGACY_MEDIA_ROOT / f"{partition}_multiview"
    return directory / "clusters_multiview.jsonl", directory / "manifest.json"


def _load_partition_rows(
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    representative_manifest_path = (
        CENTROID_DIVERSITY_REPRESENTATIVE_ROOT
        / partition
        / "manifest.json"
    ).resolve()
    representative_manifest = load_json_object(
        representative_manifest_path
    )
    source_clusters_path = Path(
        str(representative_manifest["source_clusters_path"])
    ).resolve()
    event_features_path = Path(
        str(representative_manifest["event_features_path"])
    ).resolve()
    centroid_diversity_catalog_path = Path(
        str(representative_manifest["output_clusters_path"])
    ).resolve()
    for path, expected in (
        (
            source_clusters_path,
            representative_manifest["source_clusters_sha256"],
        ),
        (
            event_features_path,
            representative_manifest["event_features_sha256"],
        ),
        (
            centroid_diversity_catalog_path,
            representative_manifest["output_clusters_sha256"],
        ),
    ):
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{partition}: representative input drifted: {path}"
            )

    selection_config = dict(representative_manifest["selection_config"])
    expected_config = {
        "min_episode_coverage": MIN_EPISODE_COVERAGE,
        "vision_weight": 1.0,
        "state_weight": 0.5,
        "progress_weight": 0.4,
        "block_normalization": "balanced",
        "require_legacy_centroid_prefix": True,
        "unique_episode_policy": "error",
    }
    if selection_config != expected_config:
        raise ValueError(
            f"{partition}: source selection config drifted: "
            f"{selection_config}"
        )

    media_catalog_path, media_manifest_path = _media_catalog_paths(partition)
    media_catalog_path = media_catalog_path.resolve()
    media_manifest_path = media_manifest_path.resolve()
    media_manifest = load_json_object(media_manifest_path)
    if Path(str(media_manifest["output_clusters_path"])).resolve() != (
        media_catalog_path
    ):
        raise ValueError(f"{partition}: media manifest path drifted")
    if _sha256_file(media_catalog_path) != media_manifest[
        "output_clusters_sha256"
    ]:
        raise ValueError(f"{partition}: media catalog hash drifted")

    source_clusters = load_jsonl(source_clusters_path)
    eligible_clusters = [
        cluster
        for cluster in source_clusters
        if float(cluster["episode_coverage"]) >= MIN_EPISODE_COVERAGE
    ]
    feature_rows = load_jsonl(event_features_path)
    feature_by_id = {
        str(record["sample_id"]): record
        for record in feature_rows
    }
    if not feature_rows or len(feature_by_id) != len(feature_rows):
        raise ValueError(f"{partition}: features must be non-empty and unique")
    vector_by_id = _vector_index(
        feature_rows,
        selection_config=selection_config,
    )

    media_rows = load_jsonl(media_catalog_path)
    media_by_cluster = {
        str(row["cluster_id"]): row for row in media_rows
    }
    centroid_diversity_rows = load_jsonl(centroid_diversity_catalog_path)
    centroid_diversity_by_cluster = {
        str(row["cluster_id"]): row for row in centroid_diversity_rows
    }
    if len(media_by_cluster) != len(media_rows):
        raise ValueError(f"{partition}: duplicate media cluster IDs")
    if len(centroid_diversity_by_cluster) != len(centroid_diversity_rows):
        raise ValueError(
            f"{partition}: duplicate centroid/diversity cluster IDs"
        )

    derived_rows: list[dict[str, Any]] = []
    for source_cluster in eligible_clusters:
        cluster_id = str(source_cluster["cluster_id"])
        try:
            media_row = media_by_cluster[cluster_id]
            centroid_diversity_row = centroid_diversity_by_cluster[cluster_id]
            member_ids = [
                str(value)
                for value in source_cluster["member_sample_ids"]
            ]
            member_records = [feature_by_id[value] for value in member_ids]
            member_vectors = np.stack(
                [vector_by_id[value] for value in member_ids]
            )
        except KeyError as exc:
            raise ValueError(
                f"{partition}/{cluster_id}: missing joined record {exc}"
            ) from exc
        (
            representatives,
            centroid_distances,
            raw_ranks,
        ) = select_centroid_representatives(
            member_records,
            member_vectors,
        )
        selected_ids = [
            str(record["sample_id"]) for record in representatives
        ]
        media_ids = [
            str(value) for value in media_row["representative_sample_ids"]
        ]
        if media_ids != selected_ids:
            raise ValueError(
                f"{partition}/{cluster_id}: legacy centroid media does not "
                f"match recomputation: media={media_ids}, selected={selected_ids}"
            )
        current_prefix = [
            str(value)
            for value in centroid_diversity_row[
                "representative_sample_ids"
            ][:3]
        ]
        if current_prefix != selected_ids[:3]:
            raise ValueError(
                f"{partition}/{cluster_id}: current centroid prefix drifted"
            )
        if [
            str(value) for value in media_row["representative_clip_paths"]
        ] != [str(record["clip_path"]) for record in representatives]:
            raise ValueError(
                f"{partition}/{cluster_id}: representative clips drifted"
            )

        excluded_fields = {
            "annotation_media_layout",
            "annotation_triptych_size",
            "annotation_view_names",
            "representative_frame_paths",
            "representative_maximin_cosine_scores",
            "representative_selection_roles",
            "representative_selection_strategy",
        }
        derived = {
            key: value
            for key, value in media_row.items()
            if key not in excluded_fields
        }
        derived.update(
            {
                "representative_sample_ids": selected_ids,
                "representative_episode_nums": [
                    int(record["episode_num"])
                    for record in representatives
                ],
                "representative_clip_paths": [
                    str(record["clip_path"])
                    for record in representatives
                ],
                "representative_waypoint_steps": [
                    int(record["waypoint_step"])
                    for record in representatives
                ],
                "representative_progress_percents": [
                    float(record["progress_percent"])
                    for record in representatives
                ],
                "representative_selection_strategy": SELECTION_STRATEGY,
                "representative_selection_roles": list(SELECTION_ROLES),
                "representative_centroid_cosine_distances": (
                    centroid_distances
                ),
                "representative_centroid_raw_distance_ranks": raw_ranks,
                "representative_unique_episode_policy": "error",
                "representative_tie_break_fields": [
                    "episode_num",
                    "waypoint_step",
                    "waypoint_rank",
                    "sample_id",
                ],
            }
        )
        if len(set(derived["representative_episode_nums"])) != 5:
            raise ValueError(
                f"{partition}/{cluster_id}: episodes are not unique"
            )
        validate_representative_media(
            derived,
            media_layout=SINGLE_VIEW_LEFT_LAYOUT,
        )
        multiview_groups = validate_representative_media(
            derived,
            media_layout=SEPARATE_MULTIVIEW_LAYOUT,
        )
        for group in multiview_groups:
            for frame in group:
                for path_value in frame.values():
                    if not Path(path_value).is_file():
                        raise FileNotFoundError(path_value)
        derived_rows.append(derived)

    expected_ids = [
        str(cluster["cluster_id"]) for cluster in eligible_clusters
    ]
    if [str(row["cluster_id"]) for row in derived_rows] != expected_ids:
        raise ValueError(f"{partition}: derived cluster order drifted")
    if set(media_by_cluster) != set(expected_ids):
        raise ValueError(f"{partition}: media cluster coverage drifted")
    if set(centroid_diversity_by_cluster) != set(expected_ids):
        raise ValueError(
            f"{partition}: centroid/diversity cluster coverage drifted"
        )

    metadata = {
        "partition": partition,
        "source_clusters_path": str(source_clusters_path),
        "source_clusters_sha256": _sha256_file(source_clusters_path),
        "event_features_path": str(event_features_path),
        "event_features_sha256": _sha256_file(event_features_path),
        "legacy_media_catalog_path": str(media_catalog_path),
        "legacy_media_catalog_sha256": _sha256_file(media_catalog_path),
        "legacy_media_manifest_path": str(media_manifest_path),
        "legacy_media_manifest_sha256": _sha256_file(media_manifest_path),
        "current_v12_catalog_path": str(centroid_diversity_catalog_path),
        "current_v12_catalog_sha256": _sha256_file(
            centroid_diversity_catalog_path
        ),
        "current_v12_manifest_path": str(representative_manifest_path),
        "current_v12_manifest_sha256": _sha256_file(
            representative_manifest_path
        ),
        "selection_config": expected_config,
        "selection_strategy": SELECTION_STRATEGY,
        "selection_roles": list(SELECTION_ROLES),
        "derived_cluster_count": len(derived_rows),
        "derived_representative_count": 5 * len(derived_rows),
        "derived_catalog_sha256": hashlib.sha256(
            _jsonl_bytes(derived_rows)
        ).hexdigest(),
        "centroid_prefix_matches_current_v12": True,
        "legacy_media_matches_centroid_five": True,
    }
    return derived_rows, metadata


def _multiview_media_features_path(
    partition: str,
    partition_metadata: Mapping[str, Any],
) -> Path:
    if partition == "p0":
        return (
            EXPERIMENT_ROOT
            / "features/r_pos/event_features_multiview.jsonl"
        ).resolve()
    return Path(str(partition_metadata["event_features_path"])).resolve()


def _source_view_frames(
    record: Mapping[str, Any],
) -> list[dict[str, str]]:
    by_view = record.get("selected_view_frame_paths")
    if not isinstance(by_view, Mapping):
        by_view = record.get("view_frame_paths")
    expected_views = ("left", "right", "wrist")
    if (
        not isinstance(by_view, Mapping)
        or set(by_view) != set(expected_views)
    ):
        raise ValueError(
            f"{record.get('sample_id')}: missing synchronized multiview media"
        )
    paths = {
        view: [str(value) for value in by_view[view]]
        for view in expected_views
    }
    if any(len(values) != 5 for values in paths.values()):
        raise ValueError(
            f"{record.get('sample_id')}: expected five frames per view"
        )
    return [
        {
            view: paths[view][frame_index]
            for view in expected_views
        }
        for frame_index in range(5)
    ]


def _load_adaptive_partition_rows(
    partition: str,
    *,
    representative_count: int = len(ADAPTIVE_REPRESENTATIVE_INDICES),
    target_ids: Sequence[str] | None = None,
    selection_strategy: str = ADAPTIVE_SELECTION_STRATEGY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extend selected centroid-five rows to a deterministic rank prefix."""

    base_rows, base_metadata = _load_partition_rows(partition)
    if representative_count < len(REPRESENTATIVE_INDICES):
        raise ValueError("Centroid extension cannot shorten the source prefix")
    target_set = (
        None if target_ids is None else {str(value) for value in target_ids}
    )
    base_cluster_ids = {str(row["cluster_id"]) for row in base_rows}
    if target_set is not None and not target_set <= base_cluster_ids:
        missing = sorted(target_set - base_cluster_ids)
        raise ValueError(f"{partition}: unknown centroid targets: {missing}")
    source_clusters_path = Path(
        str(base_metadata["source_clusters_path"])
    ).resolve()
    selection_features_path = Path(
        str(base_metadata["event_features_path"])
    ).resolve()
    media_features_path = _multiview_media_features_path(
        partition,
        base_metadata,
    )
    source_clusters = {
        str(row["cluster_id"]): row
        for row in load_jsonl(source_clusters_path)
    }
    selection_rows = load_jsonl(selection_features_path)
    selection_by_id = {
        str(row["sample_id"]): row for row in selection_rows
    }
    media_rows = load_jsonl(media_features_path)
    media_by_id = {
        str(row["sample_id"]): row for row in media_rows
    }
    if len(selection_by_id) != len(selection_rows):
        raise ValueError(f"{partition}: duplicate selection feature IDs")
    if len(media_by_id) != len(media_rows):
        raise ValueError(f"{partition}: duplicate media feature IDs")
    selection_config = dict(base_metadata["selection_config"])
    vector_by_id = _vector_index(
        selection_rows,
        selection_config=selection_config,
    )

    extended_rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        cluster_id = str(base_row["cluster_id"])
        if target_set is not None and cluster_id not in target_set:
            extended_rows.append(dict(base_row))
            continue
        try:
            source_cluster = source_clusters[cluster_id]
            member_ids = [
                str(value)
                for value in source_cluster["member_sample_ids"]
            ]
            member_records = [
                selection_by_id[sample_id] for sample_id in member_ids
            ]
            member_vectors = np.stack(
                [vector_by_id[sample_id] for sample_id in member_ids]
            )
        except KeyError as exc:
            raise ValueError(
                f"{partition}/{cluster_id}: missing adaptive join {exc}"
            ) from exc
        representatives, distances, raw_ranks = (
            select_centroid_representatives(
                member_records,
                member_vectors,
                count=representative_count,
            )
        )
        selected_ids = [
            str(record["sample_id"]) for record in representatives
        ]
        if selected_ids[:5] != [
            str(value) for value in base_row["representative_sample_ids"]
        ]:
            raise ValueError(
                f"{partition}/{cluster_id}: centroid-five prefix drifted"
            )
        try:
            selected_media = [
                media_by_id[sample_id] for sample_id in selected_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"{partition}/{cluster_id}: missing adaptive media {exc}"
            ) from exc
        source_groups = [
            _source_view_frames(record) for record in selected_media
        ]
        single_view_groups = [
            [str(frame["left"]) for frame in group]
            for group in source_groups
        ]
        if single_view_groups[:5] != [
            [str(path) for path in group]
            for group in base_row[
                "representative_single_view_frame_paths"
            ]
        ]:
            raise ValueError(
                f"{partition}/{cluster_id}: LEFT media prefix drifted"
            )
        if source_groups[:5] != base_row[
            "representative_source_view_frame_paths"
        ]:
            raise ValueError(
                f"{partition}/{cluster_id}: multiview media prefix drifted"
            )

        extended = dict(base_row)
        extended.pop("representative_frame_paths", None)
        extended.update(
            {
                "representative_sample_ids": selected_ids,
                "representative_episode_nums": [
                    int(record["episode_num"])
                    for record in representatives
                ],
                "representative_clip_paths": [
                    str(record["clip_path"])
                    for record in representatives
                ],
                "representative_waypoint_steps": [
                    int(record["waypoint_step"])
                    for record in representatives
                ],
                "representative_progress_percents": [
                    float(record["progress_percent"])
                    for record in representatives
                ],
                "representative_single_view_frame_paths": (
                    single_view_groups
                ),
                "representative_source_view_frame_paths": source_groups,
                "representative_selection_strategy": (
                    selection_strategy
                ),
                "representative_selection_roles": [
                    "centroid"
                    for _ in range(representative_count)
                ],
                "representative_centroid_cosine_distances": distances,
                "representative_centroid_raw_distance_ranks": raw_ranks,
                "representative_unique_episode_policy": "error",
            }
        )
        if (
            len(set(extended["representative_episode_nums"]))
            != representative_count
        ):
            raise ValueError(
                f"{partition}/{cluster_id}: adaptive episodes are not unique"
            )
        validate_representative_media(
            extended,
            media_layout=SINGLE_VIEW_LEFT_LAYOUT,
        )
        multiview_groups = validate_representative_media(
            extended,
            media_layout=SEPARATE_MULTIVIEW_LAYOUT,
        )
        for group in multiview_groups:
            for frame in group:
                for path_value in frame.values():
                    if not Path(path_value).is_file():
                        raise FileNotFoundError(path_value)
        extended_rows.append(extended)

    metadata = {
        "partition": partition,
        "base_partition_contract": base_metadata,
        "selection_features_path": str(selection_features_path),
        "selection_features_sha256": _sha256_file(
            selection_features_path
        ),
        "media_features_path": str(media_features_path),
        "media_features_sha256": _sha256_file(media_features_path),
        "selection_strategy": selection_strategy,
        "selection_roles": [
            "centroid" for _ in range(representative_count)
        ],
        "representative_indices": list(range(1, representative_count + 1)),
        "extended_target_ids": (
            sorted(base_cluster_ids)
            if target_set is None
            else sorted(target_set)
        ),
        "derived_cluster_count": len(extended_rows),
        "derived_representative_count": sum(
            len(row["representative_sample_ids"]) for row in extended_rows
        ),
        "derived_catalog_sha256": hashlib.sha256(
            _jsonl_bytes(extended_rows)
        ).hexdigest(),
        "centroid_five_prefix_matches_source": True,
    }
    return extended_rows, metadata


def _condition_contract(
    *,
    spec: ConditionSpec,
    rows: Sequence[Mapping[str, Any]],
    partition_metadata: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    annotation_view, view_order, _ = resolve_annotation_media_layout(
        spec.media_layout
    )
    media_records = []
    task_descriptions = sorted(
        {str(row["task_description"]) for row in rows}
    )
    prompt_hashes: dict[str, str] = {}
    schema_hashes: dict[str, str] = {}
    for task_description in task_descriptions:
        rendered = [
            build_representative_clip_annotation_prompt(
                task_description=task_description,
                cluster_id="contract-only",
                representative_index=index,
                num_frames=5,
                media_layout=spec.media_layout,
            )
            for index in REPRESENTATIVE_INDICES
        ]
        if len(set(rendered)) != 1:
            raise ValueError(
                f"{spec.condition_id}: prompt unexpectedly exposes index"
            )
        prompt_hashes[task_description] = hashlib.sha256(
            rendered[0].encode("utf-8")
        ).hexdigest()
        schema_hashes[task_description] = canonical_sha256(
            build_representative_response_schema(task_description)
        )
    for row in rows:
        groups = validate_representative_media(
            row,
            media_layout=spec.media_layout,
        )
        for representative_index, group in enumerate(groups, start=1):
            media_records.append(
                {
                    "cluster_id": str(row["cluster_id"]),
                    "representative_index": representative_index,
                    "source_media_sha256": hash_annotation_media(
                        group,
                        view_order=view_order,
                    ),
                }
            )
    return {
        "experiment_id": spec.experiment_id,
        "condition_id": spec.condition_id,
        "partition": spec.partition,
        "annotation_view": annotation_view,
        "media_layout": spec.media_layout,
        "view_order": list(view_order),
        "source_catalog_path": partition_metadata[
            "legacy_media_catalog_path"
        ],
        "source_catalog_sha256": partition_metadata[
            "legacy_media_catalog_sha256"
        ],
        "derived_catalog_sha256": partition_metadata[
            "derived_catalog_sha256"
        ],
        "selected_cluster_rows": len(rows),
        "representative_count": 5 * len(rows),
        "image_reference_count": (
            5 * len(view_order) * 5 * len(rows)
        ),
        "requested_media_sha256": canonical_sha256(media_records),
        "task_descriptions": task_descriptions,
        "rendered_prompt_sha256_by_task": prompt_hashes,
        "response_schema_sha256_by_task": schema_hashes,
        "output_path": str(
            (
                output_root
                / spec.condition_id
                / "centroid_annotations.jsonl"
            ).resolve()
        ),
    }


def _plurality_source_rows() -> dict[str, list[dict[str, Any]]]:
    source_manifest = load_json_object(
        SOURCE_CENTROID_ROOT / "run_manifest.json"
    )
    if source_manifest.get("contract_sha256") != (
        SOURCE_RUN_CONTRACT_SHA256
    ):
        raise ValueError("Source V12 run contract drifted")
    derivation_manifest = load_json_object(
        SOURCE_PLURALITY_ROOT / "derivation_manifest.json"
    )
    derivation_contract = derivation_manifest.get("contract")
    if (
        not isinstance(derivation_contract, Mapping)
        or derivation_contract.get("source_run_contract_sha256")
        != SOURCE_RUN_CONTRACT_SHA256
    ):
        raise ValueError("Source V12 plurality contract drifted")

    result: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    total_mixed = 0
    for spec in CONDITIONS:
        path = (
            SOURCE_PLURALITY_ROOT
            / spec.condition_id
            / "plurality_annotations.jsonl"
        )
        rows = load_jsonl(path)
        cluster_ids = [str(row["cluster_id"]) for row in rows]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError(
                f"{spec.condition_id}: duplicate source annotation IDs"
            )
        for row in rows:
            if row.get("status") != "mixed":
                continue
            consensus = row.get("consensus")
            counts = (
                consensus.get("phase_counts")
                if isinstance(consensus, Mapping)
                else None
            )
            representative_annotations = row.get(
                "representative_annotations"
            )
            if (
                row.get("phase") is not None
                or not isinstance(counts, Mapping)
                or sorted(int(value) for value in counts.values())
                != [1, 2, 2]
                or not isinstance(representative_annotations, list)
                or len(representative_annotations) != 5
            ):
                raise ValueError(
                    f"{spec.condition_id}/{row.get('cluster_id')}: "
                    "adaptive source must be an exact usable 2-2-1 tie"
                )
            total_mixed += 1
        result[spec.condition_id] = rows
        total_rows += len(rows)
    if total_rows != 90 or total_mixed != 18:
        raise ValueError(
            "Adaptive V12 source inventory drifted: "
            f"rows={total_rows}, mixed={total_mixed}"
        )
    return result


def _adaptive_source_file_hashes() -> dict[str, str]:
    paths = [
        SOURCE_CENTROID_ROOT / "run_manifest.json",
        SOURCE_CENTROID_ROOT / "run_summary.json",
        SOURCE_PLURALITY_ROOT / "derivation_manifest.json",
        SOURCE_PLURALITY_ROOT / "run_summary.json",
    ]
    for spec in CONDITIONS:
        paths.extend(
            (
                SOURCE_CENTROID_ROOT
                / spec.condition_id
                / "centroid_annotations.jsonl",
                SOURCE_PLURALITY_ROOT
                / spec.condition_id
                / "plurality_annotations.jsonl",
            )
        )
    return {
        str(path.resolve().relative_to(EXPERIMENT_ROOT.resolve())): (
            _sha256_file(path)
        )
        for path in paths
    }


def _continuation_source_rows() -> dict[str, list[dict[str, Any]]]:
    """Load the audited rank-nine child without executing its old runtime."""

    manifest_path = ADAPTIVE_OUTPUT_ROOT / "run_manifest.json"
    manifest = load_json_object(manifest_path)
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Rank-nine source manifest has no contract")
    embedded_sha256 = canonical_sha256(contract)
    if (
        manifest.get("contract_sha256") != embedded_sha256
        or embedded_sha256 != CONTINUATION_SOURCE_RUN_CONTRACT_SHA256
    ):
        raise ValueError("Rank-nine source run contract drifted")

    summary = load_json_object(ADAPTIVE_OUTPUT_ROOT / "run_summary.json")
    expected_summary = {
        "run_contract_sha256": CONTINUATION_SOURCE_RUN_CONTRACT_SHA256,
        "total_rows": 90,
        "accepted_rows": 89,
        "unresolved_rows": 1,
        "source_provider_responses": 450,
        "fresh_provider_responses": 46,
        "unique_response_ids": 496,
    }
    mismatched = [
        key for key, value in expected_summary.items()
        if summary.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"Rank-nine source summary drifted: {mismatched}"
        )

    result: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[tuple[str, dict[str, Any]]] = []
    response_ids: list[str] = []
    total_rows = 0
    for spec in CONDITIONS:
        source_path = (
            ADAPTIVE_OUTPUT_ROOT
            / spec.condition_id
            / "annotations.jsonl"
        )
        accepted_path = source_path.with_name("accepted_annotations.jsonl")
        rows = load_jsonl(source_path)
        accepted = load_jsonl(accepted_path)
        if accepted != [row for row in rows if row.get("phase") is not None]:
            raise ValueError(
                f"{spec.condition_id}: rank-nine accepted rows drifted"
            )
        cluster_ids = [str(row["cluster_id"]) for row in rows]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError(
                f"{spec.condition_id}: duplicate rank-nine cluster IDs"
            )
        for row in rows:
            annotations = row.get("representative_annotations")
            if not isinstance(annotations, list):
                raise ValueError(
                    f"{spec.condition_id}/{row.get('cluster_id')}: "
                    "rank-nine annotations are missing"
                )
            response_ids.extend(
                str(annotation["response_id"])
                for annotation in annotations
            )
            if row.get("phase") is None:
                unresolved.append((spec.condition_id, row))
        result[spec.condition_id] = rows
        total_rows += len(rows)

    if total_rows != 90:
        raise ValueError(f"Expected 90 rank-nine rows, got {total_rows}")
    if len(response_ids) != 496 or len(set(response_ids)) != 496:
        raise ValueError("Rank-nine response inventory drifted")
    if len(unresolved) != 1:
        raise ValueError(
            f"Expected one rank-nine tie, got {len(unresolved)}"
        )
    condition_id, target = unresolved[0]
    annotations = target.get("representative_annotations")
    consensus = target.get("consensus")
    phase_counts = (
        consensus.get("phase_counts")
        if isinstance(consensus, Mapping)
        else None
    )
    indices = (
        [int(row["representative_index"]) for row in annotations]
        if isinstance(annotations, list)
        else []
    )
    if (
        condition_id != CONTINUATION_TARGET_CONDITION_ID
        or target.get("cluster_id") != CONTINUATION_TARGET_CLUSTER_ID
        or target.get("status") != "mixed-after-9"
        or target.get("phase") is not None
        or phase_counts != {"grasp": 4, "place": 1, "transport": 4}
        or indices != list(CONTINUATION_SOURCE_INDICES)
    ):
        raise ValueError("Rank-nine continuation target drifted")
    return result


def _continuation_source_file_hashes() -> dict[str, str]:
    paths = [
        ADAPTIVE_OUTPUT_ROOT / "run_manifest.json",
        ADAPTIVE_OUTPUT_ROOT / "run_summary.json",
    ]
    for spec in CONDITIONS:
        condition_root = ADAPTIVE_OUTPUT_ROOT / spec.condition_id
        paths.extend(
            (
                condition_root / "annotations.jsonl",
                condition_root / "accepted_annotations.jsonl",
            )
        )
    return {
        str(path.resolve().relative_to(EXPERIMENT_ROOT.resolve())): (
            _sha256_file(path)
        )
        for path in paths
    }


def _adaptive_condition_contract(
    *,
    spec: ConditionSpec,
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    partition_metadata: Mapping[str, Any],
    output_root: Path,
    target_ids: Sequence[str] | None = None,
    available_indices: Sequence[int] = ADAPTIVE_REPRESENTATIVE_INDICES,
    request_indices: Sequence[int] = ADAPTIVE_REQUEST_INDICES,
    source_root: Path = SOURCE_PLURALITY_ROOT,
) -> dict[str, Any]:
    annotation_view, view_order, _ = resolve_annotation_media_layout(
        spec.media_layout
    )
    source_by_id = {
        str(row["cluster_id"]): row for row in source_rows
    }
    if target_ids is None:
        resolved_target_ids = [
            str(row["cluster_id"])
            for row in source_rows
            if row.get("status") == "mixed"
        ]
    else:
        resolved_target_ids = [str(value) for value in target_ids]
    row_by_id = {str(row["cluster_id"]): row for row in rows}
    try:
        target_rows = [
            row_by_id[cluster_id] for cluster_id in resolved_target_ids
        ]
    except KeyError as exc:
        raise ValueError(
            f"{spec.condition_id}: adaptive target is missing {exc}"
        ) from exc
    if len(source_by_id) != len(source_rows):
        raise ValueError(f"{spec.condition_id}: duplicate source row IDs")

    task_descriptions = sorted(
        {str(row["task_description"]) for row in target_rows}
    )
    prompt_hashes: dict[str, str] = {}
    schema_hashes: dict[str, str] = {}
    for task_description in task_descriptions:
        rendered = [
            build_representative_clip_annotation_prompt(
                task_description=task_description,
                cluster_id="contract-only",
                representative_index=index,
                num_frames=5,
                media_layout=spec.media_layout,
            )
            for index in available_indices
        ]
        if len(set(rendered)) != 1:
            raise ValueError(
                f"{spec.condition_id}: prompt unexpectedly exposes index"
            )
        prompt_hashes[task_description] = hashlib.sha256(
            rendered[0].encode("utf-8")
        ).hexdigest()
        schema_hashes[task_description] = canonical_sha256(
            build_representative_response_schema(task_description)
        )

    media_records = []
    for row in target_rows:
        groups = validate_representative_media(
            row,
            media_layout=spec.media_layout,
        )
        if len(groups) != len(available_indices):
            raise ValueError(
                f"{spec.condition_id}/{row['cluster_id']}: "
                "adaptive media does not match the available rank prefix"
            )
        for representative_index in request_indices:
            media_records.append(
                {
                    "cluster_id": str(row["cluster_id"]),
                    "representative_index": representative_index,
                    "sample_id": str(
                        row["representative_sample_ids"][
                            representative_index - 1
                        ]
                    ),
                    "episode_num": int(
                        row["representative_episode_nums"][
                            representative_index - 1
                        ]
                    ),
                    "source_media_sha256": hash_annotation_media(
                        groups[representative_index - 1],
                        view_order=view_order,
                    ),
                }
            )
    source_path = (
        source_root
        / spec.condition_id
        / (
            "plurality_annotations.jsonl"
            if source_root == SOURCE_PLURALITY_ROOT
            else "annotations.jsonl"
        )
    ).resolve()
    return {
        "experiment_id": spec.experiment_id,
        "condition_id": spec.condition_id,
        "partition": spec.partition,
        "annotation_view": annotation_view,
        "media_layout": spec.media_layout,
        "view_order": list(view_order),
        "source_annotation_path": str(source_path),
        "source_annotation_sha256": _sha256_file(source_path),
        "source_annotation_rows": len(source_rows),
        "target_cluster_rows": len(target_rows),
        "target_cluster_ids": resolved_target_ids,
        "target_cluster_ids_sha256": canonical_sha256(
            resolved_target_ids
        ),
        "target_source_rows_sha256": canonical_sha256(
            {
                "rows": [
                    source_by_id[cluster_id]
                    for cluster_id in resolved_target_ids
                ]
            }
        ),
        "derived_catalog_sha256": partition_metadata[
            "derived_catalog_sha256"
        ],
        "request_representative_indices": list(
            request_indices
        ),
        "maximum_fresh_requests": (
            len(request_indices) * len(target_rows)
        ),
        "maximum_image_reference_count": (
            len(request_indices)
            * len(view_order)
            * 5
            * len(target_rows)
        ),
        "requested_media_sha256": canonical_sha256(media_records),
        "task_descriptions": task_descriptions,
        "rendered_prompt_sha256_by_task": prompt_hashes,
        "response_schema_sha256_by_task": schema_hashes,
        "output_path": str(
            (
                output_root
                / spec.condition_id
                / "annotations.jsonl"
            ).resolve()
        ),
        "accepted_output_path": str(
            (
                output_root
                / spec.condition_id
                / "accepted_annotations.jsonl"
            ).resolve()
        ),
    }


def _new_fixed_contract(output_root: Path) -> dict[str, Any]:
    partitions: dict[str, dict[str, Any]] = {}
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in ("p0", "p1", "p2", "p3"):
        rows, metadata = _load_partition_rows(partition)
        rows_by_partition[partition] = rows
        partitions[partition] = metadata
    conditions = [
        _condition_contract(
            spec=spec,
            rows=rows_by_partition[spec.partition],
            partition_metadata=partitions[spec.partition],
            output_root=output_root,
        )
        for spec in CONDITIONS
    ]
    total_rows = sum(
        int(condition["selected_cluster_rows"])
        for condition in conditions
    )
    return {
        "format": RUN_FORMAT,
        "implementation_file_sha256": {
            path: _sha256_file(REPO_ROOT / path)
            for path in ADAPTER_FILES
        },
        "historical_lineage": {
            "resume_allowed": False,
            "source_archive": str(
                EXPERIMENT_ROOT
                / "archives/annotation_lineage_source_20260725.tar"
            ),
            "source_archive_sha256": (
                "71d6ddf9395c5bb41ec3e1a0d455dc55fcdf925ade75e0ec"
                "b15ebb441411543d"
            ),
        },
        "package_versions": _package_versions(),
        "model": MODEL,
        "expected_response_model_version": EXPECTED_MODEL_VERSION,
        "temperature": TEMPERATURE,
        "min_episode_coverage": MIN_EPISODE_COVERAGE,
        "prompt_id": ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
        "response_schema_id": RESPONSE_SCHEMA_ID,
        "response_mime_type": RESPONSE_MIME_TYPE,
        "response_normalization_policy_id": NORMALIZATION_POLICY_ID,
        "request_exposure_policy": dict(REQUEST_EXPOSURE_POLICY),
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "representative_selection": {
            "strategy": SELECTION_STRATEGY,
            "roles": list(SELECTION_ROLES),
            "representative_indices": list(REPRESENTATIVE_INDICES),
            "unique_episode_policy": "error",
            "distance": "cosine_to_normalized_cluster_centroid",
            "tie_break_fields": [
                "episode_num",
                "waypoint_step",
                "waypoint_rank",
                "sample_id",
            ],
        },
        "consensus": {
            "representatives_evaluated": 5,
            "accept_votes": CONSENSUS_VOTES,
            "rule": "accept one phase with at least four usable votes",
            "insufficient_visibility_votes_excluded": True,
        },
        "response_reuse": {
            "allowed": False,
            "policy": "all_five_representatives_are_fresh_provider_requests",
        },
        "transport": {
            "api": "Gemini Batch generateContent",
            "source": "inlined_requests",
            "max_serialized_chunk_bytes": MAX_SERIALIZED_CHUNK_BYTES,
            "max_request_attempts": MAX_REQUEST_ATTEMPTS,
            "create_is_non_idempotent": True,
            "ambiguous_create_auto_retry": False,
        },
        "partitions": partitions,
        "conditions": conditions,
        "expected_total_cluster_rows": total_rows,
        "expected_logical_requests": 5 * total_rows,
    }


def _new_adaptive_contract(output_root: Path) -> dict[str, Any]:
    source_rows_by_condition = _plurality_source_rows()
    partitions: dict[str, dict[str, Any]] = {}
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in ("p0", "p1", "p2", "p3"):
        rows, metadata = _load_adaptive_partition_rows(partition)
        rows_by_partition[partition] = rows
        partitions[partition] = metadata
    conditions = [
        _adaptive_condition_contract(
            spec=spec,
            rows=rows_by_partition[spec.partition],
            source_rows=source_rows_by_condition[spec.condition_id],
            partition_metadata=partitions[spec.partition],
            output_root=output_root,
        )
        for spec in CONDITIONS
    ]
    target_rows = sum(
        int(condition["target_cluster_rows"])
        for condition in conditions
    )
    if target_rows != 18:
        raise ValueError(f"Expected 18 adaptive targets, got {target_rows}")
    source_derivation_manifest = load_json_object(
        SOURCE_PLURALITY_ROOT / "derivation_manifest.json"
    )
    return {
        "format": ADAPTIVE_RUN_FORMAT,
        "run_kind": "adaptive-plurality",
        "implementation_file_sha256": {
            path: _sha256_file(REPO_ROOT / path)
            for path in ADAPTER_FILES
        },
        "package_versions": _package_versions(),
        "model": MODEL,
        "expected_response_model_version": EXPECTED_MODEL_VERSION,
        "temperature": TEMPERATURE,
        "min_episode_coverage": MIN_EPISODE_COVERAGE,
        "prompt_id": ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
        "response_schema_id": RESPONSE_SCHEMA_ID,
        "response_mime_type": RESPONSE_MIME_TYPE,
        "response_normalization_policy_id": NORMALIZATION_POLICY_ID,
        "request_exposure_policy": dict(REQUEST_EXPOSURE_POLICY),
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "source_annotation": {
            "centroid_root": str(SOURCE_CENTROID_ROOT.resolve()),
            "plurality_root": str(SOURCE_PLURALITY_ROOT.resolve()),
            "source_run_contract_sha256": (
                SOURCE_RUN_CONTRACT_SHA256
            ),
            "source_derivation_contract_sha256": (
                source_derivation_manifest["contract_sha256"]
            ),
            "files_sha256": _adaptive_source_file_hashes(),
            "reused_representative_indices": list(
                REPRESENTATIVE_INDICES
            ),
            "source_rows_reused": 90,
            "source_provider_responses_reused": 450,
        },
        "representative_selection": {
            "strategy": ADAPTIVE_SELECTION_STRATEGY,
            "roles": [
                "centroid" for _ in ADAPTIVE_REPRESENTATIVE_INDICES
            ],
            "available_representative_indices": list(
                ADAPTIVE_REPRESENTATIVE_INDICES
            ),
            "fresh_request_indices": list(ADAPTIVE_REQUEST_INDICES),
            "unique_episode_policy": "error",
            "distance": "cosine_to_normalized_cluster_centroid",
            "tie_break_fields": [
                "episode_num",
                "waypoint_step",
                "waypoint_rank",
                "sample_id",
            ],
        },
        "consensus": {
            "policy_id": (
                "adaptive_unique_plurality_at_odd_checkpoints_v1"
            ),
            "initial_representatives": 5,
            "checkpoints": [7, 9],
            "checkpoint_request_indices": [
                list(indices) for indices in ADAPTIVE_CHECKPOINTS
            ],
            "minimum_plurality_votes": 2,
            "acceptance_rule": "unique-argmax-at-checkpoint",
            "tie_policy": "advance-to-next-checkpoint",
            "max_rank_tie_status": "mixed-after-9",
            "insufficient_visibility_votes_excluded": True,
        },
        "response_reuse": {
            "source_indices_1_to_5": "read_only_reuse_by_hash",
            "fresh_indices_6_to_9": "fresh_provider_responses_only",
        },
        "transport": {
            "api": "Gemini Batch generateContent",
            "source": "inlined_requests",
            "wave": "centroid-tie-extension",
            "max_serialized_chunk_bytes": MAX_SERIALIZED_CHUNK_BYTES,
            "max_request_attempts": MAX_REQUEST_ATTEMPTS,
            "create_is_non_idempotent": True,
            "ambiguous_create_auto_retry": False,
        },
        "partitions": partitions,
        "conditions": conditions,
        "expected_source_rows": 90,
        "expected_target_rows": target_rows,
        "initial_fresh_requests": 2 * target_rows,
        "maximum_fresh_requests": (
            len(ADAPTIVE_REQUEST_INDICES) * target_rows
        ),
    }


def _validate_continuation_prefix(
    *,
    spec: ConditionSpec,
    cluster: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> None:
    """Prove that the derived ranks 1..9 are the persisted parent inputs."""

    source_annotations = source_row.get("representative_annotations")
    if (
        not isinstance(source_annotations, list)
        or len(source_annotations) != len(CONTINUATION_SOURCE_INDICES)
    ):
        raise ValueError(
            f"{spec.condition_id}/{cluster['cluster_id']}: "
            "invalid continuation source prefix"
        )
    expected_sample_ids = [
        str(value)
        for value in cluster["representative_sample_ids"][
            : len(CONTINUATION_SOURCE_INDICES)
        ]
    ]
    expected_episode_nums = [
        int(value)
        for value in cluster["representative_episode_nums"][
            : len(CONTINUATION_SOURCE_INDICES)
        ]
    ]
    if expected_sample_ids != [
        str(value) for value in source_row["representative_sample_ids"]
    ]:
        raise ValueError(
            f"{spec.condition_id}/{cluster['cluster_id']}: "
            "continuation sample prefix drifted"
        )
    if expected_episode_nums != [
        int(value) for value in source_row["representative_episode_nums"]
    ]:
        raise ValueError(
            f"{spec.condition_id}/{cluster['cluster_id']}: "
            "continuation episode prefix drifted"
        )

    _, view_order, _ = resolve_annotation_media_layout(spec.media_layout)
    groups = validate_representative_media(
        cluster,
        media_layout=spec.media_layout,
    )
    for offset, source_annotation in enumerate(source_annotations):
        representative_index = offset + 1
        expected_group = groups[offset]
        if (
            int(source_annotation["representative_index"])
            != representative_index
            or str(source_annotation["sample_id"])
            != expected_sample_ids[offset]
            or str(source_annotation["clip_path"])
            != str(cluster["representative_clip_paths"][offset])
            or source_annotation.get("source_view_frame_paths")
            != expected_group
            or source_annotation.get("source_media_sha256")
            != hash_annotation_media(
                expected_group,
                view_order=view_order,
            )
        ):
            raise ValueError(
                f"{spec.condition_id}/{cluster['cluster_id']}: "
                f"continuation media prefix drifted at rank "
                f"{representative_index}"
            )


def _new_continuation_contract(output_root: Path) -> dict[str, Any]:
    source_rows_by_condition = _continuation_source_rows()
    target_ids_by_condition = {
        spec.condition_id: [
            str(row["cluster_id"])
            for row in source_rows_by_condition[spec.condition_id]
            if row.get("phase") is None
        ]
        for spec in CONDITIONS
    }
    target_ids_by_partition: dict[str, set[str]] = defaultdict(set)
    for spec in CONDITIONS:
        target_ids_by_partition[spec.partition].update(
            target_ids_by_condition[spec.condition_id]
        )

    partitions: dict[str, dict[str, Any]] = {}
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in ("p0", "p1", "p2", "p3"):
        rows, metadata = _load_adaptive_partition_rows(
            partition,
            representative_count=len(
                CONTINUATION_REPRESENTATIVE_INDICES
            ),
            target_ids=sorted(target_ids_by_partition[partition]),
            selection_strategy=CONTINUATION_SELECTION_STRATEGY,
        )
        rows_by_partition[partition] = rows
        partitions[partition] = metadata

    for spec in CONDITIONS:
        source_by_id = {
            str(row["cluster_id"]): row
            for row in source_rows_by_condition[spec.condition_id]
        }
        cluster_by_id = {
            str(row["cluster_id"]): row
            for row in rows_by_partition[spec.partition]
        }
        for cluster_id in target_ids_by_condition[spec.condition_id]:
            _validate_continuation_prefix(
                spec=spec,
                cluster=cluster_by_id[cluster_id],
                source_row=source_by_id[cluster_id],
            )

    conditions = [
        _adaptive_condition_contract(
            spec=spec,
            rows=rows_by_partition[spec.partition],
            source_rows=source_rows_by_condition[spec.condition_id],
            partition_metadata=partitions[spec.partition],
            output_root=output_root,
            target_ids=target_ids_by_condition[spec.condition_id],
            available_indices=CONTINUATION_REPRESENTATIVE_INDICES,
            request_indices=CONTINUATION_REQUEST_INDICES,
            source_root=ADAPTIVE_OUTPUT_ROOT,
        )
        for spec in CONDITIONS
    ]
    target_rows = sum(
        int(condition["target_cluster_rows"])
        for condition in conditions
    )
    if target_rows != 1:
        raise ValueError(
            f"Expected one rank-nine continuation target, got {target_rows}"
        )
    return {
        "format": CONTINUATION_RUN_FORMAT,
        "run_kind": "adaptive-plurality",
        "stage": "rank9-to-rank17",
        "implementation_file_sha256": {
            path: _sha256_file(REPO_ROOT / path)
            for path in ADAPTER_FILES
        },
        "package_versions": _package_versions(),
        "model": MODEL,
        "expected_response_model_version": EXPECTED_MODEL_VERSION,
        "temperature": TEMPERATURE,
        "min_episode_coverage": MIN_EPISODE_COVERAGE,
        "prompt_id": ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
        "response_schema_id": RESPONSE_SCHEMA_ID,
        "response_mime_type": RESPONSE_MIME_TYPE,
        "response_normalization_policy_id": NORMALIZATION_POLICY_ID,
        "request_exposure_policy": dict(REQUEST_EXPOSURE_POLICY),
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "source_annotation": {
            "root": str(ADAPTIVE_OUTPUT_ROOT.resolve()),
            "source_run_contract_sha256": (
                CONTINUATION_SOURCE_RUN_CONTRACT_SHA256
            ),
            "files_sha256": _continuation_source_file_hashes(),
            "reused_representative_indices": list(
                CONTINUATION_SOURCE_INDICES
            ),
            "source_rows_reused": 90,
            "source_provider_responses_reused": 496,
        },
        "representative_selection": {
            "strategy": CONTINUATION_SELECTION_STRATEGY,
            "roles": [
                "centroid"
                for _ in CONTINUATION_REPRESENTATIVE_INDICES
            ],
            "available_representative_indices": list(
                CONTINUATION_REPRESENTATIVE_INDICES
            ),
            "fresh_request_indices": list(
                CONTINUATION_REQUEST_INDICES
            ),
            "unique_episode_policy": "error",
            "distance": "cosine_to_normalized_cluster_centroid",
            "tie_break_fields": [
                "episode_num",
                "waypoint_step",
                "waypoint_rank",
                "sample_id",
            ],
        },
        "consensus": {
            "policy_id": (
                "adaptive_unique_plurality_at_odd_checkpoints_v1"
            ),
            "initial_representatives": 9,
            "checkpoints": [11, 13, 15, 17],
            "checkpoint_request_indices": [
                list(indices) for indices in CONTINUATION_CHECKPOINTS
            ],
            "minimum_plurality_votes": 2,
            "acceptance_rule": "unique-argmax-at-checkpoint",
            "tie_policy": "advance-to-next-checkpoint",
            "max_rank_tie_status": "mixed-after-17",
            "insufficient_visibility_votes_excluded": True,
        },
        "response_reuse": {
            "source_indices_1_to_9": "read_only_reuse_by_hash",
            "fresh_indices_10_to_17": "fresh_provider_responses_only",
        },
        "transport": {
            "api": "Gemini Batch generateContent",
            "source": "inlined_requests",
            "wave": "centroid-tie-continuation",
            "max_serialized_chunk_bytes": MAX_SERIALIZED_CHUNK_BYTES,
            "max_request_attempts": MAX_REQUEST_ATTEMPTS,
            "create_is_non_idempotent": True,
            "ambiguous_create_auto_retry": False,
        },
        "partitions": partitions,
        "conditions": conditions,
        "expected_source_rows": 90,
        "expected_target_rows": target_rows,
        "initial_fresh_requests": 2 * target_rows,
        "maximum_fresh_requests": (
            len(CONTINUATION_REQUEST_INDICES) * target_rows
        ),
    }


def _new_contract(output_root: Path) -> dict[str, Any]:
    if output_root.resolve() == CONTINUATION_OUTPUT_ROOT.resolve():
        return _new_continuation_contract(output_root)
    if output_root.resolve() == ADAPTIVE_OUTPUT_ROOT.resolve():
        return _new_adaptive_contract(output_root)
    return _new_fixed_contract(output_root)


def freeze_or_validate_manifest(output_root: Path) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    contract = _new_contract(output_root)
    contract_sha = canonical_sha256(contract)
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        embedded = manifest.get("contract")
        if not isinstance(embedded, dict):
            raise ValueError(f"Manifest has no contract: {manifest_path}")
        embedded_sha = canonical_sha256(embedded)
        if manifest.get("contract_sha256") != embedded_sha:
            raise ValueError(f"Manifest is internally invalid: {manifest_path}")
        if embedded_sha != contract_sha:
            raise ValueError(
                "Centroid annotation contract drifted; refusing to continue: "
                f"frozen={embedded_sha}, current={contract_sha}"
            )
        return manifest
    output_root.mkdir(parents=True, exist_ok=True)
    stale = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    ]
    if stale:
        raise FileExistsError(
            f"Fresh centroid output root contains stale files: {stale}"
        )
    manifest = {
        "frozen_at_utc": utc_now(),
        "contract_sha256": contract_sha,
        "contract": contract,
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


def _selection_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    representative_indices: Sequence[int] = REPRESENTATIVE_INDICES,
    selection_strategy: str = SELECTION_STRATEGY,
) -> dict[str, Any]:
    cluster_ids = [str(row["cluster_id"]) for row in rows]
    return {
        "min_episode_coverage": MIN_EPISODE_COVERAGE,
        "selection_strategy": selection_strategy,
        "representative_indices": list(representative_indices),
        "target_cluster_count": len(rows),
        "target_cluster_ids": cluster_ids,
        "target_cluster_ids_sha256": canonical_sha256(cluster_ids),
        "target_source_rows_sha256": canonical_sha256(
            {"clusters": list(rows)}
        ),
    }


def _load_fixed_contexts(
    manifest: Mapping[str, Any],
) -> list[ConditionContext]:
    contract = manifest["contract"]
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition, expected in contract["partitions"].items():
        rows, metadata = _load_partition_rows(str(partition))
        if metadata != expected:
            raise ValueError(f"{partition}: partition contract drifted")
        rows_by_partition[str(partition)] = rows

    contexts: list[ConditionContext] = []
    for condition in contract["conditions"]:
        partition = str(condition["partition"])
        rows = rows_by_partition[partition]
        if hashlib.sha256(_jsonl_bytes(rows)).hexdigest() != condition[
            "derived_catalog_sha256"
        ]:
            raise ValueError(
                f"{condition['condition_id']}: derived catalog drifted"
            )
        annotation_view, view_order, _ = resolve_annotation_media_layout(
            str(condition["media_layout"])
        )
        if (
            annotation_view != condition["annotation_view"]
            or list(view_order) != condition["view_order"]
        ):
            raise ValueError(
                f"{condition['condition_id']}: media contract drifted"
            )
        for row in rows:
            validate_representative_media(
                row,
                media_layout=str(condition["media_layout"]),
            )
        source_clusters = {
            str(row["cluster_id"]): row for row in rows
        }
        output_path = Path(str(condition["output_path"])).resolve()
        contexts.append(
            ConditionContext(
                experiment_id=str(condition["experiment_id"]),
                condition_id=str(condition["condition_id"]),
                clusters_path=Path(
                    str(condition["source_catalog_path"])
                ).resolve(),
                output_path=output_path,
                annotation_view=annotation_view,
                media_layout=str(condition["media_layout"]),
                view_order=tuple(view_order),
                source_rows=rows,
                source_clusters=source_clusters,
                selection_contract=_selection_contract(rows),
                targeted_rows=rows,
                target_ids=[
                    str(row["cluster_id"]) for row in rows
                ],
            )
        )
    return contexts


def _load_adaptive_contexts(
    manifest: Mapping[str, Any],
) -> list[ConditionContext]:
    contract = manifest["contract"]
    continuation = contract.get("stage") == "rank9-to-rank17"
    if continuation:
        source_rows_by_condition = _continuation_source_rows()
        target_ids_by_partition: dict[str, set[str]] = defaultdict(set)
        for condition in contract["conditions"]:
            target_ids_by_partition[str(condition["partition"])].update(
                str(value)
                for value in condition["target_cluster_ids"]
            )
        adaptive_config = AdaptiveExtensionConfig(
            source_rank=9,
            maximum_rank=17,
            checkpoints=CONTINUATION_CHECKPOINTS,
            selection_strategy=CONTINUATION_SELECTION_STRATEGY,
            source_run_contract_sha256=(
                CONTINUATION_SOURCE_RUN_CONTRACT_SHA256
            ),
            source_provider_response_count=496,
            output_format=CONTINUATION_OUTPUT_FORMAT,
            summary_format=CONTINUATION_SUMMARY_FORMAT,
            audit_format=(
                "event_sae_v12_adaptive_plurality_rank17_audit_v1"
            ),
            source_annotation_field="continuation_source_annotation",
            extension_field="adaptive_continuation",
        )
    else:
        source_rows_by_condition = _plurality_source_rows()
        target_ids_by_partition = defaultdict(set)
        adaptive_config = AdaptiveExtensionConfig(
            source_rank=5,
            maximum_rank=9,
            checkpoints=ADAPTIVE_CHECKPOINTS,
            selection_strategy=ADAPTIVE_SELECTION_STRATEGY,
            source_run_contract_sha256=SOURCE_RUN_CONTRACT_SHA256,
            source_provider_response_count=450,
            output_format=ADAPTIVE_OUTPUT_FORMAT,
            summary_format=ADAPTIVE_SUMMARY_FORMAT,
            audit_format="event_sae_v12_adaptive_plurality_audit_v1",
            source_annotation_field="source_annotation",
            extension_field="adaptive_extension",
        )
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition, expected in contract["partitions"].items():
        if continuation:
            rows, metadata = _load_adaptive_partition_rows(
                str(partition),
                representative_count=adaptive_config.maximum_rank,
                target_ids=sorted(
                    target_ids_by_partition[str(partition)]
                ),
                selection_strategy=adaptive_config.selection_strategy,
            )
        else:
            rows, metadata = _load_adaptive_partition_rows(str(partition))
        if metadata != expected:
            raise ValueError(f"{partition}: adaptive partition drifted")
        rows_by_partition[str(partition)] = rows

    contexts: list[ConditionContext] = []
    for condition in contract["conditions"]:
        condition_id = str(condition["condition_id"])
        partition = str(condition["partition"])
        rows = rows_by_partition[partition]
        if hashlib.sha256(_jsonl_bytes(rows)).hexdigest() != condition[
            "derived_catalog_sha256"
        ]:
            raise ValueError(
                f"{condition_id}: adaptive catalog drifted"
            )
        source_rows = source_rows_by_condition[condition_id]
        source_path = Path(
            str(condition["source_annotation_path"])
        ).resolve()
        if (
            _sha256_file(source_path)
            != condition["source_annotation_sha256"]
            or len(source_rows)
            != int(condition["source_annotation_rows"])
        ):
            raise ValueError(
                f"{condition_id}: source annotation drifted"
            )
        annotation_view, view_order, _ = (
            resolve_annotation_media_layout(
                str(condition["media_layout"])
            )
        )
        if (
            annotation_view != condition["annotation_view"]
            or list(view_order) != condition["view_order"]
        ):
            raise ValueError(
                f"{condition_id}: adaptive media contract drifted"
            )
        source_clusters = {
            str(row["cluster_id"]): row for row in rows
        }
        target_ids = [
            str(value) for value in condition["target_cluster_ids"]
        ]
        try:
            targeted_rows = [
                source_clusters[cluster_id]
                for cluster_id in target_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"{condition_id}: unknown adaptive target {exc}"
            ) from exc
        for row in targeted_rows:
            groups = validate_representative_media(
                row,
                media_layout=str(condition["media_layout"]),
            )
            if len(groups) != adaptive_config.maximum_rank:
                raise ValueError(
                    f"{condition_id}/{row['cluster_id']}: "
                    "adaptive representative prefix drifted"
                )
            if continuation:
                _validate_continuation_prefix(
                    spec=next(
                        spec
                        for spec in CONDITIONS
                        if spec.condition_id == condition_id
                    ),
                    cluster=row,
                    source_row=next(
                        source
                        for source in source_rows
                        if str(source["cluster_id"])
                        == str(row["cluster_id"])
                    ),
                )
        base_annotations = {
            str(row["cluster_id"]): row for row in source_rows
        }
        if len(base_annotations) != len(source_rows):
            raise ValueError(
                f"{condition_id}: duplicate base annotation IDs"
            )
        contexts.append(
            ConditionContext(
                experiment_id=str(condition["experiment_id"]),
                condition_id=condition_id,
                clusters_path=source_path,
                output_path=Path(
                    str(condition["output_path"])
                ).resolve(),
                annotation_view=annotation_view,
                media_layout=str(condition["media_layout"]),
                view_order=tuple(view_order),
                source_rows=rows,
                source_clusters=source_clusters,
                selection_contract=_selection_contract(
                    targeted_rows,
                    representative_indices=adaptive_config.available_indices,
                    selection_strategy=adaptive_config.selection_strategy,
                ),
                targeted_rows=targeted_rows,
                target_ids=target_ids,
                base_annotations=base_annotations,
                adaptive_config=adaptive_config,
            )
        )
    return contexts


def load_contexts(
    manifest: Mapping[str, Any],
) -> list[ConditionContext]:
    if manifest["contract"].get("run_kind") == "adaptive-plurality":
        return _load_adaptive_contexts(manifest)
    return _load_fixed_contexts(manifest)


def load_attempts(
    output_root: Path,
    logical_id: str,
) -> list[dict[str, Any]]:
    return _load_attempts(output_root, logical_id)


def successful_attempt(
    output_root: Path,
    logical_id: str,
) -> dict[str, Any] | None:
    return _successful_attempt(output_root, logical_id)


def batch_wave(manifest: Mapping[str, Any]) -> str:
    if manifest["contract"].get("run_kind") == "adaptive-plurality":
        return str(manifest["contract"]["transport"]["wave"])
    return "centroid-five"


def requestable_representative_indices(
    manifest: Mapping[str, Any],
) -> tuple[int, ...]:
    if manifest["contract"].get("run_kind") == "adaptive-plurality":
        return tuple(
            int(value)
            for value in manifest["contract"][
                "representative_selection"
            ]["fresh_request_indices"]
        )
    return REPRESENTATIVE_INDICES


def _request_descriptor(
    *,
    context: ConditionContext,
    cluster: Mapping[str, Any],
    representative_index: int,
    attempt: int,
) -> tuple[dict[str, Any], Any]:
    """Build one deterministic provider request from the semantic API."""

    source_groups = validate_representative_media(
        cluster,
        media_layout=context.media_layout,
    )
    source_frames = source_groups[representative_index - 1]
    prompt = build_representative_clip_annotation_prompt(
        task_description=str(cluster["task_description"]),
        cluster_id=str(cluster["cluster_id"]),
        representative_index=representative_index,
        num_frames=len(source_frames),
        media_layout=context.media_layout,
    )
    clip_paths = [
        str(value) for value in cluster["representative_clip_paths"]
    ]
    validate_clean_visual_annotation_prompt(
        prompt,
        forbidden_identifiers=(
            str(cluster["cluster_id"]),
            *(
                str(value)
                for value in cluster["representative_sample_ids"]
            ),
            *clip_paths,
            *(Path(value).name for value in clip_paths),
            *ROBOCASA_PHASE_LABELER_PROVENANCE.values(),
        ),
    )
    raw_parts = build_representative_request_parts(
        prompt=prompt,
        source_view_frames=source_frames,
        view_order=context.view_order,
    )
    parts = [
        (
            types.Part.from_bytes(
                data=part.data,
                mime_type=part.mime_type,
            )
            if isinstance(part, AnnotationImagePart)
            else types.Part.from_text(text=part)
        )
        for part in raw_parts
    ]
    logical_id = annotation_request_id(
        context.condition_id,
        str(cluster["cluster_id"]),
        representative_index,
    )
    request_key = _request_key(logical_id, attempt)
    request = types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        metadata={"key": request_key},
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            response_mime_type=RESPONSE_MIME_TYPE,
            response_json_schema=build_representative_response_schema(
                str(cluster["task_description"])
            ),
            media_resolution=(
                types.MediaResolution.MEDIA_RESOLUTION_HIGH
            ),
        ),
    )
    request_dump = serialize_inlined_request(request)
    semantic_dump = dict(request_dump)
    semantic_dump.pop("metadata", None)
    entry = {
        "logical_id": logical_id,
        "condition_id": context.condition_id,
        "experiment_id": context.experiment_id,
        "cluster_id": str(cluster["cluster_id"]),
        "representative_index": int(representative_index),
        "attempt": int(attempt),
        "request_key": request_key,
        "task_description": str(cluster["task_description"]),
        "source_cluster_sha256": hash_canonical_record(cluster),
        "representative_set_sha256": hash_representative_set(cluster),
        "source_media_sha256": hash_annotation_media(
            source_frames,
            view_order=context.view_order,
        ),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "semantic_request_sha256": canonical_sha256(
            {
                "batch_model": MODEL,
                "inlined_request_without_metadata": semantic_dump,
            }
        ),
        "wire_request_sha256": canonical_sha256(
            {
                "batch_model": MODEL,
                "inlined_request": request_dump,
            }
        ),
        "serialized_request_size_bytes": len(
            _canonical_json_bytes(request_dump)
        ),
    }
    return entry, request


def required_requests(
    output_root: Path,
    contexts: Sequence[ConditionContext],
) -> list[tuple[dict[str, Any], Any]]:
    """Compile every missing fixed-five request, including deterministic retry."""

    if contexts and contexts[0].base_annotations is not None:
        return _adaptive_required_requests(output_root, contexts)
    compiled: list[tuple[dict[str, Any], Any]] = []
    for context in contexts:
        for cluster in context.targeted_rows:
            for representative_index in REPRESENTATIVE_INDICES:
                logical_id = (
                    f"{context.condition_id}|{cluster['cluster_id']}|"
                    f"representative-{representative_index}"
                )
                success = _successful_attempt(output_root, logical_id)
                attempts = _load_attempts(output_root, logical_id)
                if success is not None:
                    entry, _ = _request_descriptor(
                        context=context,
                        cluster=cluster,
                        representative_index=representative_index,
                        attempt=int(success["attempt"]),
                    )
                    if (
                        success.get("semantic_request_sha256")
                        != entry["semantic_request_sha256"]
                        or success.get("wire_request_sha256")
                        != entry["wire_request_sha256"]
                    ):
                        raise ValueError(
                            f"Successful request contract drifted: {logical_id}"
                        )
                    continue
                next_attempt = len(attempts) + 1
                if next_attempt > MAX_REQUEST_ATTEMPTS:
                    raise RuntimeError(
                        f"{logical_id} exhausted "
                        f"{MAX_REQUEST_ATTEMPTS} attempts"
                    )
                entry, request = _request_descriptor(
                    context=context,
                    cluster=cluster,
                    representative_index=representative_index,
                    attempt=next_attempt,
                )
                for prior in attempts:
                    prior_entry, _ = _request_descriptor(
                        context=context,
                        cluster=cluster,
                        representative_index=representative_index,
                        attempt=int(prior["attempt"]),
                    )
                    if (
                        prior.get("semantic_request_sha256")
                        != prior_entry["semantic_request_sha256"]
                        or prior.get("wire_request_sha256")
                        != prior_entry["wire_request_sha256"]
                    ):
                        raise ValueError(
                            f"Prior request contract drifted: {logical_id}"
                        )
                compiled.append((entry, request))
    return compiled


def _compile_missing_request(
    *,
    output_root: Path,
    context: ConditionContext,
    cluster: Mapping[str, Any],
    representative_index: int,
) -> tuple[dict[str, Any], Any] | None:
    logical_id = annotation_request_id(
        context.condition_id,
        str(cluster["cluster_id"]),
        representative_index,
    )
    success = _successful_attempt(output_root, logical_id)
    attempts = _load_attempts(output_root, logical_id)
    if success is not None:
        entry, _ = _request_descriptor(
            context=context,
            cluster=cluster,
            representative_index=representative_index,
            attempt=int(success["attempt"]),
        )
        if (
            success.get("semantic_request_sha256")
            != entry["semantic_request_sha256"]
            or success.get("wire_request_sha256")
            != entry["wire_request_sha256"]
        ):
            raise ValueError(
                f"Successful request contract drifted: {logical_id}"
            )
        return None
    next_attempt = len(attempts) + 1
    if next_attempt > MAX_REQUEST_ATTEMPTS:
        raise RuntimeError(
            f"{logical_id} exhausted {MAX_REQUEST_ATTEMPTS} attempts"
        )
    entry, request = _request_descriptor(
        context=context,
        cluster=cluster,
        representative_index=representative_index,
        attempt=next_attempt,
    )
    for prior in attempts:
        prior_entry, _ = _request_descriptor(
            context=context,
            cluster=cluster,
            representative_index=representative_index,
            attempt=int(prior["attempt"]),
        )
        if (
            prior.get("semantic_request_sha256")
            != prior_entry["semantic_request_sha256"]
            or prior.get("wire_request_sha256")
            != prior_entry["wire_request_sha256"]
        ):
            raise ValueError(
                f"Prior request contract drifted: {logical_id}"
            )
    return entry, request


def _adaptive_base_annotations(
    context: ConditionContext,
    cluster_id: str,
) -> list[dict[str, Any]]:
    if (
        context.base_annotations is None
        or context.adaptive_config is None
    ):
        raise ValueError("Adaptive context has no base annotations")
    try:
        source_row = context.base_annotations[cluster_id]
    except KeyError as exc:
        raise ValueError(
            f"{context.condition_id}: missing base annotation {cluster_id}"
        ) from exc
    annotations = [
        dict(record)
        for record in source_row["representative_annotations"]
    ]
    indices = [
        int(record["representative_index"]) for record in annotations
    ]
    if indices != list(context.adaptive_config.source_indices):
        raise ValueError(
            f"{context.condition_id}/{cluster_id}: "
            f"source indices drifted: {indices}"
        )
    return annotations


def _adaptive_required_requests(
    output_root: Path,
    contexts: Sequence[ConditionContext],
) -> list[tuple[dict[str, Any], Any]]:
    compiled: list[tuple[dict[str, Any], Any]] = []
    for context in contexts:
        if (
            context.base_annotations is None
            or context.adaptive_config is None
        ):
            raise ValueError("Mixed fixed/adaptive contexts are forbidden")
        config = context.adaptive_config
        for cluster in context.targeted_rows:
            cluster_id = str(cluster["cluster_id"])
            annotations = _adaptive_base_annotations(
                context,
                cluster_id,
            )
            initial = resolve_adaptive_plurality(
                annotations,
                maximum_rank=config.maximum_rank,
            )
            if initial["phase"] is not None:
                raise ValueError(
                    f"{context.condition_id}/{cluster_id}: "
                    "adaptive target is already resolved"
                )
            for checkpoint_indices in config.checkpoints:
                checkpoint_annotations: list[dict[str, Any]] = []
                checkpoint_missing: list[tuple[dict[str, Any], Any]] = []
                for representative_index in checkpoint_indices:
                    logical_id = annotation_request_id(
                        context.condition_id,
                        cluster_id,
                        representative_index,
                    )
                    success = _successful_attempt(
                        output_root,
                        logical_id,
                    )
                    if success is not None:
                        checkpoint_annotations.append(
                            _success_for_representative(
                                output_root=output_root,
                                context=context,
                                cluster=cluster,
                                representative_index=representative_index,
                            )
                        )
                        continue
                    request = _compile_missing_request(
                        output_root=output_root,
                        context=context,
                        cluster=cluster,
                        representative_index=representative_index,
                    )
                    if request is not None:
                        checkpoint_missing.append(request)
                if checkpoint_missing:
                    compiled.extend(checkpoint_missing)
                    break
                if len(checkpoint_annotations) != len(checkpoint_indices):
                    raise RuntimeError(
                        f"{context.condition_id}/{cluster_id}: "
                        "adaptive checkpoint is incomplete"
                    )
                checkpoint_annotations.sort(
                    key=lambda row: int(row["representative_index"])
                )
                annotations.extend(checkpoint_annotations)
                consensus = resolve_adaptive_plurality(
                    annotations,
                    maximum_rank=config.maximum_rank,
                )
                if consensus["phase"] is not None:
                    break
    return compiled


def expected_logical_request_ids(
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[ConditionContext],
) -> list[str]:
    """Return the exact logical IDs required by the frozen stopping rule."""

    if manifest["contract"].get("run_kind") != "adaptive-plurality":
        return sorted(
            annotation_request_id(
                context.condition_id,
                str(cluster["cluster_id"]),
                representative_index,
            )
            for context in contexts
            for cluster in context.targeted_rows
            for representative_index in REPRESENTATIVE_INDICES
        )

    expected: set[str] = set()
    for context in contexts:
        if context.adaptive_config is None:
            raise ValueError("Adaptive context has no extension config")
        config = context.adaptive_config
        for cluster in context.targeted_rows:
            cluster_id = str(cluster["cluster_id"])
            annotations = _adaptive_base_annotations(
                context,
                cluster_id,
            )
            for checkpoint_indices in config.checkpoints:
                for representative_index in checkpoint_indices:
                    logical_id = annotation_request_id(
                        context.condition_id,
                        cluster_id,
                        representative_index,
                    )
                    expected.add(logical_id)
                    annotations.append(
                        _success_for_representative(
                            output_root=output_root,
                            context=context,
                            cluster=cluster,
                            representative_index=representative_index,
                        )
                    )
                annotations.sort(
                    key=lambda row: int(row["representative_index"])
                )
                consensus = resolve_adaptive_plurality(
                    annotations,
                    maximum_rank=config.maximum_rank,
                )
                if consensus["phase"] is not None:
                    break
    return sorted(expected)


def validate_plan_entry(
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[ConditionContext],
    entry: Mapping[str, Any],
) -> tuple[ConditionContext, dict[str, Any], Any]:
    del output_root
    contract = manifest["contract"]
    response_reuse = contract["response_reuse"]
    if contract.get("run_kind") == "adaptive-plurality":
        source_indices = [
            int(value)
            for value in contract["source_annotation"][
                "reused_representative_indices"
            ]
        ]
        fresh_indices = [
            int(value)
            for value in contract["representative_selection"][
                "fresh_request_indices"
            ]
        ]
        expected_response_reuse = {
            f"source_indices_1_to_{source_indices[-1]}": (
                "read_only_reuse_by_hash"
            ),
            (
                f"fresh_indices_{fresh_indices[0]}_to_"
                f"{fresh_indices[-1]}"
            ): "fresh_provider_responses_only",
        }
        if response_reuse != expected_response_reuse:
            raise ValueError("Adaptive response-reuse contract drifted")
    elif response_reuse.get("allowed") is not False:
        raise ValueError("Fresh-response contract drifted")
    context_by_id = {
        context.condition_id: context for context in contexts
    }
    condition_id = str(entry["condition_id"])
    try:
        context = context_by_id[condition_id]
        cluster = context.source_clusters[str(entry["cluster_id"])]
    except KeyError as exc:
        raise ValueError(f"Plan entry target is unknown: {entry}") from exc
    representative_index = int(entry["representative_index"])
    if representative_index not in requestable_representative_indices(
        manifest
    ):
        raise ValueError(f"Invalid representative index: {entry}")
    expected_entry, request = _request_descriptor(
        context=context,
        cluster=cluster,
        representative_index=representative_index,
        attempt=int(entry["attempt"]),
    )
    if dict(entry) != expected_entry:
        changed = sorted(
            set(entry).union(expected_entry)
            - {
                key
                for key in set(entry).intersection(expected_entry)
                if entry[key] == expected_entry[key]
            }
        )
        raise ValueError(
            f"Plan entry request drifted for {entry.get('logical_id')}: "
            f"{changed}"
        )
    return context, cluster, request


@dataclass
class _NormalizedResponseProxy:
    text: str
    usage_metadata: Any
    model_version: Any
    response_id: Any


def _inline_response_dump(inline_response: Any) -> dict[str, Any]:
    if isinstance(inline_response, Mapping):
        return dict(inline_response)
    if not hasattr(inline_response, "model_dump"):
        raise TypeError("Provider inline response has an unexpected type")
    return inline_response.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


def _provider_error_text(error: Any) -> str:
    if hasattr(error, "model_dump"):
        value = error.model_dump(mode="json", exclude_none=True)
    else:
        value = str(error)
    return redact_sensitive_error(
        json.dumps(value, ensure_ascii=False)
        if not isinstance(value, str)
        else value
    )


def _optional_response_text(response: Any) -> str | None:
    try:
        value = getattr(response, "text", None)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _base_attempt_record(
    *,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    entry: Mapping[str, Any],
    job_name: str,
) -> dict[str, Any]:
    return {
        "format": ATTEMPT_FORMAT,
        "origin": "fresh_provider_batch",
        "run_contract_sha256": manifest["contract_sha256"],
        "logical_id": entry["logical_id"],
        "condition_id": entry["condition_id"],
        "cluster_id": entry["cluster_id"],
        "representative_index": int(entry["representative_index"]),
        "attempt": int(entry["attempt"]),
        "request_key": entry["request_key"],
        "semantic_request_sha256": entry["semantic_request_sha256"],
        "wire_request_sha256": entry["wire_request_sha256"],
        "plan_id": plan["plan_id"],
        "chunk_id": chunk["chunk_id"],
        "job_name": job_name,
    }


def _representative_record_from_response(
    *,
    entry: Mapping[str, Any],
    context: ConditionContext,
    cluster: Mapping[str, Any],
    response: Any,
    plan_id: str,
    chunk_id: str,
    job_name: str,
    response_sha256: str,
) -> dict[str, Any]:
    model_version = getattr(response, "model_version", None)
    if model_version != EXPECTED_MODEL_VERSION:
        raise ValueError(
            "Batch response actual model_version failed hard gate: "
            f"{model_version!r}"
        )
    response_id = getattr(response, "response_id", None)
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("Batch response has no response_id")
    usage_metadata = serialize_usage_metadata(response)
    if usage_metadata is None:
        raise ValueError("Batch response has no usage_metadata")
    raw_response = getattr(response, "text", None)
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("Batch response has no non-empty text")
    parsed, parse_error = parse_representative_response(
        raw_response,
        task_description=str(cluster["task_description"]),
    )
    if parsed is None or parse_error is not None:
        raise ValueError(
            f"Batch structured response failed schema: {parse_error}"
        )

    representative_index = int(entry["representative_index"])
    source_frames = validate_representative_media(
        cluster,
        media_layout=context.media_layout,
    )[representative_index - 1]
    prompt = build_representative_clip_annotation_prompt(
        task_description=str(cluster["task_description"]),
        cluster_id=str(cluster["cluster_id"]),
        representative_index=representative_index,
        num_frames=len(source_frames),
        media_layout=context.media_layout,
    )
    record = {
        "representative_index": representative_index,
        "sample_id": str(
            cluster["representative_sample_ids"][
                representative_index - 1
            ]
        ),
        "clip_path": str(
            cluster["representative_clip_paths"][
                representative_index - 1
            ]
        ),
        "source_view_frame_paths": [
            {
                view: str(frame[view])
                for view in context.view_order
            }
            for frame in source_frames
        ],
        "source_media_sha256": hash_annotation_media(
            source_frames,
            view_order=context.view_order,
        ),
        "prompt_text": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "phase": parsed["phase"],
        "raw_phase": parsed["raw_phase"],
        "phrase": parsed["phrase"],
        "visibility": parsed["visibility"],
        "raw_response": raw_response,
        "parse_error": None,
        "api_error": None,
        "usage_metadata": usage_metadata,
        "model_version": model_version,
        "response_id": response_id,
        "request_attempts": int(entry["attempt"]),
        "batch_provenance": {
            "transport": "Gemini Batch generateContent",
            "request_key": str(entry["request_key"]),
            "plan_id": plan_id,
            "chunk_id": chunk_id,
            "job_name": job_name,
            "semantic_request_sha256": entry[
                "semantic_request_sha256"
            ],
            "wire_request_sha256": entry["wire_request_sha256"],
            "response_sha256": response_sha256,
        },
    }
    validate_representative_annotation(
        record,
        task_description=str(cluster["task_description"]),
    )
    return record


def build_inline_attempt_record(
    *,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    entry: Mapping[str, Any],
    context: ConditionContext,
    cluster: Mapping[str, Any],
    inline_response: Any,
    job_name: str,
) -> tuple[dict[str, Any], bool]:
    """Replay one provider envelope into its exact persisted attempt record."""

    inline_dump = _inline_response_dump(inline_response)
    response_sha256 = canonical_sha256(inline_dump)
    base = {
        **_base_attempt_record(
            manifest=manifest,
            plan=plan,
            chunk=chunk,
            entry=entry,
            job_name=job_name,
        ),
        "batch_response_sha256": response_sha256,
        "provider_inline_response": inline_dump,
        "collected_at_utc": utc_now(),
    }

    def failure(
        status: str,
        error: str,
        *,
        provider_raw_response: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return (
            {
                **base,
                "status": status,
                "error": redact_sensitive_error(error),
                "raw_response": provider_raw_response,
                "provider_raw_response": provider_raw_response,
                "response_normalization": None,
                "representative_annotation": None,
            },
            False,
        )

    metadata = getattr(inline_response, "metadata", None)
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("key") != entry["request_key"]
    ):
        return failure(
            "response_contract_error",
            "Inline response metadata key does not match its request: "
            f"expected={entry['request_key']!r}",
        )
    provider_error = getattr(inline_response, "error", None)
    response = getattr(inline_response, "response", None)
    if provider_error is not None:
        return failure(
            "request_error",
            _provider_error_text(provider_error),
        )
    if response is None:
        return failure(
            "response_contract_error",
            "Batch result has neither response nor error: "
            f"{entry['request_key']}",
        )

    provider_raw_response = _optional_response_text(response)
    model_version = getattr(response, "model_version", None)
    response_id = getattr(response, "response_id", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    if model_version != EXPECTED_MODEL_VERSION:
        return failure(
            "response_contract_error",
            "Batch response actual model_version failed hard gate: "
            f"{model_version!r}",
            provider_raw_response=provider_raw_response,
        )
    if not isinstance(response_id, str) or not response_id.strip():
        return failure(
            "response_contract_error",
            "Batch response has no response_id",
            provider_raw_response=provider_raw_response,
        )
    if usage_metadata is None:
        return failure(
            "response_contract_error",
            "Batch response has no usage_metadata",
            provider_raw_response=provider_raw_response,
        )
    if (
        not isinstance(provider_raw_response, str)
        or not provider_raw_response.strip()
    ):
        return failure(
            "parse_error",
            "Batch response has no non-empty text",
            provider_raw_response=provider_raw_response,
        )
    try:
        normalized = normalize_response_text(
            provider_raw_response,
            task_description=str(cluster["task_description"]),
        )
    except ValueError as exc:
        return failure(
            "parse_error",
            str(exc),
            provider_raw_response=provider_raw_response,
        )

    proxy = _NormalizedResponseProxy(
        text=normalized.normalized_text,
        usage_metadata=usage_metadata,
        model_version=model_version,
        response_id=response_id,
    )
    try:
        annotation = _representative_record_from_response(
            entry=entry,
            context=context,
            cluster=cluster,
            response=proxy,
            plan_id=str(plan["plan_id"]),
            chunk_id=str(chunk["chunk_id"]),
            job_name=job_name,
            response_sha256=response_sha256,
        )
        normalization_record = normalization_provenance(normalized)
        annotation["response_normalization"] = normalization_record
        annotation["batch_provenance"] = {
            **dict(annotation["batch_provenance"]),
            "response_normalization_policy_id": (
                NORMALIZATION_POLICY_ID
            ),
            "provider_raw_response_sha256": normalization_record[
                "source_text_sha256"
            ],
            "normalized_response_sha256": normalization_record[
                "normalized_text_sha256"
            ],
            "fresh_response_required": True,
        }
        validate_representative_annotation(
            annotation,
            task_description=str(cluster["task_description"]),
        )
    except ValueError as exc:
        return failure(
            "response_contract_error",
            str(exc),
            provider_raw_response=provider_raw_response,
        )
    return (
        {
            **base,
            "status": "success",
            "error": None,
            "raw_response": normalized.normalized_text,
            "provider_raw_response": provider_raw_response,
            "response_normalization": normalization_record,
            "representative_annotation": annotation,
        },
        True,
    )


def record_job_failure(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    entry: Mapping[str, Any],
    job_name: str,
    state: str,
    error: str,
) -> bool:
    path = _attempt_path(
        output_root,
        str(entry["logical_id"]),
        int(entry["attempt"]),
    )
    existed = path.exists()
    record = {
        **_base_attempt_record(
            manifest=manifest,
            plan=plan,
            chunk=chunk,
            entry=entry,
            job_name=job_name,
        ),
        "status": "job_error",
        "error": redact_sensitive_error(f"{state}: {error}"),
        "raw_response": None,
        "provider_raw_response": None,
        "response_normalization": None,
        "batch_response_sha256": None,
        "provider_inline_response": None,
        "representative_annotation": None,
        "collected_at_utc": utc_now(),
    }
    _write_attempt(path, record)
    return not existed


def write_attempt_record(
    output_root: Path,
    entry: Mapping[str, Any],
    record: Mapping[str, Any],
) -> bool:
    path = _attempt_path(
        output_root,
        str(entry["logical_id"]),
        int(entry["attempt"]),
    )
    existed = path.exists()
    _write_attempt(path, record)
    return not existed


@dataclass(frozen=True)
class ConsensusPolicy:
    policy_id: str
    minimum_votes: int
    require_unique_winner: bool = True


FIXED_STRONG_CONSENSUS = ConsensusPolicy(
    policy_id="strong_consensus_4_of_fixed_5_v1",
    minimum_votes=4,
)
STRICT_MAJORITY = ConsensusPolicy(
    policy_id="strict_majority_3_of_fixed_5_v1",
    minimum_votes=3,
)
UNIQUE_PLURALITY = ConsensusPolicy(
    policy_id="unique_plurality_of_fixed_5_minimum_2_votes_v1",
    minimum_votes=2,
)
CONSENSUS_POLICIES = {
    "strict-majority": STRICT_MAJORITY,
    "unique-plurality": UNIQUE_PLURALITY,
}


def resolve_adaptive_plurality(
    annotations: Sequence[Mapping[str, Any]],
    *,
    exhausted: bool = False,
    maximum_rank: int = 9,
) -> dict[str, Any]:
    """Resolve a unique plurality over a contiguous centroid prefix."""

    records = list(annotations)
    if maximum_rank < 5 or maximum_rank % 2 == 0:
        raise ValueError("Adaptive maximum rank must be odd and at least five")
    if not 5 <= len(records) <= maximum_rank:
        raise ValueError(
            f"Adaptive plurality requires 5..{maximum_rank} records"
        )
    indices = [int(record["representative_index"]) for record in records]
    expected_indices = list(range(1, len(records) + 1))
    if indices != expected_indices:
        raise ValueError(
            "Adaptive records must be ordered contiguous indices 1..N, "
            f"got {indices}"
        )
    if exhausted and len(records) != maximum_rank:
        raise ValueError(
            "Only the configured maximum-rank prefix can be exhausted"
        )
    usable = [
        record
        for record in records
        if (
            record.get("api_error") is None
            and record.get("parse_error") is None
            and record.get("visibility") != "insufficient"
            and isinstance(record.get("phase"), str)
        )
    ]
    phase_counts = Counter(str(record["phase"]) for record in usable)
    ordered = sorted(
        phase_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_phase = ordered[0][0] if ordered else None
    dominant_votes = ordered[0][1] if ordered else 0
    runner_up_votes = ordered[1][1] if len(ordered) > 1 else 0
    unique_winner = (
        dominant_votes >= 2 and dominant_votes > runner_up_votes
    )
    phase = dominant_phase if unique_winner else None
    if unique_winner:
        decision = (
            "majority"
            if dominant_votes * 2 > len(usable)
            else "plurality"
        )
        status = (
            f"{decision}-{dominant_votes}-of-{len(records)}"
        )
    elif exhausted:
        status = f"mixed-after-{maximum_rank}"
    elif len(usable) < 2:
        status = "insufficient"
    else:
        status = "mixed"
    matching = [
        record
        for record in usable
        if phase is not None and str(record["phase"]) == phase
    ]
    matching.sort(
        key=lambda record: (
            0 if record.get("visibility") == "clear" else 1,
            int(record["representative_index"]),
        )
    )
    phrase_source = matching[0] if matching else None
    return {
        "policy_id": (
            "adaptive_unique_plurality_at_odd_checkpoints_v1"
        ),
        "status": status,
        "phase": phase,
        "phrase": (
            phrase_source.get("phrase")
            if phrase_source is not None
            else None
        ),
        "phrase_source_representative_index": (
            int(phrase_source["representative_index"])
            if phrase_source is not None
            else None
        ),
        "phase_counts": dict(sorted(phase_counts.items())),
        "num_representatives_evaluated": len(records),
        "num_usable_votes": len(usable),
        "num_insufficient_visibility": sum(
            record.get("visibility") == "insufficient"
            for record in records
        ),
        "dominant_votes": dominant_votes,
        "runner_up_votes": runner_up_votes,
        "vote_margin": dominant_votes - runner_up_votes,
        "unique_winner": unique_winner,
        "exhausted": exhausted,
        "dominant_unaccepted_phase": (
            dominant_phase if phase is None else None
        ),
        "vote_fraction_of_evaluated": (
            dominant_votes / len(records)
        ),
        "vote_fraction_of_usable": (
            dominant_votes / len(usable) if usable else 0.0
        ),
    }


def resolve_representative_consensus(
    annotations: Sequence[Mapping[str, Any]],
    policy: ConsensusPolicy,
) -> dict[str, Any]:
    records = list(annotations)
    if len(records) != 5:
        raise ValueError("Fixed-five consensus requires exactly five records")
    indices = [int(record["representative_index"]) for record in records]
    if indices != list(REPRESENTATIVE_INDICES):
        raise ValueError(
            f"Representative records must be ordered 1..5, got {indices}"
        )
    usable = [
        record
        for record in records
        if (
            record.get("api_error") is None
            and record.get("parse_error") is None
            and record.get("visibility") != "insufficient"
            and isinstance(record.get("phase"), str)
        )
    ]
    phase_counts = Counter(str(record["phase"]) for record in usable)
    ordered = sorted(
        phase_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_phase = ordered[0][0] if ordered else None
    dominant_votes = ordered[0][1] if ordered else 0
    runner_up_votes = ordered[1][1] if len(ordered) > 1 else 0
    accepted = (
        dominant_votes >= policy.minimum_votes
        and (
            not policy.require_unique_winner
            or dominant_votes > runner_up_votes
        )
    )
    if accepted:
        status = (
            "plurality-2-of-5"
            if dominant_votes == 2
            else f"consensus-{dominant_votes}-of-5"
        )
        phase = dominant_phase
    elif len(usable) < policy.minimum_votes:
        status = "insufficient"
        phase = None
    else:
        status = "mixed"
        phase = None
    matching = [
        record
        for record in usable
        if phase is not None and str(record["phase"]) == phase
    ]
    matching.sort(
        key=lambda record: (
            0 if record.get("visibility") == "clear" else 1,
            int(record["representative_index"]),
        )
    )
    phrase_source = matching[0] if matching else None
    return {
        "policy_id": policy.policy_id,
        "status": status,
        "phase": phase,
        "phrase": (
            phrase_source.get("phrase")
            if phrase_source is not None
            else None
        ),
        "phrase_source_representative_index": (
            int(phrase_source["representative_index"])
            if phrase_source is not None
            else None
        ),
        "phase_counts": dict(sorted(phase_counts.items())),
        "num_representatives_evaluated": 5,
        "num_usable_votes": len(usable),
        "num_insufficient_visibility": sum(
            record.get("visibility") == "insufficient"
            for record in records
        ),
        "dominant_unaccepted_phase": (
            None if phase is not None else dominant_phase
        ),
        "dominant_votes": dominant_votes,
        "runner_up_votes": runner_up_votes,
        "unique_winner": dominant_votes > runner_up_votes,
        "vote_fraction_of_evaluated": dominant_votes / 5,
        "vote_fraction_of_usable": (
            dominant_votes / len(usable) if usable else 0.0
        ),
    }


def fixed_five_consensus(
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility name for the raw run's four-of-five policy."""

    return resolve_representative_consensus(
        annotations,
        FIXED_STRONG_CONSENSUS,
    )


def _success_for_representative(
    *,
    output_root: Path,
    context: ConditionContext,
    cluster: Mapping[str, Any],
    representative_index: int,
) -> dict[str, Any]:
    logical_id = (
        f"{context.condition_id}|{cluster['cluster_id']}|"
        f"representative-{representative_index}"
    )
    attempt = _successful_attempt(output_root, logical_id)
    if attempt is None:
        raise RuntimeError(f"Missing successful response: {logical_id}")
    expected_entry, _ = _request_descriptor(
        context=context,
        cluster=cluster,
        representative_index=representative_index,
        attempt=int(attempt["attempt"]),
    )
    for field in (
        "semantic_request_sha256",
        "wire_request_sha256",
        "request_key",
    ):
        if attempt.get(field) != expected_entry[field]:
            raise ValueError(
                f"Successful response request drifted: {logical_id}/{field}"
            )
    annotation = attempt.get("representative_annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"Success has no annotation: {logical_id}")
    validate_representative_annotation(
        annotation,
        task_description=str(cluster["task_description"]),
    )
    normalization = annotation.get("response_normalization")
    if not isinstance(normalization, dict) or normalization.get(
        "policy_id"
    ) != NORMALIZATION_POLICY_ID:
        raise ValueError(f"Response normalization drifted: {logical_id}")
    return annotation


def _materialized_row(
    *,
    output_root: Path,
    context: ConditionContext,
    cluster: Mapping[str, Any],
) -> dict[str, Any]:
    annotations = [
        _success_for_representative(
            output_root=output_root,
            context=context,
            cluster=cluster,
            representative_index=index,
        )
        for index in REPRESENTATIVE_INDICES
    ]
    consensus = fixed_five_consensus(annotations)
    task_description = str(cluster["task_description"])
    phase_labels, _ = resolve_representative_phase_vocabulary(
        task_description
    )
    return {
        "format": OUTPUT_FORMAT,
        "cluster_id": str(cluster["cluster_id"]),
        "task_description": task_description,
        "episode_coverage": float(cluster["episode_coverage"]),
        "annotation_min_episode_coverage": MIN_EPISODE_COVERAGE,
        "source_catalog_path": str(context.clusters_path),
        "source_cluster_sha256": hash_canonical_record(cluster),
        "representative_set_sha256": hash_representative_set(cluster),
        "representative_sample_ids": list(
            cluster["representative_sample_ids"]
        ),
        "representative_episode_nums": list(
            cluster["representative_episode_nums"]
        ),
        "representative_selection_strategy": SELECTION_STRATEGY,
        "representative_selection_roles": list(SELECTION_ROLES),
        "model": MODEL,
        "prompt_version": ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
        "phase_scheme": "robocasa_action",
        "allowed_phase_labels": list(phase_labels),
        "request_exposure_policy": dict(REQUEST_EXPOSURE_POLICY),
        "phase_taxonomy_provenance": {
            **ROBOCASA_PHASE_LABELER_PROVENANCE,
            "sent_to_model": False,
        },
        "oracle_phase_used": False,
        "annotation_view": context.annotation_view,
        "media_layout": context.media_layout,
        "view_order": list(context.view_order),
        "selection_contract": dict(context.selection_contract),
        "generation_config": {
            "temperature": TEMPERATURE,
            "response_mime_type": RESPONSE_MIME_TYPE,
            "response_schema_version": RESPONSE_SCHEMA_ID,
            "response_json_schema": build_representative_response_schema(
                task_description
            ),
            "response_normalization_policy_id": NORMALIZATION_POLICY_ID,
        },
        "fixed_consensus_config": {
            "representatives_evaluated": 5,
            "representative_indices": list(REPRESENTATIVE_INDICES),
            "accept_votes": CONSENSUS_VOTES,
            "response_reuse_allowed": False,
        },
        "annotation_unit": "one_label_per_representative_clip",
        "frames_per_representative": 5,
        "views_per_timestamp": list(context.view_order),
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "representative_annotations": annotations,
        "consensus": consensus,
        "status": consensus["status"],
        "phase": consensus["phase"],
        "phrase": consensus["phrase"],
        "human_review_completed": False,
    }


def _adaptive_confidence_tier(
    consensus: Mapping[str, Any],
) -> str:
    if consensus.get("phase") is None:
        return "unresolved"
    dominant_votes = int(consensus["dominant_votes"])
    usable_votes = int(consensus["num_usable_votes"])
    return (
        "majority"
        if dominant_votes * 2 > usable_votes
        else "plurality"
    )


def _adaptive_materialized_row(
    *,
    output_root: Path,
    context: ConditionContext,
    cluster: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    if context.adaptive_config is None:
        raise ValueError("Adaptive context has no extension config")
    config = context.adaptive_config
    cluster_id = str(cluster["cluster_id"])
    target = cluster_id in set(context.target_ids)
    source_row_sha256 = hash_canonical_record(source_row)
    row = dict(source_row)
    row["format"] = config.output_format
    row[config.source_annotation_field] = {
        "path": str(context.clusters_path),
        "row_sha256": source_row_sha256,
        "source_run_contract_sha256": (
            config.source_run_contract_sha256
        ),
        "source_status": source_row.get("status"),
        "source_phase": source_row.get("phase"),
    }
    if not target:
        source_annotations = source_row.get("representative_annotations")
        source_rank = (
            len(source_annotations)
            if isinstance(source_annotations, list)
            else config.source_rank
        )
        row[config.extension_field] = {
            "applied": False,
            "fresh_representative_indices": [],
            "source_rank": source_rank,
            "stopping_rank": source_rank,
            "maximum_rank": config.maximum_rank,
            "resolved": row.get("phase") is not None,
        }
        return row

    annotations = _adaptive_base_annotations(context, cluster_id)
    for checkpoint_indices in config.checkpoints:
        for representative_index in checkpoint_indices:
            annotations.append(
                _success_for_representative(
                    output_root=output_root,
                    context=context,
                    cluster=cluster,
                    representative_index=representative_index,
                )
            )
        annotations.sort(
            key=lambda record: int(record["representative_index"])
        )
        consensus = resolve_adaptive_plurality(
            annotations,
            maximum_rank=config.maximum_rank,
        )
        if consensus["phase"] is not None:
            break
    exhausted = (
        len(annotations) == config.maximum_rank
        and consensus["phase"] is None
    )
    if exhausted:
        consensus = resolve_adaptive_plurality(
            annotations,
            exhausted=True,
            maximum_rank=config.maximum_rank,
        )
    used_count = len(annotations)
    representative_ids = [
        str(value)
        for value in cluster["representative_sample_ids"][:used_count]
    ]
    representative_episodes = [
        int(value)
        for value in cluster["representative_episode_nums"][:used_count]
    ]
    identity = {
        "cluster_id": cluster_id,
        "task_description": str(cluster["task_description"]),
        "representative_sample_ids": representative_ids,
        "representative_episode_nums": representative_episodes,
        "representative_selection_strategy": (
            config.selection_strategy
        ),
        "representative_selection_roles": [
            "centroid" for _ in range(used_count)
        ],
    }
    fresh_indices = list(range(config.source_rank + 1, used_count + 1))
    row.update(
        {
            "representative_set_sha256": hash_representative_set(identity),
            "representative_sample_ids": representative_ids,
            "representative_episode_nums": representative_episodes,
            "representative_selection_strategy": (
                config.selection_strategy
            ),
            "representative_selection_roles": [
                "centroid" for _ in range(used_count)
            ],
            "selection_contract": dict(context.selection_contract),
            "representative_annotations": annotations,
            "consensus": consensus,
            "consensus_config": {
                "policy_id": consensus["policy_id"],
                "initial_representatives": config.source_rank,
                "checkpoints": [
                    indices[-1] for indices in config.checkpoints
                ],
                "representatives_evaluated": used_count,
                "representative_indices": list(
                    range(1, used_count + 1)
                ),
                "minimum_plurality_votes": 2,
                "acceptance_rule": "unique-argmax-at-checkpoint",
                "tie_policy": (
                    "advance-or-preserve-mixed-after-"
                    f"{config.maximum_rank}"
                ),
                "source_response_reuse_allowed": True,
                "fresh_response_reuse_allowed": False,
            },
            "status": consensus["status"],
            "phase": consensus["phase"],
            "phrase": consensus["phrase"],
            "confidence_tier": _adaptive_confidence_tier(consensus),
            "human_review_completed": False,
            config.extension_field: {
                "applied": True,
                "fresh_representative_indices": fresh_indices,
                "source_rank": config.source_rank,
                "stopping_rank": used_count,
                "maximum_rank": config.maximum_rank,
                "resolved": consensus["phase"] is not None,
                "exhausted": exhausted,
                "vote_margin": int(consensus["vote_margin"]),
                "source_row_sha256": source_row_sha256,
            },
        }
    )
    return row


def _adaptive_rows_for_context(
    *,
    output_root: Path,
    context: ConditionContext,
) -> list[dict[str, Any]]:
    if context.base_annotations is None:
        raise ValueError("Adaptive context has no source annotations")
    rows = []
    for source_row in context.base_annotations.values():
        cluster_id = str(source_row["cluster_id"])
        try:
            cluster = context.source_clusters[cluster_id]
        except KeyError as exc:
            raise ValueError(
                f"{context.condition_id}: source cluster is missing "
                f"{cluster_id}"
            ) from exc
        rows.append(
            _adaptive_materialized_row(
                output_root=output_root,
                context=context,
                cluster=cluster,
                source_row=source_row,
            )
        )
    return rows


def _materialize_adaptive(
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[ConditionContext],
) -> dict[str, Any]:
    configs = {
        context.adaptive_config for context in contexts
    }
    if None in configs or len(configs) != 1:
        raise ValueError("Adaptive contexts have inconsistent run config")
    config = next(iter(configs))
    assert config is not None
    outstanding = _adaptive_required_requests(output_root, contexts)
    if outstanding:
        raise RuntimeError(
            "Adaptive annotation still has outstanding requests: "
            f"{len(outstanding)}"
        )
    condition_summaries = []
    all_rows: list[dict[str, Any]] = []
    response_ids: list[str] = []
    for context in contexts:
        rows = _adaptive_rows_for_context(
            output_root=output_root,
            context=context,
        )
        accepted = [row for row in rows if row.get("phase") is not None]
        accepted_path = context.output_path.with_name(
            "accepted_annotations.jsonl"
        )
        _write_or_compare_jsonl(context.output_path, rows)
        _write_or_compare_jsonl(accepted_path, accepted)
        for row in rows:
            response_ids.extend(
                str(annotation["response_id"])
                for annotation in row["representative_annotations"]
            )
        condition_summaries.append(
            {
                "experiment_id": context.experiment_id,
                "condition_id": context.condition_id,
                "output_path": str(context.output_path),
                "accepted_output_path": str(accepted_path),
                "num_rows": len(rows),
                "accepted_rows": len(accepted),
                "extended_rows": sum(
                    bool(row[config.extension_field]["applied"])
                    for row in rows
                ),
                "resolved_extended_rows": sum(
                    bool(row[config.extension_field]["applied"])
                    and row.get("phase") is not None
                    for row in rows
                ),
                "status_counts": dict(
                    sorted(
                        Counter(str(row["status"]) for row in rows).items()
                    )
                ),
            }
        )
        all_rows.extend(rows)
    if len(all_rows) != 90:
        raise ValueError(
            f"Adaptive materialization expected 90 rows, got {len(all_rows)}"
        )
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Adaptive materialization reused a response ID")
    fresh_responses = (
        len(response_ids) - config.source_provider_response_count
    )
    if fresh_responses < 0:
        raise ValueError("Adaptive response count is below its source")
    result = {
        "format": config.summary_format,
        "run_contract_sha256": manifest["contract_sha256"],
        "completed_at_utc": utc_now(),
        "total_rows": len(all_rows),
        "accepted_rows": sum(
            row.get("phase") is not None for row in all_rows
        ),
        "unresolved_rows": sum(
            row.get("phase") is None for row in all_rows
        ),
        "source_provider_responses": (
            config.source_provider_response_count
        ),
        "fresh_provider_responses": fresh_responses,
        "unique_response_ids": len(response_ids),
        "conditions": condition_summaries,
    }
    summary_path = output_root / "run_summary.json"
    if summary_path.exists():
        existing = load_json_object(summary_path)
        comparable_existing = dict(existing)
        comparable_result = dict(result)
        comparable_existing.pop("completed_at_utc", None)
        comparable_result.pop("completed_at_utc", None)
        if comparable_existing != comparable_result:
            raise FileExistsError(
                f"Adaptive summary is incompatible: {summary_path}"
            )
        return existing
    write_json_exclusive(summary_path, result)
    return result


def materialize(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest = freeze_or_validate_manifest(output_root)
    contexts = load_contexts(manifest)
    if manifest["contract"].get("run_kind") == "adaptive-plurality":
        return _materialize_adaptive(output_root, manifest, contexts)
    condition_summaries = []
    response_ids: list[str] = []
    for context in contexts:
        rows = [
            _materialized_row(
                output_root=output_root,
                context=context,
                cluster=cluster,
            )
            for cluster in context.targeted_rows
        ]
        if context.output_path.exists():
            if load_jsonl(context.output_path) != rows:
                raise FileExistsError(
                    f"Refusing to replace divergent output: "
                    f"{context.output_path}"
                )
        else:
            _atomic_write_jsonl(context.output_path, rows)
        for row in rows:
            response_ids.extend(
                str(annotation["response_id"])
                for annotation in row["representative_annotations"]
            )
        condition_summaries.append(
            {
                "experiment_id": context.experiment_id,
                "condition_id": context.condition_id,
                "output_path": str(context.output_path),
                "num_rows": len(rows),
                "logical_requests": 5 * len(rows),
                "status_counts": dict(
                    sorted(
                        Counter(
                            str(row["status"]) for row in rows
                        ).items()
                    )
                ),
            }
        )
    if len(response_ids) != 450 or len(set(response_ids)) != 450:
        raise ValueError(
            "Fresh centroid run requires 450 unique provider response IDs"
        )
    result = {
        "format": SUMMARY_FORMAT,
        "run_contract_sha256": manifest["contract_sha256"],
        "completed_at_utc": utc_now(),
        "total_rows": sum(
            int(summary["num_rows"])
            for summary in condition_summaries
        ),
        "logical_requests": len(response_ids),
        "unique_response_ids": len(set(response_ids)),
        "conditions": condition_summaries,
    }
    summary_path = output_root / "run_summary.json"
    if summary_path.exists():
        existing = load_json_object(summary_path)
        immutable = {
            key: value
            for key, value in result.items()
            if key != "completed_at_utc"
        }
        existing_immutable = {
            key: value
            for key, value in existing.items()
            if key != "completed_at_utc"
        }
        if existing_immutable != immutable:
            raise FileExistsError(
                f"Existing summary is incompatible: {summary_path}"
            )
        return existing
    write_json_exclusive(summary_path, result)
    return result


def _audit_adaptive_results(
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[ConditionContext],
) -> dict[str, Any]:
    configs = {
        context.adaptive_config for context in contexts
    }
    if None in configs or len(configs) != 1:
        raise ValueError("Adaptive contexts have inconsistent run config")
    config = next(iter(configs))
    assert config is not None
    if _adaptive_required_requests(output_root, contexts):
        raise ValueError("Adaptive annotation is not terminal")
    total_rows = 0
    accepted_rows = 0
    unresolved_rows = 0
    fresh_response_ids: list[str] = []
    all_response_ids: list[str] = []
    for context in contexts:
        expected = _adaptive_rows_for_context(
            output_root=output_root,
            context=context,
        )
        actual = load_jsonl(context.output_path)
        if actual != expected:
            raise ValueError(
                f"Adaptive rows drifted: {context.condition_id}"
            )
        expected_accepted = [
            row for row in expected if row.get("phase") is not None
        ]
        accepted_path = context.output_path.with_name(
            "accepted_annotations.jsonl"
        )
        if load_jsonl(accepted_path) != expected_accepted:
            raise ValueError(
                f"Adaptive accepted rows drifted: {context.condition_id}"
            )
        total_rows += len(expected)
        accepted_rows += len(expected_accepted)
        unresolved_rows += len(expected) - len(expected_accepted)
        for row in expected:
            annotations = row["representative_annotations"]
            all_response_ids.extend(
                str(annotation["response_id"])
                for annotation in annotations
            )
            for annotation in annotations[config.source_rank:]:
                fresh_response_ids.append(str(annotation["response_id"]))
    if len(all_response_ids) != len(set(all_response_ids)):
        raise ValueError("Adaptive response IDs are not unique")
    if len(fresh_response_ids) != len(set(fresh_response_ids)):
        raise ValueError("Adaptive fresh response IDs are not unique")
    summary = load_json_object(output_root / "run_summary.json")
    expected_summary_fields = {
        "format": config.summary_format,
        "run_contract_sha256": manifest["contract_sha256"],
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "unresolved_rows": unresolved_rows,
        "fresh_provider_responses": len(fresh_response_ids),
    }
    mismatched = [
        field
        for field, value in expected_summary_fields.items()
        if summary.get(field) != value
    ]
    if mismatched:
        raise ValueError(
            f"Adaptive summary drifted: {mismatched}"
        )
    return {
        "format": config.audit_format,
        "run_contract_sha256": manifest["contract_sha256"],
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "unresolved_rows": unresolved_rows,
        "fresh_provider_responses": len(fresh_response_ids),
        "complete": True,
    }


def audit_results(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest = freeze_or_validate_manifest(output_root)
    contexts = load_contexts(manifest)
    if manifest["contract"].get("run_kind") == "adaptive-plurality":
        return _audit_adaptive_results(
            output_root,
            manifest,
            contexts,
        )
    response_ids: list[str] = []
    condition_summaries: list[dict[str, Any]] = []
    total_rows = 0
    for context in contexts:
        if not context.output_path.is_file():
            raise FileNotFoundError(context.output_path)
        actual = load_jsonl(context.output_path)
        expected = [
            _materialized_row(
                output_root=output_root,
                context=context,
                cluster=cluster,
            )
            for cluster in context.targeted_rows
        ]
        if actual != expected:
            raise ValueError(
                f"Materialized output does not replay: "
                f"{context.output_path}"
            )
        if context.output_path.read_bytes() != _jsonl_bytes(expected):
            raise ValueError(
                f"Materialized bytes are not canonical: "
                f"{context.output_path}"
            )
        total_rows += len(expected)
        condition_summaries.append(
            {
                "experiment_id": context.experiment_id,
                "condition_id": context.condition_id,
                "output_path": str(context.output_path),
                "num_rows": len(expected),
                "logical_requests": 5 * len(expected),
                "status_counts": dict(
                    sorted(
                        Counter(
                            str(row["status"]) for row in expected
                        ).items()
                    )
                ),
            }
        )
        response_ids.extend(
            str(annotation["response_id"])
            for row in expected
            for annotation in row["representative_annotations"]
        )
    if total_rows != 90:
        raise ValueError(f"Expected 90 rows, found {total_rows}")
    if len(response_ids) != 450 or len(set(response_ids)) != 450:
        raise ValueError("Expected 450 unique fresh responses")
    summary_path = output_root / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json_object(summary_path)
    expected_summary = {
        "format": SUMMARY_FORMAT,
        "run_contract_sha256": manifest["contract_sha256"],
        "total_rows": total_rows,
        "logical_requests": len(response_ids),
        "unique_response_ids": len(set(response_ids)),
        "conditions": condition_summaries,
    }
    comparable_summary = {
        key: value
        for key, value in summary.items()
        if key != "completed_at_utc"
    }
    if comparable_summary != expected_summary:
        raise ValueError(f"Run summary does not replay: {summary_path}")
    return {
        "format": "event_sae_centroid_nearest_five_results_audit_v1",
        "audited_at_utc": utc_now(),
        "run_contract_sha256": manifest["contract_sha256"],
        "total_rows": total_rows,
        "logical_requests": len(response_ids),
        "unique_response_ids": len(set(response_ids)),
        "complete": True,
    }


def _confidence_tier(consensus: Mapping[str, Any]) -> str:
    if consensus.get("phase") is None:
        return "unresolved"
    votes = int(consensus["dominant_votes"])
    if votes >= 4:
        return "strong"
    if votes == 3:
        return "majority"
    return "plurality"


def _phase_override_source_files(
    source_root: Path,
    condition_paths: Sequence[tuple[str, Path]],
) -> dict[str, str]:
    source_paths = [
        source_root / "run_manifest.json",
        source_root / "run_summary.json",
    ]
    for _, annotations_path in condition_paths:
        source_paths.extend(
            (
                annotations_path,
                annotations_path.with_name("accepted_annotations.jsonl"),
            )
        )
    return {
        str(path.relative_to(source_root)): _sha256_file(path)
        for path in source_paths
    }


def _excluded_continuation_provenance(
    root: Path | None,
) -> dict[str, Any] | None:
    if root is None:
        return None
    root = Path(root).resolve()
    manifest = load_json_object(root / "run_manifest.json")
    contract = manifest.get("contract")
    if (
        not isinstance(contract, Mapping)
        or manifest.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("Excluded continuation manifest drifted")
    artifact_paths = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
    )
    attempts = [
        load_json_object(path)
        for path in artifact_paths
        if path.name.startswith("attempt-")
    ]
    status_counts = dict(
        sorted(Counter(str(row["status"]) for row in attempts).items())
    )
    successful_indices = sorted(
        int(row["representative_index"])
        for row in attempts
        if row.get("status") == "success"
    )
    cancelled_indices = sorted(
        int(row["representative_index"])
        for row in attempts
        if (
            row.get("status") == "request_error"
            and "cancel" in str(row.get("error", "")).lower()
        )
    )
    return {
        "root": str(root),
        "run_contract_sha256": str(manifest["contract_sha256"]),
        "used_in_phase_decision": False,
        "attempt_status_counts": status_counts,
        "successful_but_excluded_representative_indices": (
            successful_indices
        ),
        "provider_cancelled_representative_indices": cancelled_indices,
        "files_sha256": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in artifact_paths
        },
    }


def _apply_user_phase_override(
    source_row: Mapping[str, Any],
    *,
    source_run_contract_sha256: str,
    decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = dict(source_row)
    source_row_sha256 = hash_canonical_record(source_row)
    row["format"] = "event_sae_user_phase_override_annotation_v1"
    row["actual_human_review_completed"] = False
    if decision is None:
        row["phase_source"] = "source_annotation"
        row["phase_override"] = {
            "applied": False,
            "source_run_contract_sha256": source_run_contract_sha256,
            "source_row_sha256": source_row_sha256,
        }
        return row

    if source_row.get("phase") is not None:
        raise ValueError(
            "User phase override is restricted to an unresolved source row"
        )
    phase = str(decision["phase"]).strip()
    phrase = str(decision["phrase"]).strip()
    if not phase or not phrase:
        raise ValueError("User phase override needs a phase and phrase")
    allowed = {
        str(value) for value in source_row.get("allowed_phase_labels", [])
    }
    if allowed and phase not in allowed:
        raise ValueError(
            f"Override phase {phase!r} is outside the row vocabulary"
        )
    annotations = source_row.get("representative_annotations")
    if not isinstance(annotations, list):
        raise ValueError("Override source has no representative annotations")
    phrase_source_indices = [
        int(annotation["representative_index"])
        for annotation in annotations
        if (
            annotation.get("phase") == phase
            and annotation.get("phrase") == phrase
        )
    ]
    if not phrase_source_indices:
        raise ValueError(
            "Override phrase must be supported by an existing matching vote"
        )
    row.update(
        {
            "status": "user-directed-phase-override",
            "phase": phase,
            "phrase": phrase,
            "confidence_tier": "user-directed",
            "phase_source": "explicit_user_analysis_override",
            "human_review_completed": False,
            "actual_human_review_completed": False,
            "phase_override": {
                "applied": True,
                "authorized_by": str(decision["authorized_by"]),
                "reason": str(decision["reason"]),
                "phase": phase,
                "phrase": phrase,
                "phrase_source": "existing_matching_representative_votes",
                "phrase_source_representative_indices": (
                    phrase_source_indices
                ),
                "provider_response_generated": False,
                "formal_blind_review_completed": False,
                "source_run_contract_sha256": (
                    source_run_contract_sha256
                ),
                "source_row_sha256": source_row_sha256,
                "source_status": source_row.get("status"),
                "source_phase": source_row.get("phase"),
                "source_consensus": source_row.get("consensus"),
            },
        }
    )
    return row


def _phase_override_rows(
    *,
    condition_id: str,
    source_path: Path,
    source_run_contract_sha256: str,
    decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_condition_id = str(decision["condition_id"])
    target_cluster_id = str(decision["cluster_id"])
    rows = []
    matches = 0
    for source_row in load_jsonl(source_path):
        applies = (
            condition_id == target_condition_id
            and str(source_row["cluster_id"]) == target_cluster_id
        )
        matches += int(applies)
        rows.append(
            _apply_user_phase_override(
                source_row,
                source_run_contract_sha256=source_run_contract_sha256,
                decision=decision if applies else None,
            )
        )
    if condition_id == target_condition_id and matches != 1:
        raise ValueError(
            f"Expected one override target in {condition_id}, got {matches}"
        )
    return rows


def derive_user_phase_override(
    *,
    source_root: Path,
    output_root: Path,
    condition_id: str,
    cluster_id: str,
    phase: str,
    phrase: str,
    reason: str,
    excluded_run_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize one explicit user-directed phase override offline."""

    source_root = Path(source_root).resolve()
    output_root = _validate_derivation_output_root(
        source_root,
        output_root,
    )
    source_contract_sha256, condition_paths = _source_condition_paths(
        source_root
    )
    decision = {
        "condition_id": str(condition_id),
        "cluster_id": str(cluster_id),
        "phase": str(phase),
        "phrase": str(phrase),
        "authorized_by": "workspace_owner",
        "reason": str(reason),
        "provider_response_generated": False,
        "formal_blind_review_completed": False,
    }
    excluded_continuation = _excluded_continuation_provenance(
        excluded_run_root
    )
    contract = {
        "format": "event_sae_user_phase_override_contract_v1",
        "source_root": str(source_root),
        "source_run_contract_sha256": source_contract_sha256,
        "source_files_sha256": _phase_override_source_files(
            source_root,
            condition_paths,
        ),
        "output_root": str(output_root),
        "decision": decision,
        "excluded_continuation": excluded_continuation,
    }
    contract_sha256 = canonical_sha256(contract)
    manifest = {
        "format": "event_sae_user_phase_override_manifest_v1",
        "contract_sha256": contract_sha256,
        "contract": contract,
    }
    _write_or_compare_json(
        output_root / "derivation_manifest.json",
        manifest,
    )

    all_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for source_condition_id, source_path in condition_paths:
        rows = _phase_override_rows(
            condition_id=source_condition_id,
            source_path=source_path,
            source_run_contract_sha256=source_contract_sha256,
            decision=decision,
        )
        accepted = [row for row in rows if row.get("phase") is not None]
        output_path = (
            output_root / source_condition_id / "annotations.jsonl"
        )
        accepted_path = output_path.with_name("accepted_annotations.jsonl")
        _write_or_compare_jsonl(output_path, rows)
        _write_or_compare_jsonl(accepted_path, accepted)
        condition_summaries.append(
            {
                "condition_id": source_condition_id,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "accepted_output_path": str(accepted_path),
                "num_rows": len(rows),
                "accepted_rows": len(accepted),
                "overridden_rows": sum(
                    bool(row["phase_override"]["applied"])
                    for row in rows
                ),
                "status_counts": dict(
                    sorted(Counter(str(row["status"]) for row in rows).items())
                ),
            }
        )
        all_rows.extend(rows)
    if (
        len(all_rows) != 90
        or sum(row.get("phase") is not None for row in all_rows) != 90
        or sum(
            bool(row["phase_override"]["applied"]) for row in all_rows
        )
        != 1
    ):
        raise ValueError("User phase override did not produce 90/90 rows")
    summary = {
        "format": "event_sae_user_phase_override_summary_v1",
        "derivation_contract_sha256": contract_sha256,
        "source_run_contract_sha256": source_contract_sha256,
        "total_rows": 90,
        "accepted_rows": 90,
        "overridden_rows": 1,
        "actual_human_review_completed": False,
        "conditions": condition_summaries,
    }
    _write_or_compare_json(output_root / "run_summary.json", summary)
    return summary


def audit_user_phase_override(output_root: Path) -> dict[str, Any]:
    """Replay a user phase override and verify its immutable source hashes."""

    output_root = Path(output_root).resolve()
    manifest = load_json_object(output_root / "derivation_manifest.json")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Phase-override manifest has no contract")
    contract_sha256 = canonical_sha256(contract)
    if manifest.get("contract_sha256") != contract_sha256:
        raise ValueError("Phase-override manifest self-hash drifted")
    source_root = Path(str(contract["source_root"])).resolve()
    for relative_path, expected_sha256 in contract[
        "source_files_sha256"
    ].items():
        if _sha256_file(source_root / relative_path) != expected_sha256:
            raise ValueError(
                f"Phase-override source drifted: {relative_path}"
            )
    excluded = contract.get("excluded_continuation")
    excluded_root = (
        None
        if excluded is None
        else Path(str(excluded["root"])).resolve()
    )
    if _excluded_continuation_provenance(excluded_root) != excluded:
        raise ValueError("Excluded continuation provenance drifted")

    source_contract_sha256, condition_paths = _source_condition_paths(
        source_root
    )
    if source_contract_sha256 != contract["source_run_contract_sha256"]:
        raise ValueError("Phase-override source contract drifted")
    decision = contract["decision"]
    all_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for condition_id, source_path in condition_paths:
        expected = _phase_override_rows(
            condition_id=condition_id,
            source_path=source_path,
            source_run_contract_sha256=source_contract_sha256,
            decision=decision,
        )
        output_path = output_root / condition_id / "annotations.jsonl"
        actual = load_jsonl(output_path)
        if actual != expected:
            raise ValueError(f"Phase-override rows drifted: {condition_id}")
        expected_accepted = [
            row for row in expected if row.get("phase") is not None
        ]
        accepted_path = output_path.with_name("accepted_annotations.jsonl")
        if load_jsonl(accepted_path) != expected_accepted:
            raise ValueError(
                f"Phase-override accepted rows drifted: {condition_id}"
            )
        condition_summaries.append(
            {
                "condition_id": condition_id,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "accepted_output_path": str(accepted_path),
                "num_rows": len(expected),
                "accepted_rows": len(expected_accepted),
                "overridden_rows": sum(
                    bool(row["phase_override"]["applied"])
                    for row in expected
                ),
                "status_counts": dict(
                    sorted(
                        Counter(str(row["status"]) for row in expected).items()
                    )
                ),
            }
        )
        all_rows.extend(expected)
    expected_summary = {
        "format": "event_sae_user_phase_override_summary_v1",
        "derivation_contract_sha256": contract_sha256,
        "source_run_contract_sha256": source_contract_sha256,
        "total_rows": 90,
        "accepted_rows": 90,
        "overridden_rows": 1,
        "actual_human_review_completed": False,
        "conditions": condition_summaries,
    }
    if load_json_object(output_root / "run_summary.json") != expected_summary:
        raise ValueError("Phase-override summary drifted")
    if (
        len(all_rows) != 90
        or sum(row.get("phase") is not None for row in all_rows) != 90
        or sum(
            bool(row["phase_override"]["applied"]) for row in all_rows
        )
        != 1
    ):
        raise ValueError("Phase-override inventory drifted")
    return {
        "format": "event_sae_user_phase_override_audit_v1",
        "derivation_contract_sha256": contract_sha256,
        "total_rows": 90,
        "accepted_rows": 90,
        "overridden_rows": 1,
        "complete": True,
    }


def derive_consensus_row(
    source_row: Mapping[str, Any],
    *,
    policy: ConsensusPolicy,
    source_run_contract_sha256: str,
) -> dict[str, Any]:
    """Apply one offline policy without changing the provider responses."""

    annotations = source_row.get("representative_annotations")
    if not isinstance(annotations, list):
        raise ValueError("Source row has no representative annotations")
    for annotation in annotations:
        validate_representative_annotation(
            annotation,
            task_description=str(source_row["task_description"]),
        )
    consensus = resolve_representative_consensus(annotations, policy)
    row = dict(source_row)
    source_consensus = row.get("consensus")
    row.update(
        {
            "format": "event_sae_consensus_derivation_v2",
            "source_annotation": {
                "format": source_row.get("format"),
                "run_contract_sha256": source_run_contract_sha256,
                "row_sha256": hash_canonical_record(source_row),
                "consensus": source_consensus,
                "status": source_row.get("status"),
                "phase": source_row.get("phase"),
                "phrase": source_row.get("phrase"),
            },
            "consensus_config": {
                "policy_id": policy.policy_id,
                "representatives_evaluated": 5,
                "representative_indices": list(REPRESENTATIVE_INDICES),
                "minimum_votes": policy.minimum_votes,
                "require_unique_winner": policy.require_unique_winner,
                "response_reuse_allowed": False,
            },
            "consensus": consensus,
            "status": consensus["status"],
            "phase": consensus["phase"],
            "phrase": consensus["phrase"],
            "confidence_tier": _confidence_tier(consensus),
        }
    )
    return row


def _source_condition_paths(
    source_root: Path,
) -> tuple[str, list[tuple[str, Path]]]:
    source_root = Path(source_root).resolve()
    manifest = load_json_object(source_root / "run_manifest.json")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Source manifest has no contract")
    if manifest.get("contract_sha256") != canonical_sha256(contract):
        raise ValueError("Source manifest self-hash drifted")
    summary = load_json_object(source_root / "run_summary.json")
    conditions = summary.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Source summary has no conditions")
    result: list[tuple[str, Path]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        path = Path(str(condition["output_path"])).resolve()
        if source_root not in path.parents or path.parent.name != condition_id:
            raise ValueError(f"Source condition escaped its root: {path}")
        result.append((condition_id, path))
    return str(manifest["contract_sha256"]), result


def _validate_derivation_output_root(
    source_root: Path,
    output_root: Path,
) -> Path:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    allowed_parent = EXPERIMENT_ROOT.resolve()
    temporary_root = Path("/tmp").resolve()
    if (
        allowed_parent not in output_root.parents
        and temporary_root not in output_root.parents
    ):
        raise ValueError(
            "Consensus output must be under the experiment root or /tmp"
        )
    protected = (
        source_root,
        FROZEN_INTERACTIVE_ROOT.resolve(),
        LEGACY_MEDIA_ROOT.resolve(),
        CENTROID_DIVERSITY_REPRESENTATIVE_ROOT.resolve(),
        *(root.resolve() for root in HISTORICAL_ANNOTATION_ROOTS),
    )
    for root in protected:
        if (
            output_root == root
            or output_root in root.parents
            or root in output_root.parents
        ):
            raise ValueError(
                "Consensus output must be disjoint from source/historical "
                f"artifacts: output={output_root}, protected={root}"
            )
    return output_root


def _write_or_compare_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    if path.exists():
        if load_json_object(path) != dict(value):
            raise FileExistsError(f"Refusing to replace divergent file: {path}")
        return
    write_json_exclusive(path, value)


def _write_or_compare_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = [dict(row) for row in rows]
    if path.exists():
        if load_jsonl(path) != expected:
            raise FileExistsError(f"Refusing to replace divergent file: {path}")
        return
    _atomic_write_jsonl(path, expected)


def derive_consensus_annotations(
    *,
    source_root: Path,
    output_root: Path,
    policy: ConsensusPolicy = UNIQUE_PLURALITY,
) -> dict[str, Any]:
    """Materialize a new policy under an explicit, disjoint output root."""

    source_root = Path(source_root).resolve()
    output_root = _validate_derivation_output_root(
        source_root,
        output_root,
    )
    source_contract_sha256, condition_paths = _source_condition_paths(
        source_root
    )
    source_files = {
        str(path.relative_to(source_root)): _sha256_file(path)
        for path in (
            source_root / "run_manifest.json",
            source_root / "run_summary.json",
            *(path for _, path in condition_paths),
        )
    }
    contract = {
        "format": "event_sae_consensus_derivation_contract_v2",
        "source_root": str(source_root),
        "source_run_contract_sha256": source_contract_sha256,
        "source_files_sha256": dict(sorted(source_files.items())),
        "output_root": str(output_root),
        "policy": {
            "policy_id": policy.policy_id,
            "minimum_votes": policy.minimum_votes,
            "require_unique_winner": policy.require_unique_winner,
        },
    }
    contract_sha256 = canonical_sha256(contract)
    manifest = {
        "format": "event_sae_consensus_derivation_manifest_v2",
        "contract_sha256": contract_sha256,
        "contract": contract,
    }
    _write_or_compare_json(output_root / "derivation_manifest.json", manifest)

    condition_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for condition_id, source_path in condition_paths:
        rows = [
            derive_consensus_row(
                source_row,
                policy=policy,
                source_run_contract_sha256=source_contract_sha256,
            )
            for source_row in load_jsonl(source_path)
        ]
        accepted = [row for row in rows if row.get("phase") is not None]
        output_path = output_root / condition_id / "annotations.jsonl"
        accepted_path = (
            output_root / condition_id / "accepted_annotations.jsonl"
        )
        _write_or_compare_jsonl(output_path, rows)
        _write_or_compare_jsonl(accepted_path, accepted)
        condition_summaries.append(
            {
                "condition_id": condition_id,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "accepted_output_path": str(accepted_path),
                "num_rows": len(rows),
                "accepted_rows": len(accepted),
                "status_counts": dict(
                    sorted(Counter(str(row["status"]) for row in rows).items())
                ),
                "confidence_tier_counts": dict(
                    sorted(
                        Counter(
                            str(row["confidence_tier"]) for row in rows
                        ).items()
                    )
                ),
            }
        )
        all_rows.extend(rows)
    summary = {
        "format": "event_sae_consensus_derivation_summary_v2",
        "derivation_contract_sha256": contract_sha256,
        "source_run_contract_sha256": source_contract_sha256,
        "policy_id": policy.policy_id,
        "total_rows": len(all_rows),
        "accepted_rows": sum(
            row.get("phase") is not None for row in all_rows
        ),
        "conditions": condition_summaries,
    }
    _write_or_compare_json(output_root / "run_summary.json", summary)
    return summary


def audit_consensus_derivation(output_root: Path) -> dict[str, Any]:
    """Replay a v2 derivation without writing to its source or output root."""

    output_root = Path(output_root).resolve()
    manifest = load_json_object(output_root / "derivation_manifest.json")
    contract = manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Derivation manifest has no contract")
    contract_sha256 = canonical_sha256(contract)
    if manifest.get("contract_sha256") != contract_sha256:
        raise ValueError("Derivation manifest self-hash drifted")
    source_root = Path(str(contract["source_root"])).resolve()
    for relative_path, expected_sha256 in contract[
        "source_files_sha256"
    ].items():
        if _sha256_file(source_root / relative_path) != expected_sha256:
            raise ValueError(f"Derivation source drifted: {relative_path}")
    policy_contract = contract["policy"]
    policy = ConsensusPolicy(
        policy_id=str(policy_contract["policy_id"]),
        minimum_votes=int(policy_contract["minimum_votes"]),
        require_unique_winner=bool(
            policy_contract["require_unique_winner"]
        ),
    )
    source_contract_sha256, condition_paths = _source_condition_paths(
        source_root
    )
    total_rows = 0
    accepted_rows = 0
    for condition_id, source_path in condition_paths:
        expected = [
            derive_consensus_row(
                source_row,
                policy=policy,
                source_run_contract_sha256=source_contract_sha256,
            )
            for source_row in load_jsonl(source_path)
        ]
        actual = load_jsonl(
            output_root / condition_id / "annotations.jsonl"
        )
        accepted = load_jsonl(
            output_root / condition_id / "accepted_annotations.jsonl"
        )
        if actual != expected:
            raise ValueError(f"Derived rows drifted: {condition_id}")
        expected_accepted = [
            row for row in expected if row.get("phase") is not None
        ]
        if accepted != expected_accepted:
            raise ValueError(f"Accepted rows drifted: {condition_id}")
        total_rows += len(expected)
        accepted_rows += len(expected_accepted)
    summary = load_json_object(output_root / "run_summary.json")
    if (
        summary.get("derivation_contract_sha256") != contract_sha256
        or int(summary.get("total_rows", -1)) != total_rows
        or int(summary.get("accepted_rows", -1)) != accepted_rows
    ):
        raise ValueError("Derivation summary drifted")
    return {
        "format": "event_sae_consensus_derivation_audit_v2",
        "derivation_contract_sha256": contract_sha256,
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "complete": True,
    }


__all__ = [
    "ADAPTIVE_OUTPUT_ROOT",
    "CONSENSUS_POLICIES",
    "CONTINUATION_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_MODEL_VERSION",
    "FIXED_STRONG_CONSENSUS",
    "MAX_REQUEST_ATTEMPTS",
    "MAX_SERIALIZED_CHUNK_BYTES",
    "MODEL",
    "STRICT_MAJORITY",
    "UNIQUE_PLURALITY",
    "ConsensusPolicy",
    "annotation_request_id",
    "atomic_write_json",
    "audit_consensus_derivation",
    "audit_results",
    "audit_user_phase_override",
    "batch_wave",
    "build_inline_attempt_record",
    "derive_consensus_annotations",
    "derive_consensus_row",
    "derive_user_phase_override",
    "exclusive_run_lock",
    "expected_logical_request_ids",
    "fixed_five_consensus",
    "freeze_or_validate_manifest",
    "canonical_sha256",
    "load_attempts",
    "load_contexts",
    "load_json_object",
    "materialize",
    "normalize_response_text",
    "record_job_failure",
    "requestable_representative_indices",
    "resolve_adaptive_plurality",
    "resolve_representative_consensus",
    "required_requests",
    "select_centroid_representatives",
    "serialize_batch_payload",
    "serialize_inlined_request",
    "sha256_bytes",
    "successful_attempt",
    "utc_now",
    "validate_plan_entry",
    "write_json_exclusive",
    "write_attempt_record",
]
