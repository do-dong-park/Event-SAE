"""Read-only simulator-Oracle result validation and catalog building."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from event_sae.events.io import load_jsonl
from event_sae.groot.phase_feature_results import (
    RESULT_RANKINGS,
    compact_ranking_row,
    format_checkpoint_label,
)


ORACLE_PHASE_VIEWS = ("left", "right", "wrist")


ORACLE_PHASE_BROWSER_FORMAT = "event_sae_oracle_phase_results_browser_v1"


ORACLE_PHASE_CLAIM_SCOPE = "simulator-oracle diagnostic upper bound"


def build_oracle_waiting_payload(
    *,
    status: str,
    message: str,
) -> dict:
    """Return a stable empty state while an oracle experiment is in flight."""

    return {
        "format": ORACLE_PHASE_BROWSER_FORMAT,
        "available": False,
        "status": status,
        "message": message,
        "claim_scope": ORACLE_PHASE_CLAIM_SCOPE,
        "annotation_mode": "programmatic_oracle_no_vlm",
        "human_review_completed": False,
        "meta": {
            "num_tasks": 0,
            "num_episodes": 0,
            "num_keyframes": 0,
            "num_phases": 0,
            "num_variants": 0,
            "num_ranking_packages": 0,
            "num_discovered_ranking_packages": 0,
            "num_unmapped_ranking_packages": 0,
            "num_ranking_checkpoints": 0,
            "num_encoded_checkpoints": 0,
        },
        "stages": [
            {
                "id": "trajectory",
                "label": "입력 궤적",
                "state": "waiting",
                "detail": "실험 대상 episode 목록 대기",
            },
            {
                "id": "keyframes",
                "label": "Oracle phase keyframe",
                "state": "waiting",
                "detail": "시뮬레이터 phase 전환점을 기다리는 중",
            },
            {
                "id": "media",
                "label": "3-view 영상",
                "state": "waiting",
                "detail": "LEFT · RIGHT · WRIST 5-frame 묶음 대기",
            },
            {
                "id": "features",
                "label": "3-view SigLIP",
                "state": "waiting",
                "detail": "영상 descriptor 생성 대기",
            },
            {
                "id": "clustering",
                "label": "Phase 안 상태 군집",
                "state": "waiting",
                "detail": "Oracle phase별 SigLIP 군집 대기",
            },
            {
                "id": "ranking",
                "label": "SAE feature ranking",
                "state": "waiting",
                "detail": "3개 checkpoint 결과 대기",
            },
        ],
        "views": [],
        "phases": [],
        "variants": [],
        "default_variant_id": None,
        "ranking_checkpoints": [],
        "default_checkpoint_id": None,
        "phase_groups": [],
        "ranking_packages": [],
        "annotation": {
            "label_source": "env_step_phases",
            "mode": "programmatic_oracle_no_vlm",
            "review_status": "not_applicable",
            "oracle_upper_bound": True,
        },
        "alignment": {},
        "audit_gates": [],
    }


def _oracle_resolved_frame_path(
    *,
    experiment_root: Path,
    raw_path: object,
    label: str,
) -> Path:
    frame_path = Path(str(raw_path)).expanduser().resolve()
    if (
        frame_path != experiment_root
        and experiment_root not in frame_path.parents
    ):
        raise ValueError(f"{label} escapes the oracle experiment root")
    if not frame_path.is_file():
        raise FileNotFoundError(f"{label} not found: {frame_path}")
    return frame_path


def _oracle_checkpoint_metadata(
    *,
    experiment_root: Path,
    checkpoint_source: str,
    config: dict | None = None,
) -> dict:
    """Resolve one SAE checkpoint identity from a sparse Top-K artifact."""

    config = config or {}
    checkpoint_source_path = (
        Path(checkpoint_source).expanduser().resolve()
        if checkpoint_source
        else None
    )
    topk_manifest: dict = {}
    if checkpoint_source_path is not None:
        topk_manifest_path = checkpoint_source_path / "manifest.json"
        if (
            topk_manifest_path.is_file()
            and experiment_root in topk_manifest_path.parents
        ):
            topk_manifest = json.loads(
                topk_manifest_path.read_text(encoding="utf-8")
            )

    checkpoint_id = str(config.get("checkpoint_id", "")).strip()
    if not checkpoint_id:
        checkpoint_manifest = str(
            topk_manifest.get("checkpoint_source_manifest", "")
        ).strip()
        if checkpoint_manifest:
            checkpoint_id = Path(checkpoint_manifest).parent.name
    if not checkpoint_id:
        sae_path = str(topk_manifest.get("sae_path", "")).strip()
        if sae_path:
            checkpoint_id = Path(sae_path).parent.parent.name
    if not checkpoint_id:
        checkpoint_id = (
            checkpoint_source_path.parent.name
            if checkpoint_source_path is not None
            else "unknown_checkpoint"
        )

    checkpoint_batch_size = None
    checkpoint_training_steps = None
    checkpoint_match = re.search(
        r"bs(?P<batch>\d+)_steps(?P<steps>\d+)",
        checkpoint_id,
    )
    if checkpoint_match:
        checkpoint_batch_size = int(checkpoint_match.group("batch"))
        checkpoint_training_steps = int(checkpoint_match.group("steps"))
    checkpoint_label = str(config.get("checkpoint_label", "")).strip()
    if not checkpoint_label and checkpoint_match:
        steps = int(checkpoint_match.group("steps"))
        step_label = f"{steps / 1000:g}k" if steps >= 1000 else str(steps)
        checkpoint_label = (
            f"SAE {step_label} · bs{checkpoint_match.group('batch')}"
        )
    if not checkpoint_label:
        checkpoint_label = format_checkpoint_label(checkpoint_id)

    encoding_stats = topk_manifest.get("encoding_stats") or {}
    is_reference = bool(
        checkpoint_source_path is not None
        and checkpoint_source_path
        == (experiment_root / "activations/topk96").resolve()
    )
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_label": checkpoint_label,
        "checkpoint_batch_size": checkpoint_batch_size,
        "checkpoint_training_steps": checkpoint_training_steps,
        "checkpoint_sae_sha256": (
            str(topk_manifest["sae_sha256"])
            if topk_manifest.get("sae_sha256")
            else None
        ),
        "checkpoint_dict_size": (
            int(topk_manifest["dict_size"])
            if topk_manifest.get("dict_size") is not None
            else None
        ),
        "checkpoint_topk": (
            int(topk_manifest["topk"])
            if topk_manifest.get("topk") is not None
            else None
        ),
        "checkpoint_total_rows": (
            int(topk_manifest["total_rows"])
            if topk_manifest.get("total_rows") is not None
            else None
        ),
        "checkpoint_lossless_topk": (
            bool(encoding_stats["lossless_topk"])
            if encoding_stats.get("lossless_topk") is not None
            else None
        ),
        "checkpoint_topk_run_id": (
            checkpoint_source_path.name
            if checkpoint_source_path is not None
            else None
        ),
        "checkpoint_is_reference": is_reference,
        "checkpoint_encoded": (
            topk_manifest.get("format") == "token_topk_sparse_v1"
        ),
    }


def _oracle_score_family_id(score_run_id: str) -> str:
    """Remove a checkpoint suffix while preserving the analysis window."""

    return re.sub(
        r"_sae[a-z0-9]+(?:_v\d+)?$",
        "",
        score_run_id,
        flags=re.IGNORECASE,
    )


def _oracle_ranking_packages(experiment_root: Path) -> list[dict]:
    """Load compact, completed ranking packages without touching score tensors."""

    packages: list[dict] = []
    for config_path in sorted(experiment_root.rglob("ranking_config.json")):
        ranking_dir = config_path.parent
        ranking_paths = {
            ranking: ranking_dir / f"{ranking}.jsonl"
            for ranking in RESULT_RANKINGS
        }
        if not all(path.is_file() for path in ranking_paths.values()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows = {
            ranking: [
                compact_ranking_row(row)
                for row in load_jsonl(path)
            ]
            for ranking, path in ranking_paths.items()
        }
        checkpoint_source = str(config.get("topk_run_dir", "")).strip()
        checkpoint = _oracle_checkpoint_metadata(
            experiment_root=experiment_root,
            checkpoint_source=checkpoint_source,
            config=config,
        )
        scores_source = str(config.get("scores_pt", "")).strip()
        declared_source_variant_id = str(
            config.get("source_variant_id", "")
        ).strip()
        source_variant_id = (
            declared_source_variant_id
            or (
                Path(scores_source).parent.name
                if scores_source
                else ranking_dir.name
            )
        )
        score_family_id = _oracle_score_family_id(source_variant_id)
        window_match = re.search(r"_w(?P<window>\d+)(?:_|$)", source_variant_id)
        configured_window = config.get("window_size")
        relative_dir = ranking_dir.relative_to(experiment_root).as_posix()
        packages.append(
            {
                "id": relative_dir.replace("/", "__"),
                "label": relative_dir,
                "analysis_id": ranking_dir.name,
                "source_variant_id": source_variant_id,
                "score_family_id": score_family_id,
                "window_size": (
                    int(configured_window)
                    if configured_window is not None
                    else (
                        int(window_match.group("window"))
                        if window_match is not None
                        else None
                    )
                ),
                **checkpoint,
                "top_k": int(config.get("top_k", 0)),
                "top_n_per_row": int(config.get("top_n_per_row", 0)),
                "min_coverage": float(config.get("min_coverage", 0.0)),
                "rankings": rows,
            }
        )
    return packages


def _oracle_checkpoint_catalog(
    *,
    experiment_root: Path,
    ranking_packages: list[dict],
) -> tuple[list[dict], str | None]:
    """Describe encoded checkpoints and the rankings connected to each one."""

    checkpoints: dict[str, dict] = {}
    activation_root = experiment_root / "activations"
    for manifest_path in sorted(activation_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "token_topk_sparse_v1":
            continue
        metadata = _oracle_checkpoint_metadata(
            experiment_root=experiment_root,
            checkpoint_source=str(manifest_path.parent),
        )
        checkpoint_id = str(metadata["checkpoint_id"])
        existing = checkpoints.get(checkpoint_id)
        if (
            existing is not None
            and existing.get("sae_sha256")
            and metadata.get("checkpoint_sae_sha256")
            and existing["sae_sha256"]
            != metadata["checkpoint_sae_sha256"]
        ):
            raise ValueError(
                f"Conflicting SAE hashes for Oracle checkpoint {checkpoint_id}"
            )
        checkpoints[checkpoint_id] = {
            "id": checkpoint_id,
            "label": str(metadata["checkpoint_label"]),
            "batch_size": metadata["checkpoint_batch_size"],
            "training_steps": metadata["checkpoint_training_steps"],
            "sae_sha256": metadata["checkpoint_sae_sha256"],
            "dict_size": metadata["checkpoint_dict_size"],
            "topk": metadata["checkpoint_topk"],
            "total_rows": metadata["checkpoint_total_rows"],
            "lossless_topk": metadata["checkpoint_lossless_topk"],
            "topk_run_id": metadata["checkpoint_topk_run_id"],
            "is_reference": bool(metadata["checkpoint_is_reference"]),
            "encoded": bool(metadata["checkpoint_encoded"]),
        }

    for package in ranking_packages:
        checkpoint_id = str(package.get("checkpoint_id") or "")
        if not checkpoint_id:
            continue
        checkpoints.setdefault(
            checkpoint_id,
            {
                "id": checkpoint_id,
                "label": str(package["checkpoint_label"]),
                "batch_size": package.get("checkpoint_batch_size"),
                "training_steps": package.get("checkpoint_training_steps"),
                "sae_sha256": package.get("checkpoint_sae_sha256"),
                "dict_size": package.get("checkpoint_dict_size"),
                "topk": package.get("checkpoint_topk"),
                "total_rows": package.get("checkpoint_total_rows"),
                "lossless_topk": package.get("checkpoint_lossless_topk"),
                "topk_run_id": package.get("checkpoint_topk_run_id"),
                "is_reference": bool(
                    package.get("checkpoint_is_reference", False)
                ),
                "encoded": bool(package.get("checkpoint_encoded", False)),
            },
        )

    catalog: list[dict] = []
    for checkpoint_id, checkpoint in checkpoints.items():
        discovered = [
            package
            for package in ranking_packages
            if package.get("checkpoint_id") == checkpoint_id
        ]
        connected = [
            package
            for package in discovered
            if package.get("source_mapping_verified") is True
        ]
        catalog.append(
            {
                **checkpoint,
                "state": (
                    "ready"
                    if connected
                    else "encoded"
                    if checkpoint["encoded"]
                    else "discovered"
                ),
                "num_discovered_packages": len(discovered),
                "num_connected_packages": len(connected),
                "variant_ids": sorted(
                    {
                        str(package["source_variant_id"])
                        for package in connected
                        if package.get("source_variant_id")
                    }
                ),
                "analysis_ids": sorted(
                    str(package["analysis_id"]) for package in connected
                ),
                "window_sizes": sorted(
                    {
                        int(package["window_size"])
                        for package in connected
                        if package.get("window_size") is not None
                    }
                ),
                "scopes": sorted(
                    {
                        str(package["scope"])
                        for package in connected
                        if package.get("scope")
                    }
                ),
            }
        )
    catalog.sort(
        key=lambda checkpoint: (
            checkpoint["training_steps"] is None,
            int(checkpoint["training_steps"] or 0),
            int(checkpoint["batch_size"] or 0),
            str(checkpoint["id"]),
        )
    )
    ready = [
        checkpoint
        for checkpoint in catalog
        if int(checkpoint["num_connected_packages"]) > 0
    ]
    default_checkpoint = next(
        (
            checkpoint
            for checkpoint in ready
            if bool(checkpoint["is_reference"])
        ),
        ready[0] if ready else next(
            (
                checkpoint
                for checkpoint in catalog
                if bool(checkpoint["is_reference"])
            ),
            catalog[0] if catalog else None,
        ),
    )
    return (
        catalog,
        str(default_checkpoint["id"]) if default_checkpoint else None,
    )


def build_oracle_phase_results_dataset(
    experiment_root: Path | None,
) -> tuple[dict, dict[tuple[str, str], tuple[Path, ...]]]:
    """Build a compact browser view of an in-progress oracle-phase experiment.

    The adapter intentionally treats keyframes, media, clustering, and SAE
    ranking as separate readiness stages. This lets the shared UI remain
    usable while later artifacts are still being produced.
    """

    if experiment_root is None:
        return (
            build_oracle_waiting_payload(
                status="not_configured",
                message="Oracle phase 실험 경로가 아직 지정되지 않았습니다.",
            ),
            {},
        )
    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        return (
            build_oracle_waiting_payload(
                status="waiting_for_artifacts",
                message="Oracle phase 실험 산출물을 기다리는 중입니다.",
            ),
            {},
        )

    keyframe_manifest_path = (
        root / "oracle_keyframes/oracle_phase_keyframes_manifest.json"
    )
    events_path = root / "oracle_keyframes/oracle_phase_events.jsonl"
    if not keyframe_manifest_path.is_file() or not events_path.is_file():
        return (
            build_oracle_waiting_payload(
                status="collecting_keyframes",
                message="Oracle phase keyframe 생성이 진행 중입니다.",
            ),
            {},
        )

    keyframe_manifest = json.loads(
        keyframe_manifest_path.read_text(encoding="utf-8")
    )
    if (
        keyframe_manifest.get("format")
        != "event_sae_oracle_phase_keyframes_v1"
    ):
        raise ValueError("Unsupported oracle phase keyframe manifest format")
    if keyframe_manifest.get("claim_scope") != ORACLE_PHASE_CLAIM_SCOPE:
        raise ValueError("Oracle phase claim scope is missing or unsafe")
    if keyframe_manifest.get("label_source") != "env_step_phases":
        raise ValueError("Oracle phase labels must come from env_step_phases")

    events = load_jsonl(events_path)
    event_by_id = {str(row["sample_id"]): row for row in events}
    if not events or len(event_by_id) != len(events):
        raise ValueError("Oracle phase events must have unique sample IDs")
    if len(events) != int(keyframe_manifest["num_keyframes"]):
        raise ValueError("Oracle phase event count does not match its manifest")
    if any(row.get("oracle_upper_bound") is not True for row in events):
        raise ValueError("Every oracle phase event must be marked upper-bound")
    phase_schemes = sorted(
        {
            str(row.get("phase_scheme", "")).strip()
            for row in events
            if str(row.get("phase_scheme", "")).strip()
        }
    )
    if len(phase_schemes) != 1:
        raise ValueError(
            "Oracle results UI requires exactly one explicit phase_scheme"
        )
    phase_scheme = phase_schemes[0]

    phase_assignments_path = (
        root
        / "oracle_keyframes/oracle_phase_cluster_assignments.jsonl"
    )
    phase_annotations_path = (
        root
        / "oracle_keyframes/oracle_phase_cluster_annotations.jsonl"
    )
    if not phase_assignments_path.is_file() or not phase_annotations_path.is_file():
        raise FileNotFoundError("Oracle phase-only scoring groups are incomplete")
    phase_assignments = load_jsonl(phase_assignments_path)
    phase_annotations = load_jsonl(phase_annotations_path)
    phase_annotation_by_id = {
        str(row["cluster_id"]): row for row in phase_annotations
    }
    if len(phase_annotation_by_id) != len(phase_annotations):
        raise ValueError("Oracle phase-only scoring group IDs are duplicated")
    phase_members: dict[str, list[str]] = defaultdict(list)
    for assignment in phase_assignments:
        cluster_id = str(assignment["cluster_id"])
        sample_id = str(assignment["sample_id"])
        if (
            cluster_id not in phase_annotation_by_id
            or sample_id not in event_by_id
        ):
            raise ValueError("Invalid oracle phase-only scoring assignment")
        phase_members[cluster_id].append(sample_id)
    if {
        sample_id
        for sample_ids in phase_members.values()
        for sample_id in sample_ids
    } != set(event_by_id):
        raise ValueError("Oracle phase-only scoring groups do not cover events")
    phase_groups = []
    for cluster_id, annotation in sorted(phase_annotation_by_id.items()):
        member_ids = list(phase_members[cluster_id])
        member_schemes = {
            str(event_by_id[sample_id].get("phase_scheme", "")).strip()
            for sample_id in member_ids
        }
        if member_schemes != {phase_scheme}:
            raise ValueError(
                f"Oracle phase-only group mixes phase schemes: {cluster_id}"
            )
        task_episode_coverage = float(annotation["episode_coverage"])
        phase_groups.append(
            {
                "phase_group_id": cluster_id,
                "task_description": str(annotation["task_description"]),
                "phase_scheme": phase_scheme,
                "phase": str(annotation["phase"]),
                "phrase": str(annotation["phrase"]),
                "num_members": int(annotation["num_members"]),
                "num_episodes": int(annotation["num_episodes"]),
                # Compatibility alias. Phase-only groups use the task-wide
                # denominator, unlike state clusters below.
                "episode_coverage": task_episode_coverage,
                "task_episode_coverage": task_episode_coverage,
                "coverage_scope": "all_task_episodes",
                "sample_ids": member_ids,
            }
        )

    trajectory_manifest_path = root / "trajectory/trajectory_manifest.json"
    trajectory_manifest = (
        json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
        if trajectory_manifest_path.is_file()
        else {}
    )
    inventory_verified = bool(
        trajectory_manifest.get("inventory_verified", False)
    )

    media_paths: dict[tuple[str, str], tuple[Path, ...]] = {}
    media_by_view: dict[str, dict[str, dict]] = {}
    media_reports: dict[str, dict] = {}
    complete_views: list[str] = []
    for view in ORACLE_PHASE_VIEWS:
        report_path = root / "media" / view / "packaging_report.json"
        samples_path = root / "media" / view / "samples.jsonl"
        if not report_path.is_file() or not samples_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("format") != "event_sae_oracle_phase_media_v1"
            or str(report.get("view_name")) != view
            or not bool(report.get("passed", False))
        ):
            raise ValueError(f"Invalid oracle media report for {view}")
        samples = load_jsonl(samples_path)
        indexed_samples = {
            str(row["sample_id"]): row
            for row in samples
        }
        if (
            len(indexed_samples) != len(samples)
            or set(indexed_samples) != set(event_by_id)
        ):
            raise ValueError(
                f"Oracle {view} media does not exactly cover keyframes"
            )
        for sample_id, sample in indexed_samples.items():
            if (
                str(sample.get("view_name")) != view
                or str(sample.get("phase"))
                != str(event_by_id[sample_id]["phase"])
            ):
                raise ValueError(
                    f"Oracle {view} media identity mismatch: {sample_id}"
                )
            paths = tuple(
                _oracle_resolved_frame_path(
                    experiment_root=root,
                    raw_path=raw_path,
                    label=f"{view} frame for {sample_id}",
                )
                for raw_path in sample.get("frame_paths", [])
            )
            if not paths:
                raise ValueError(f"Oracle {view} media has no frames: {sample_id}")
            media_paths[(sample_id, view)] = paths
        media_reports[view] = report
        media_by_view[view] = indexed_samples
        complete_views.append(view)

    sample_rows: dict[str, dict] = {}
    for sample_id, event in event_by_id.items():
        view_frame_counts = {
            view: len(media_paths.get((sample_id, view), ()))
            for view in complete_views
        }
        media_sample = (
            media_by_view[complete_views[0]].get(sample_id)
            if complete_views
            else {}
        )
        sample_rows[sample_id] = {
            "sample_id": sample_id,
            "episode_num": int(event["episode_num"]),
            "task_episode_idx": int(event["task_episode_idx"]),
            "success": bool(event["success"]),
            "phase_scheme": phase_scheme,
            "phase": str(event["phase"]),
            "state_env_step_index": int(event["state_env_step_index"]),
            "activation_env_step_index": int(
                event["activation_env_step_index"]
            ),
            "action_token_offset": int(event["action_token_offset"]),
            "progress_percent": float(event["progress_percent"]),
            "anchor_source": str(event["anchor_source"]),
            "event_labels": [
                str(value) for value in event.get("event_labels", [])
            ],
            "anchor_env_step_error": (
                int(media_sample["anchor_env_step_error"])
                if media_sample
                else None
            ),
            "phase_frame_padding_count": (
                int(media_sample["phase_frame_padding_count"])
                if media_sample
                else 0
            ),
            "frames_within_oracle_phase": (
                bool(media_sample["frames_within_oracle_phase"])
                if media_sample
                else None
            ),
            "frame_counts": view_frame_counts,
        }

    variants: list[dict] = []
    clustering_root = root / "clustering"
    if clustering_root.is_dir():
        for manifest_path in sorted(
            clustering_root.glob("*/oracle_phase_clustering_manifest.json")
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            clustering = manifest.get("clustering")
            if (
                manifest.get("format")
                != "event_sae_oracle_phase_state_clustering_v1"
                or not isinstance(clustering, dict)
                or clustering.get("claim_scope") != ORACLE_PHASE_CLAIM_SCOPE
                or clustering.get("annotation_mode")
                != "programmatic_oracle_no_vlm"
                or clustering.get("coverage_scope")
                != "episodes_with_oracle_phase_keyframe"
            ):
                raise ValueError(
                    f"Invalid oracle clustering manifest: {manifest_path}"
                )
            variant_events_path = Path(
                str(manifest.get("alignment", {}).get("oracle_events_path", ""))
            ).expanduser().resolve()
            if (
                not variant_events_path.is_file()
                or (
                    variant_events_path != root
                    and root not in variant_events_path.parents
                )
            ):
                raise ValueError(
                    f"Invalid oracle clustering event selection: {manifest_path}"
                )
            variant_events = load_jsonl(variant_events_path)
            variant_event_ids = {
                str(row["sample_id"]) for row in variant_events
            }
            if (
                not variant_event_ids
                or len(variant_event_ids) != len(variant_events)
                or not variant_event_ids.issubset(event_by_id)
                or len(variant_event_ids) != int(clustering["num_events"])
            ):
                raise ValueError(
                    f"Oracle clustering event selection mismatch: {manifest_path}"
                )
            cluster_dir = manifest_path.parent / "phase_state_clusters"
            clusters_path = cluster_dir / "clusters.jsonl"
            assignments_path = cluster_dir / "cluster_assignments.jsonl"
            annotations_path = cluster_dir / "cluster_annotations.jsonl"
            if not all(
                path.is_file()
                for path in (clusters_path, assignments_path, annotations_path)
            ):
                raise FileNotFoundError(
                    f"Incomplete oracle cluster bundle: {cluster_dir}"
                )
            cluster_rows = load_jsonl(clusters_path)
            assignment_rows = load_jsonl(assignments_path)
            annotation_rows = load_jsonl(annotations_path)
            cluster_by_id = {
                str(row["cluster_id"]): row for row in cluster_rows
            }
            annotation_by_id = {
                str(row["cluster_id"]): row for row in annotation_rows
            }
            if (
                len(cluster_by_id) != len(cluster_rows)
                or len(annotation_by_id) != len(annotation_rows)
                or set(cluster_by_id) != set(annotation_by_id)
            ):
                raise ValueError(
                    f"Oracle cluster/annotation ID mismatch: {cluster_dir}"
                )
            assignments_by_cluster: dict[str, list[dict]] = defaultdict(list)
            assignment_sample_ids: set[str] = set()
            for assignment in assignment_rows:
                cluster_id = str(assignment["cluster_id"])
                sample_id = str(assignment["sample_id"])
                if (
                    cluster_id not in cluster_by_id
                    or sample_id not in event_by_id
                    or sample_id in assignment_sample_ids
                ):
                    raise ValueError(
                        f"Invalid oracle cluster assignment: {sample_id}"
                    )
                assignment_sample_ids.add(sample_id)
                assignments_by_cluster[cluster_id].append(assignment)
            if assignment_sample_ids != variant_event_ids:
                raise ValueError(
                    "Oracle clustering does not exactly cover its selected "
                    f"events: {cluster_dir}"
                )

            compact_clusters: list[dict] = []
            for cluster_id, cluster in cluster_by_id.items():
                annotation = annotation_by_id[cluster_id]
                if (
                    cluster.get("oracle_upper_bound") is not True
                    or annotation.get("oracle_upper_bound") is not True
                    or str(annotation.get("phase"))
                    != str(cluster.get("phase"))
                    or bool(annotation.get("actual_human_review_completed"))
                ):
                    raise ValueError(
                        f"Unsafe oracle cluster metadata: {cluster_id}"
                    )
                representative_ids = [
                    str(value)
                    for value in cluster.get("representative_sample_ids", [])
                ]
                member_ids = [
                    str(value)
                    for value in cluster.get("member_sample_ids", [])
                ]
                ordered_ids = list(
                    dict.fromkeys([*representative_ids, *member_ids])
                )
                if not ordered_ids:
                    ordered_ids = [
                        str(row["sample_id"])
                        for row in assignments_by_cluster[cluster_id]
                    ]
                if any(sample_id not in sample_rows for sample_id in ordered_ids):
                    raise ValueError(
                        f"Oracle cluster references unknown sample: {cluster_id}"
                    )
                member_phase_schemes = {
                    str(sample_rows[sample_id]["phase_scheme"])
                    for sample_id in ordered_ids
                }
                cluster_phase_scheme = str(
                    cluster.get("phase_scheme", phase_scheme)
                ).strip()
                if (
                    member_phase_schemes != {phase_scheme}
                    or cluster_phase_scheme != phase_scheme
                ):
                    raise ValueError(
                        f"Oracle cluster phase scheme mismatch: {cluster_id}"
                    )
                legacy_coverage = float(cluster["episode_coverage"])
                phase_episode_coverage = float(
                    cluster.get(
                        "phase_episode_coverage",
                        legacy_coverage,
                    )
                )
                if not math.isclose(
                    legacy_coverage,
                    phase_episode_coverage,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"Oracle cluster phase coverage mismatch: {cluster_id}"
                    )
                task_coverage_value = cluster.get("task_episode_coverage")
                task_episode_coverage = (
                    float(task_coverage_value)
                    if task_coverage_value is not None
                    else None
                )
                total_task_value = (
                    cluster.get("total_task_episodes")
                    if task_episode_coverage is not None
                    else None
                )
                total_task_episodes = (
                    int(total_task_value)
                    if total_task_value is not None
                    else None
                )
                if (
                    task_episode_coverage is not None
                    and (
                        not 0.0 <= task_episode_coverage <= 1.0
                        or task_episode_coverage
                        > phase_episode_coverage + 1e-9
                        or total_task_episodes is None
                        or total_task_episodes
                        < int(cluster["total_phase_episodes"])
                    )
                ):
                    raise ValueError(
                        f"Unsafe oracle task coverage metadata: {cluster_id}"
                    )
                compact_clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "phase_scheme": phase_scheme,
                        "phase": str(cluster["phase"]),
                        "task_id": int(cluster["task_id"]),
                        "task_description": str(
                            cluster["task_description"]
                        ),
                        "phase_cluster_index": int(
                            cluster["phase_cluster_index"]
                        ),
                        "num_members": int(cluster["num_members"]),
                        # Compatibility alias: state-cluster filtering is
                        # intentionally phase-conditional.
                        "episode_coverage": phase_episode_coverage,
                        "phase_episode_coverage": phase_episode_coverage,
                        "task_episode_coverage": task_episode_coverage,
                        "total_phase_episodes": int(
                            cluster["total_phase_episodes"]
                        ),
                        "total_task_episodes": total_task_episodes,
                        "coverage_schema": (
                            "explicit_phase_and_task"
                            if task_episode_coverage is not None
                            else "legacy_phase_only"
                        ),
                        "meets_min_coverage": bool(
                            cluster["meets_min_coverage"]
                        ),
                        "phrase": str(annotation["phrase"]),
                        "sample_ids": ordered_ids,
                        "samples": [
                            sample_rows[sample_id]
                            for sample_id in ordered_ids
                        ],
                    }
                )
            compact_clusters.sort(
                key=lambda row: (
                    row["task_description"],
                    row["phase"],
                    row["phase_cluster_index"],
                )
            )
            variant_id = manifest_path.parent.name
            state_weight = float(clustering.get("state_weight", 0.0))
            distance_threshold = float(
                clustering.get("distance_threshold", 0.0)
            )
            window_match = re.search(
                r"_causal_w(?P<window>\d+)(?:_v\d+)?$",
                variant_id,
            )
            revision_match = re.search(r"_v(?P<revision>\d+)$", variant_id)
            aggregation_scope = (
                "phase" if variant_id.startswith("phase_only_")
                else "phase_state"
            )
            event_selection = (
                "all_phase_entries"
                if variant_events_path.resolve() == events_path.resolve()
                else variant_events_path.parent.name
            )
            analysis_label = (
                "전체 phase-entry"
                if event_selection == "all_phase_entries"
                else (
                    f"Causal W{window_match.group('window')}"
                    if window_match is not None
                    else event_selection
                )
            )
            if revision_match is not None:
                analysis_label += f" · v{revision_match.group('revision')}"
            if aggregation_scope == "phase":
                analysis_label = f"Phase 단일 그룹 · {analysis_label}"
            variants.append(
                {
                    "id": variant_id,
                    "aggregation_scope": aggregation_scope,
                    "analysis_label": analysis_label,
                    "revision": (
                        int(revision_match.group("revision"))
                        if revision_match is not None
                        else 1
                    ),
                    "window_size": (
                        int(window_match.group("window"))
                        if window_match is not None
                        else None
                    ),
                    "label": (
                        "Oracle phase 단일 그룹"
                        if aggregation_scope == "phase"
                        else "SigLIP + chunk-start robot state"
                        if state_weight > 0
                        else "3-view SigLIP only"
                    ),
                    "descriptor": (
                        "Phase-level 직접 집계"
                        if aggregation_scope == "phase"
                        else "민감도 확인용"
                        if state_weight > 0
                        else "영상 상태 기준"
                    ),
                    "distance_threshold": distance_threshold,
                    "vision_weight": float(
                        clustering.get("vision_weight", 0.0)
                    ),
                    "state_weight": state_weight,
                    "progress_weight": float(
                        clustering.get("progress_weight", 0.0)
                    ),
                    "event_selection": event_selection,
                    "min_coverage": float(
                        clustering.get("min_coverage", 0.0)
                    ),
                    "num_events": int(clustering["num_events"]),
                    "num_clusters": int(clustering["num_clusters"]),
                    "num_singleton_clusters": int(
                        clustering["num_singleton_clusters"]
                    ),
                    "singleton_event_fraction": float(
                        clustering["singleton_event_fraction"]
                    ),
                    "median_cluster_size": float(
                        clustering["median_cluster_size"]
                    ),
                    "max_cluster_size": int(clustering["max_cluster_size"]),
                    "num_clusters_meeting_min_coverage": int(
                        clustering["num_clusters_meeting_min_coverage"]
                    ),
                    "max_state_vector_env_step_lag": int(
                        manifest.get("alignment", {}).get(
                            "max_state_vector_env_step_lag",
                            0,
                        )
                    ),
                    "phase_scheme": phase_scheme,
                    "coverage_schema": (
                        "explicit_phase_and_task"
                        if compact_clusters
                        and all(
                            cluster["coverage_schema"]
                            == "explicit_phase_and_task"
                            for cluster in compact_clusters
                        )
                        else "legacy_phase_only"
                    ),
                    "progress_used_for_fitting": bool(
                        clustering.get("progress_used_for_fitting", False)
                    ),
                    "success_used_for_fitting_or_selection": bool(
                        clustering.get(
                            "success_used_for_fitting_or_selection",
                            False,
                        )
                    ),
                    "phases": [
                        {
                            "phase_scheme": str(
                                row.get("phase_scheme", phase_scheme)
                            ),
                            "phase": str(row["phase"]),
                            "num_events": int(row["num_events"]),
                            "num_phase_episodes": int(
                                row["num_phase_episodes"]
                            ),
                            "num_task_episodes": int(
                                row.get(
                                    "num_task_episodes",
                                    row["num_phase_episodes"],
                                )
                            ),
                            "num_clusters": int(row["num_clusters"]),
                            "num_singleton_clusters": int(
                                row["num_singleton_clusters"]
                            ),
                        }
                        for row in clustering.get(
                            "partition_summaries", []
                        )
                    ],
                    "clusters": compact_clusters,
                }
            )

    default_variant_id = None
    if variants:
        default_variant_id = next(
            (
                variant["id"]
                for variant in variants
                if variant["id"] == "c2_siglip_only"
            ),
            variants[0]["id"],
        )
    for phase_group in phase_groups:
        phase_group["source_cluster_ids_by_variant"] = {
            variant["id"]: [
                cluster["cluster_id"]
                for cluster in variant["clusters"]
                if (
                    cluster["task_description"]
                    == phase_group["task_description"]
                    and cluster["phase_scheme"]
                    == phase_group["phase_scheme"]
                    and cluster["phase"] == phase_group["phase"]
                )
            ]
            for variant in variants
        }

    ranking_packages = _oracle_ranking_packages(root)
    phase_group_ids = {
        str(phase_group["phase_group_id"]) for phase_group in phase_groups
    }
    variant_by_id = {
        str(variant["id"]): variant for variant in variants
    }
    variant_ids = {str(variant["id"]) for variant in variants}
    variant_cluster_ids = {
        str(variant["id"]): {
            str(cluster["cluster_id"]) for cluster in variant["clusters"]
        }
        for variant in variants
    }
    state_cluster_ids = {
        str(cluster["cluster_id"])
        for variant in variants
        for cluster in variant["clusters"]
    }
    for package in ranking_packages:
        ranked_cluster_ids = {
            str(row["cluster_id"])
            for ranking in ("event_aligned", "window_mean")
            for row in package["rankings"].get(ranking, [])
            if row.get("cluster_id") is not None
        }
        if ranked_cluster_ids and ranked_cluster_ids.issubset(phase_group_ids):
            package["scope"] = "phase"
        elif ranked_cluster_ids and ranked_cluster_ids.issubset(
            state_cluster_ids
        ):
            package["scope"] = "state_cluster"
        else:
            package["scope"] = "unknown"
        score_run_id = str(package["source_variant_id"])
        score_family_id = str(
            package.get("score_family_id") or score_run_id
        )
        package["score_run_id"] = score_run_id
        package["source_variant_inferred"] = False
        package["source_mapping_reason"] = "explicit_or_exact"
        if score_run_id not in variant_ids:
            window_match = re.fullmatch(
                r"(?P<prefix>.+)_w\d+(?P<revision>_v\d+)?",
                score_family_id,
            )
            matching_variants = (
                sorted(
                    variant_id
                    for variant_id in variant_ids
                    if (
                        window_match is not None
                        and re.fullmatch(
                            re.escape(window_match.group("prefix"))
                            + r"_w\d+"
                            + re.escape(
                                window_match.group("revision") or ""
                            ),
                            variant_id,
                        )
                    )
                )
            )
            if package["scope"] == "state_cluster":
                desired_aggregation_scope = (
                    "phase"
                    if score_family_id.startswith("phase_only_")
                    else None
                )
                matching_variants = [
                    variant_id
                    for variant_id in matching_variants
                    if (
                        ranked_cluster_ids.issubset(
                            variant_cluster_ids[variant_id]
                        )
                        and (
                            desired_aggregation_scope is None
                            or variant_by_id[variant_id][
                                "aggregation_scope"
                            ]
                            == desired_aggregation_scope
                        )
                    )
                ]
                if not matching_variants:
                    cluster_matches = [
                        variant_id
                        for variant_id, cluster_ids in (
                            variant_cluster_ids.items()
                        )
                        if (
                            ranked_cluster_ids.issubset(cluster_ids)
                            and (
                                desired_aggregation_scope is None
                                or variant_by_id[variant_id][
                                    "aggregation_scope"
                                ]
                                == desired_aggregation_scope
                            )
                        )
                    ]
                    if len(cluster_matches) == 1:
                        matching_variants = cluster_matches
                        package["source_mapping_reason"] = (
                            "unique_ranked_cluster_membership"
                        )
            elif package["scope"] == "phase":
                matching_variants = [
                    variant_id
                    for variant_id in matching_variants
                    if variant_by_id[variant_id]["aggregation_scope"]
                    == "phase"
                ]
                if not matching_variants:
                    phase_variants = [
                        variant_id
                        for variant_id, variant in variant_by_id.items()
                        if variant["aggregation_scope"] == "phase"
                    ]
                    if len(phase_variants) == 1:
                        matching_variants = phase_variants
                        package["source_mapping_reason"] = (
                            "unique_phase_aggregation_variant"
                        )
            if len(matching_variants) == 1:
                package["source_variant_id"] = matching_variants[0]
                package["source_variant_inferred"] = True
                if package["source_mapping_reason"] == "explicit_or_exact":
                    package["source_mapping_reason"] = (
                        "window_family_match"
                    )
        source_variant_id = str(package["source_variant_id"])
        package["source_mapping_verified"] = (
            source_variant_id in variant_ids
            and (
                (
                    package["scope"] == "state_cluster"
                    and ranked_cluster_ids.issubset(
                        variant_cluster_ids[source_variant_id]
                    )
                )
                or (
                    package["scope"] == "phase"
                    and ranked_cluster_ids.issubset(phase_group_ids)
                )
            )
        )
        if not package["source_mapping_verified"]:
            package["source_variant_id"] = None
    for variant in variants:
        variant_packages = [
            package
            for package in ranking_packages
            if package["source_variant_id"] == variant["id"]
        ]
        variant["num_ranking_packages"] = len(variant_packages)
        variant["num_ranking_checkpoints"] = len(
            {
                str(package["checkpoint_id"])
                for package in variant_packages
                if package.get("checkpoint_id")
            }
        )
        variant["ranking_checkpoint_ids"] = sorted(
            {
                str(package["checkpoint_id"])
                for package in variant_packages
                if package.get("checkpoint_id")
            }
        )
    ranked_variants = [
        variant
        for variant in variants
        if int(variant["num_ranking_packages"]) > 0
    ]
    if ranked_variants:
        default_variant_id = max(
            ranked_variants,
            key=lambda variant: (
                int(variant["revision"]),
                int(variant["num_ranking_packages"]),
                int(variant.get("window_size") or 0),
                str(variant["id"]),
            ),
        )["id"]
    ranking_checkpoints, default_checkpoint_id = _oracle_checkpoint_catalog(
        experiment_root=root,
        ranking_packages=ranking_packages,
    )
    connected_ranking_packages = [
        package
        for package in ranking_packages
        if package.get("source_mapping_verified") is True
    ]
    multiview_features_path = (
        root / "features/multiview/event_features_manifest.json"
    )
    multiview_features = (
        json.loads(multiview_features_path.read_text(encoding="utf-8"))
        if multiview_features_path.is_file()
        else {}
    )
    features_complete = (
        multiview_features.get("format")
        == "event_sae_multiview_event_features_v1"
        and bool(multiview_features.get("passed", False))
        and int(multiview_features.get("num_samples", -1)) == len(events)
    )
    tasks = sorted(
        {
            (int(row["task_id"]), str(row["task_description"]))
            for row in events
        }
    )
    phase_counts = {
        str(key): int(value)
        for key, value in keyframe_manifest.get("phase_counts", {}).items()
    }
    phase_outcomes: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {"success": set(), "failure": set()}
    )
    for event in events:
        outcome = "success" if bool(event["success"]) else "failure"
        phase_outcomes[str(event["phase"])][outcome].add(
            int(event["episode_num"])
        )
    success_only_phases = sorted(
        phase
        for phase, outcomes in phase_outcomes.items()
        if outcomes["success"] and not outcomes["failure"]
    )
    media_complete = set(complete_views) == set(ORACLE_PHASE_VIEWS)
    clustering_complete = bool(variants)
    ranking_checkpoint_count = sum(
        int(checkpoint["num_connected_packages"]) > 0
        for checkpoint in ranking_checkpoints
    )
    encoded_checkpoint_count = sum(
        bool(checkpoint["encoded"]) for checkpoint in ranking_checkpoints
    )
    ranking_started = bool(connected_ranking_packages)
    ranking_complete = ranking_checkpoint_count >= 3
    status = (
        "feature_rankings_ready"
        if ranking_complete
        else "feature_rankings_partial"
        if ranking_started
        else "clustering_ready"
        if clustering_complete
        else "media_ready"
        if media_complete
        else "keyframes_ready"
    )
    stages = [
        {
            "id": "trajectory",
            "label": "입력 궤적",
            "state": "complete" if inventory_verified else "needs_review",
            "detail": (
                f"{keyframe_manifest['num_source_episodes']}개 episode · 목록 검증됨"
                if inventory_verified
                else (
                    f"{keyframe_manifest['num_source_episodes']}개 pilot episode · "
                    "전체 목록 검증 필요"
                )
            ),
        },
        {
            "id": "keyframes",
            "label": "Oracle phase keyframe",
            "state": "complete",
            "detail": f"{len(events)}개 · {len(phase_counts)}개 phase",
        },
        {
            "id": "media",
            "label": "3-view 영상",
            "state": "complete" if media_complete else "in_progress",
            "detail": (
                f"{len(complete_views)}/3 view · keyframe당 5-frame"
            ),
        },
        {
            "id": "features",
            "label": "3-view SigLIP",
            "state": "complete" if features_complete else "waiting",
            "detail": (
                f"{int(multiview_features.get('num_samples', 0))}개 descriptor"
                if features_complete
                else "영상 descriptor 결과 대기"
            ),
        },
        {
            "id": "clustering",
            "label": "Phase 안 상태 군집",
            "state": "complete" if clustering_complete else "waiting",
            "detail": (
                f"{len(variants)}개 설정 비교 가능"
                if clustering_complete
                else "SigLIP 군집 결과 대기"
            ),
        },
        {
            "id": "ranking",
            "label": "SAE feature ranking",
            "state": (
                "complete"
                if ranking_complete
                else "in_progress"
                if ranking_started
                else "waiting"
            ),
            "detail": (
                f"{ranking_checkpoint_count}/3 checkpoint · "
                f"{len(connected_ranking_packages)}개 analysis 연결"
                if ranking_started
                else (
                    f"{encoded_checkpoint_count}/3 checkpoint encoding 완료 · "
                    "ranking 연결 대기"
                    if encoded_checkpoint_count
                    else "3개 checkpoint 결과 생성 전"
                )
            ),
        },
    ]
    first_media_report = (
        media_reports[complete_views[0]] if complete_views else {}
    )
    default_variant = next(
        (
            variant
            for variant in variants
            if variant["id"] == default_variant_id
        ),
        None,
    )
    alignment = keyframe_manifest.get("activation_alignment", {})
    return (
        {
            "format": ORACLE_PHASE_BROWSER_FORMAT,
            "available": True,
            "status": status,
            "message": (
                "Pilot은 조회 가능하지만 입력 episode 목록 검증이 필요합니다."
                if not inventory_verified
                else (
                    "SAE feature ranking을 기다리는 중입니다."
                    if not ranking_complete
                    else "Oracle phase feature ranking까지 준비되었습니다."
                )
            ),
            "claim_scope": ORACLE_PHASE_CLAIM_SCOPE,
            "annotation_mode": "programmatic_oracle_no_vlm",
            "human_review_completed": False,
            "annotation": {
                "label_source": "env_step_phases",
                "mode": "programmatic_oracle_no_vlm",
                "review_status": "not_applicable",
                "oracle_upper_bound": True,
            },
            "meta": {
                "num_tasks": len(tasks),
                "num_episodes": int(
                    keyframe_manifest["num_source_episodes"]
                ),
                "num_keyframes": len(events),
                "num_phases": len(phase_counts),
                "num_variants": len(variants),
                "num_ranking_packages": len(connected_ranking_packages),
                "num_discovered_ranking_packages": len(ranking_packages),
                "num_unmapped_ranking_packages": (
                    len(ranking_packages) - len(connected_ranking_packages)
                ),
                "num_ranking_checkpoints": ranking_checkpoint_count,
                "num_encoded_checkpoints": encoded_checkpoint_count,
                "inventory_verified": inventory_verified,
                "num_success_episodes": len(
                    {
                        int(row["episode_num"])
                        for row in events
                        if bool(row["success"])
                    }
                ),
                "num_failure_episodes": len(
                    {
                        int(row["episode_num"])
                        for row in events
                        if not bool(row["success"])
                    }
                ),
                "phase_schemes": phase_schemes,
            },
            "stages": stages,
            "tasks": [
                {"task_id": task_id, "task_description": description}
                for task_id, description in tasks
            ],
            "views": complete_views,
            "phases": [
                {
                    "phase": phase,
                    "num_keyframes": count,
                    "num_success_episodes": len(
                        phase_outcomes[phase]["success"]
                    ),
                    "num_failure_episodes": len(
                        phase_outcomes[phase]["failure"]
                    ),
                    "success_only": phase in success_only_phases,
                }
                for phase, count in sorted(phase_counts.items())
            ],
            "variants": variants,
            "default_variant_id": default_variant_id,
            "ranking_checkpoints": ranking_checkpoints,
            "default_checkpoint_id": default_checkpoint_id,
            "phase_groups": phase_groups,
            "ranking_packages": ranking_packages,
            "alignment": {
                "label_source": str(keyframe_manifest["label_source"]),
                "label_resolution": str(
                    keyframe_manifest["label_resolution"]
                ),
                "state_index": str(alignment.get("state_index", "")),
                "phase_action_index": str(
                    alignment.get("phase_action_index", "")
                ),
                "causal_action_index": str(
                    alignment.get("causal_action_index", "")
                ),
                "frame_rule": str(
                    first_media_report.get("alignment", "")
                ),
                "max_state_vector_env_step_lag": (
                    int(default_variant["max_state_vector_env_step_lag"])
                    if default_variant
                    else None
                ),
            },
            "audit_gates": [
                {
                    "gate": "입력 episode 목록",
                    "state": "pass" if inventory_verified else "pending",
                    "evidence": (
                        "trajectory inventory가 검증되었습니다."
                        if inventory_verified
                        else (
                            "trajectory manifest의 inventory_verified=false입니다. "
                            "현재 8개는 전체 모집단이 아닌 pilot로만 봅니다."
                        )
                    ),
                },
                {
                    "gate": "Task·instruction 균형",
                    "state": "pending",
                    "evidence": (
                        f"현재 pilot은 {len(tasks)}개 task 범위입니다. "
                        "다른 task·instruction으로 일반화할 수 없습니다."
                    ),
                },
                {
                    "gate": "SAE checkpoint 범위",
                    "state": (
                        "pass"
                        if ranking_checkpoint_count >= 3
                        else "pending"
                    ),
                    "evidence": (
                        f"군집 source 연결이 검증된 결과는 현재 "
                        f"{ranking_checkpoint_count}/3 SAE입니다. "
                        "W4와 W5는 서로 다른 checkpoint가 아니라 동일 "
                        "checkpoint의 분석 window 민감도이며, UI에서 SAE를 "
                        "하나씩 선택해 봅니다."
                    ),
                },
                {
                    "gate": "Phase·dwell 통제",
                    "state": "confounded",
                    "evidence": (
                        "Confounded — 판정 보류. Oracle phase로 hard "
                        "partition했지만 phase 체류시간 matching은 feature "
                        "ranking 단계에서 별도 확인해야 합니다."
                    ),
                },
                {
                    "gate": "Phase·success 혼입",
                    "state": (
                        "confounded" if success_only_phases else "pass"
                    ),
                    "evidence": (
                        "Confounded — 판정 보류. "
                        f"{', '.join(success_only_phases)} phase는 현재 "
                        "success rollout에만 있어 phase feature와 success "
                        "신호를 구분할 수 없습니다."
                        if success_only_phases
                        else "각 phase에 success와 failure rollout이 모두 있습니다."
                    ),
                },
                {
                    "gate": "관찰과 인과 구분",
                    "state": "pass",
                    "evidence": (
                        "이 화면은 diagnostic upper-bound로만 표시하며 "
                        "detector·policy 성능을 주장하지 않습니다."
                    ),
                },
                {
                    "gate": "Scene 일반화",
                    "state": "pending",
                    "evidence": (
                        "현재 cell 밖 replication 전까지 판정 보류입니다."
                    ),
                },
            ],
        },
        media_paths,
    )


__all__ = [
    "build_oracle_phase_results_dataset",
    "build_oracle_waiting_payload",
]
