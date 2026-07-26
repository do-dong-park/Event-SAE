"""Artifact preparation helpers for the GR00T anchor/view ablation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from event_sae import sha256_file
from event_sae.events.io import load_jsonl


VIEW_ORDER = ("left", "right", "wrist")
VIEW_INDEX = {view: index for index, view in enumerate(VIEW_ORDER)}
REUSABLE_ANCHOR_SOURCES = {"position", "both"}
GRIPPER_ONLY_ANCHOR_SOURCE = "gripper_close"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _require_new_outputs(output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    manifest_path = _manifest_path(output_path)
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    return manifest_path


def _index_unique(
    rows: list[dict],
    *,
    label: str,
    key_fields: tuple[str, ...] = ("episode_num", "waypoint_step"),
) -> dict[tuple[Any, ...], dict]:
    indexed: dict[tuple[Any, ...], dict] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in indexed:
            raise ValueError(f"{label}: duplicate key {key}")
        indexed[key] = row
    return indexed


def _waypoint_step(row: dict) -> int:
    if "waypoint_step" in row:
        return int(row["waypoint_step"])
    return int(row["waypoint_index"])


def _index_media_rows(path: Path, *, label: str) -> dict[tuple[int, int], dict]:
    rows = load_jsonl(path)
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "episode_num": int(row["episode_num"]),
                "waypoint_step": _waypoint_step(row),
            }
        )
    return _index_unique(normalized, label=label)


def _resolve_packaged_media_frames(
    samples_path: Path,
    row: dict,
) -> list[str]:
    if row.get("format") != "event_sae_stage3_media_v4":
        raise ValueError(
            f"Expected event_sae_stage3_media_v4, got {row.get('format')!r}"
        )
    frames = sorted(row["frames"], key=lambda item: int(item["position"]))
    positions = [int(item["position"]) for item in frames]
    if positions != list(range(len(frames))):
        raise ValueError(
            f"{row['sample_id']}: non-contiguous frame positions {positions}"
        )
    paths: list[str] = []
    for frame in frames:
        path = (samples_path.parent / str(frame["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = frame.get("sha256")
        if expected_hash is not None and sha256_file(path) != str(expected_hash):
            raise ValueError(f"{row['sample_id']}: frame hash mismatch {path}")
        paths.append(str(path))
    return paths


def _resolved_virtual_frames(row: dict) -> list[str]:
    paths = [str(Path(value).resolve()) for value in row["frame_paths"]]
    if not paths:
        raise ValueError(f"{row['sample_id']}: no virtual frames")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _source_video_metadata(row: dict) -> tuple[str, str | None]:
    relative_path = str(row.get("source_video_relative_path", ""))
    if not relative_path:
        raise ValueError(f"{row['sample_id']}: missing source_video_relative_path")
    video_hash = row.get("source_video_sha256")
    return relative_path, None if video_hash is None else str(video_hash)


def _base_virtual_record(
    *,
    episode: dict,
    source_row: dict,
    sample_id: str,
    waypoint_rank: int,
    waypoint_index: int,
    anchor_source: str,
    frame_paths: list[str],
    trajectory_records_path: Path,
    view: str,
    source_media_kind: str,
) -> dict:
    relative_path, video_hash = _source_video_metadata(source_row)
    return {
        "sample_id": sample_id,
        "task_id": int(episode["task_id"]),
        "task_description": str(episode["task_description"]),
        "prompt_task_description": str(
            episode.get(
                "prompt_task_description",
                source_row.get(
                    "prompt_task_description",
                    episode["task_description"],
                ),
            )
        ),
        "episode_num": int(episode["episode_num"]),
        "task_episode_idx": int(episode["task_episode_idx"]),
        "cell_id": source_row.get("cell_id"),
        "success": bool(episode["success"]),
        "waypoint_rank": int(waypoint_rank),
        "waypoint_index": int(waypoint_index),
        "waypoint_step": int(waypoint_index),
        "anchor_source": anchor_source,
        "frame_paths": frame_paths,
        "clip_path": relative_path,
        "source_trajectory_records_path": str(trajectory_records_path),
        "source_media_format": source_row.get(
            "format",
            source_row.get("source_media_format"),
        ),
        "source_media_kind": source_media_kind,
        "source_media_sample_id": str(source_row["sample_id"]),
        "source_video_relative_path": relative_path,
        "source_video_sha256": video_hash,
        "video_frame_indices": source_row.get("video_frame_indices"),
        "boundary_shift_category": source_row.get("boundary_shift_category"),
        "anchor_env_step_error": source_row.get("anchor_env_step_error"),
        "view": view,
    }


def materialize_position_virtual_media(
    *,
    waypoint_summary_path: Path,
    source_samples_path: Path,
    trajectory_records_path: Path,
    output_path: Path,
    view: str,
    expected_samples: int,
) -> dict:
    """Repackage an exact position-only media bundle with explicit provenance."""

    if view not in VIEW_ORDER:
        raise ValueError(f"Unsupported view: {view}")
    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    source_samples_path = Path(source_samples_path).resolve()
    trajectory_records_path = Path(trajectory_records_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = _require_new_outputs(output_path)
    summary = json.loads(waypoint_summary_path.read_text(encoding="utf-8"))
    source = _index_media_rows(source_samples_path, label=f"{view} position media")
    records: list[dict] = []
    used: set[tuple[int, int]] = set()
    for episode in summary["episodes"]:
        episode_num = int(episode["episode_num"])
        for waypoint_rank, value in enumerate(episode["waypoint_indices"]):
            waypoint_index = int(value)
            key = (episode_num, waypoint_index)
            row = source.get(key)
            if row is None:
                raise ValueError(f"{view}: missing position media {key}")
            used.add(key)
            records.append(
                _base_virtual_record(
                    episode=episode,
                    source_row=row,
                    sample_id=str(row["sample_id"]),
                    waypoint_rank=waypoint_rank,
                    waypoint_index=waypoint_index,
                    anchor_source="position",
                    frame_paths=_resolve_packaged_media_frames(
                        source_samples_path,
                        row,
                    ),
                    trajectory_records_path=trajectory_records_path,
                    view=view,
                    source_media_kind="r_pos_reuse",
                )
            )
    if len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} position samples, found {len(records)}"
        )
    if used != set(source):
        raise ValueError(
            f"{view}: position media not consumed exactly: "
            f"used={len(used)}, available={len(source)}"
        )
    _write_jsonl(output_path, records)
    manifest = {
        "format": "event_sae_anchor_view_virtual_media_v1",
        "anchor_set": "r_pos",
        "view": view,
        "waypoint_summary_path": str(waypoint_summary_path),
        "waypoint_summary_sha256": sha256_file(waypoint_summary_path),
        "source_samples_path": str(source_samples_path),
        "source_samples_sha256": sha256_file(source_samples_path),
        "trajectory_records_path": str(trajectory_records_path),
        "trajectory_records_sha256": sha256_file(trajectory_records_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "num_samples": len(records),
        "source_counts": {"r_pos_reuse": len(records)},
        "passed": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _nearest_rendered_frame(
    *,
    waypoint_index: int,
    n_action_steps: int,
    first_video_frame_env_step: int,
    steps_per_render: int,
) -> tuple[int, int]:
    waypoint_env_step = waypoint_index * n_action_steps
    fractional = (
        waypoint_env_step - first_video_frame_env_step
    ) / steps_per_render
    # Round half upward, matching "nearest rendered frame, ties later".
    frame_index = math.floor(fractional + 0.5)
    frame_env_step = first_video_frame_env_step + frame_index * steps_per_render
    return frame_index, frame_env_step


def video_frame_window(
    *,
    waypoint_index: int,
    timeline: dict,
    offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> tuple[list[int], list[int], str, int]:
    """Map one policy-record anchor to the established five video frames."""

    center, _ = _nearest_rendered_frame(
        waypoint_index=waypoint_index,
        n_action_steps=int(timeline["n_action_steps"]),
        first_video_frame_env_step=int(timeline["first_video_frame_env_step"]),
        steps_per_render=int(timeline["steps_per_render"]),
    )
    requested = [center + offset for offset in offsets]
    frame_count = int(timeline["expected_num_frames"])
    shift = 0
    if requested[0] < 0:
        shift = -requested[0]
    elif requested[-1] >= frame_count:
        shift = frame_count - 1 - requested[-1]
    actual = [value + shift for value in requested]
    if actual[0] < 0 or actual[-1] >= frame_count:
        raise ValueError(
            f"Cannot fit frame window {requested} into {frame_count} frames"
        )
    category = (
        "interior"
        if shift == 0
        else ("shifted_start" if shift > 0 else "shifted_end")
    )
    return actual, requested, category, center


def _find_video(
    *,
    relative_path: str,
    expected_hash: str | None,
    video_roots: list[Path],
) -> Path:
    existing = [
        (Path(root).resolve() / relative_path).resolve()
        for root in video_roots
        if (Path(root).resolve() / relative_path).is_file()
    ]
    if expected_hash is not None:
        existing = [
            path for path in existing if sha256_file(path) == expected_hash
        ]
    if not existing:
        raise FileNotFoundError(
            f"No source video matched {relative_path!r} and its expected hash"
        )
    return sorted(set(existing))[0]


def _render_missing_frames(
    *,
    episode_num: int,
    waypoint_rank: int,
    waypoint_index: int,
    reference_row: dict,
    output_root: Path,
    view: str,
    video_roots: list[Path],
) -> tuple[list[str], list[int], str, int, str]:
    timeline = reference_row.get("video_timeline")
    if not isinstance(timeline, dict):
        raise ValueError(
            f"episode {episode_num}: reference media lacks video_timeline"
        )
    frame_indices, _, category, center = video_frame_window(
        waypoint_index=waypoint_index,
        timeline=timeline,
    )
    relative_path, expected_hash = _source_video_metadata(reference_row)
    video_path = _find_video(
        relative_path=relative_path,
        expected_hash=expected_hash,
        video_roots=video_roots,
    )
    sample_id = (
        f"ep{episode_num:04d}_wp{waypoint_rank:03d}_r{waypoint_index:04d}"
    )
    frame_root = output_root / "frames" / sample_id
    frame_root.mkdir(parents=True, exist_ok=False)
    reader = imageio.get_reader(video_path)
    paths: list[str] = []
    try:
        for position, frame_index in enumerate(frame_indices):
            frame = reader.get_data(frame_index)
            if frame.ndim != 3 or frame.shape[1] % 3 != 0:
                raise ValueError(
                    f"{video_path}: unsupported multiview frame shape {frame.shape}"
                )
            panel_width = frame.shape[1] // 3
            if panel_width != 256 or frame.shape[0] < 256:
                raise ValueError(
                    f"{video_path}: expected 3x256 panels with height >=256, "
                    f"got {frame.shape}"
                )
            x0 = VIEW_INDEX[view] * panel_width
            y0 = frame.shape[0] - 256
            crop = frame[y0 : y0 + 256, x0 : x0 + panel_width]
            frame_env_step = (
                int(timeline["first_video_frame_env_step"])
                + frame_index * int(timeline["steps_per_render"])
            )
            path = (
                frame_root
                / f"frame_{position:02d}_v{frame_index:04d}_s{frame_env_step:04d}.jpg"
            )
            Image.fromarray(crop).save(
                path,
                format="JPEG",
                quality=95,
                optimize=True,
            )
            paths.append(str(path.resolve()))
    finally:
        reader.close()
    _, center_env_step = _nearest_rendered_frame(
        waypoint_index=waypoint_index,
        n_action_steps=int(timeline["n_action_steps"]),
        first_video_frame_env_step=int(timeline["first_video_frame_env_step"]),
        steps_per_render=int(timeline["steps_per_render"]),
    )
    waypoint_env_step = waypoint_index * int(timeline["n_action_steps"])
    return (
        paths,
        frame_indices,
        category,
        center_env_step - waypoint_env_step,
        str(video_path),
    )


def materialize_rpg_virtual_media(
    *,
    waypoint_summary_path: Path,
    primary_samples_path: Path,
    secondary_samples_path: Path,
    trajectory_records_path: Path,
    output_path: Path,
    view: str,
    video_roots: list[Path],
    expected_samples: int,
    expected_primary_reuse: int,
    expected_secondary_reuse: int,
    expected_computed: int,
) -> dict:
    """Assemble R-PG media from R-P, A-PG overlap, and exact new crops."""

    if view not in VIEW_ORDER:
        raise ValueError(f"Unsupported view: {view}")
    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    primary_samples_path = Path(primary_samples_path).resolve()
    secondary_samples_path = Path(secondary_samples_path).resolve()
    trajectory_records_path = Path(trajectory_records_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = _require_new_outputs(output_path)
    summary = json.loads(waypoint_summary_path.read_text(encoding="utf-8"))
    primary = _index_media_rows(primary_samples_path, label=f"{view} R-P media")
    secondary = _index_media_rows(
        secondary_samples_path,
        label=f"{view} A-PG media",
    )
    primary_by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in primary.values():
        primary_by_episode[int(row["episode_num"])].append(row)

    records: list[dict] = []
    source_counts: Counter[str] = Counter()
    used_primary: set[tuple[int, int]] = set()
    used_secondary: set[tuple[int, int]] = set()
    computed_video_paths: set[str] = set()
    for episode in summary["episodes"]:
        episode_num = int(episode["episode_num"])
        anchors = episode.get("waypoint_anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"episode {episode_num}: missing waypoint_anchors")
        anchor_by_index = {
            int(row["waypoint_index"]): str(row["anchor_source"])
            for row in anchors
        }
        indices = [int(value) for value in episode["waypoint_indices"]]
        if set(anchor_by_index) != set(indices):
            raise ValueError(f"episode {episode_num}: waypoint anchor mismatch")
        references = sorted(
            primary_by_episode.get(episode_num, []),
            key=lambda row: int(row["waypoint_step"]),
        )
        if not references:
            raise ValueError(
                f"episode {episode_num}: no R-P reference media for video metadata"
            )

        for waypoint_rank, waypoint_index in enumerate(indices):
            key = (episode_num, waypoint_index)
            anchor_source = anchor_by_index[waypoint_index]
            source_row: dict
            if anchor_source in REUSABLE_ANCHOR_SOURCES:
                source_row = primary.get(key)
                if source_row is None:
                    raise ValueError(f"{view}: missing R-P media {key}")
                frame_paths = _resolve_packaged_media_frames(
                    primary_samples_path,
                    source_row,
                )
                source_kind = "r_pos_reuse"
                used_primary.add(key)
            elif anchor_source == GRIPPER_ONLY_ANCHOR_SOURCE and key in secondary:
                source_row = secondary[key]
                frame_paths = _resolved_virtual_frames(source_row)
                source_kind = "a_pos_gripper_reuse"
                used_secondary.add(key)
            elif anchor_source == GRIPPER_ONLY_ANCHOR_SOURCE:
                source_row = references[0]
                (
                    frame_paths,
                    video_frame_indices,
                    boundary_shift_category,
                    anchor_env_step_error,
                    video_path,
                ) = _render_missing_frames(
                    episode_num=episode_num,
                    waypoint_rank=waypoint_rank,
                    waypoint_index=waypoint_index,
                    reference_row=source_row,
                    output_root=output_path.parent,
                    view=view,
                    video_roots=video_roots,
                )
                source_row = {
                    **source_row,
                    "sample_id": (
                        f"computed_ep{episode_num:04d}_r{waypoint_index:04d}"
                    ),
                    "video_frame_indices": video_frame_indices,
                    "boundary_shift_category": boundary_shift_category,
                    "anchor_env_step_error": anchor_env_step_error,
                    "source_video_path": video_path,
                }
                source_kind = "computed"
                computed_video_paths.add(video_path)
            else:
                raise ValueError(f"Unsupported anchor_source={anchor_source!r}")

            sample_id = (
                f"ep{episode_num:04d}_wp{waypoint_rank:03d}_r{waypoint_index:04d}"
            )
            records.append(
                _base_virtual_record(
                    episode=episode,
                    source_row=source_row,
                    sample_id=sample_id,
                    waypoint_rank=waypoint_rank,
                    waypoint_index=waypoint_index,
                    anchor_source=anchor_source,
                    frame_paths=frame_paths,
                    trajectory_records_path=trajectory_records_path,
                    view=view,
                    source_media_kind=source_kind,
                )
            )
            source_counts[source_kind] += 1

    expected_counts = {
        "r_pos_reuse": expected_primary_reuse,
        "a_pos_gripper_reuse": expected_secondary_reuse,
        "computed": expected_computed,
    }
    if len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, found {len(records)}")
    actual_counts = {
        key: int(source_counts[key]) for key in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"{view}: media source counts {actual_counts} != {expected_counts}"
        )
    if used_primary != set(primary):
        raise ValueError(
            f"{view}: R-P media not consumed exactly: "
            f"used={len(used_primary)}, available={len(primary)}"
        )
    if len(used_secondary) != expected_secondary_reuse:
        raise ValueError(f"{view}: secondary reuse key-count mismatch")

    _write_jsonl(output_path, records)
    manifest = {
        "format": "event_sae_anchor_view_multisource_media_v1",
        "anchor_set": "r_pos_gripper",
        "view": view,
        "waypoint_summary_path": str(waypoint_summary_path),
        "waypoint_summary_sha256": sha256_file(waypoint_summary_path),
        "primary_samples_path": str(primary_samples_path),
        "primary_samples_sha256": sha256_file(primary_samples_path),
        "secondary_samples_path": str(secondary_samples_path),
        "secondary_samples_sha256": sha256_file(secondary_samples_path),
        "trajectory_records_path": str(trajectory_records_path),
        "trajectory_records_sha256": sha256_file(trajectory_records_path),
        "video_roots": [str(Path(root).resolve()) for root in video_roots],
        "computed_video_paths": sorted(computed_video_paths),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "num_samples": len(records),
        "source_counts": actual_counts,
        "primary_consumed_exactly": used_primary == set(primary),
        "secondary_rows_available": len(secondary),
        "secondary_rows_consumed": len(used_secondary),
        "passed": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_multisource_reusable_features(
    *,
    virtual_samples_path: Path,
    primary_features_path: Path,
    secondary_features_path: Path,
    output_path: Path,
    expected_primary: int,
    expected_secondary: int,
) -> dict:
    """Create the exact feature subset consumed by the single-source provider."""

    virtual_samples_path = Path(virtual_samples_path).resolve()
    primary_features_path = Path(primary_features_path).resolve()
    secondary_features_path = Path(secondary_features_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = _require_new_outputs(output_path)
    samples = load_jsonl(virtual_samples_path)
    primary = _index_unique(
        load_jsonl(primary_features_path),
        label="primary features",
    )
    secondary = _index_unique(
        load_jsonl(secondary_features_path),
        label="secondary features",
    )
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    used_primary: set[tuple[int, int]] = set()
    used_secondary: set[tuple[int, int]] = set()
    for sample in samples:
        source_kind = str(sample["source_media_kind"])
        key = (int(sample["episode_num"]), int(sample["waypoint_step"]))
        if source_kind == "r_pos_reuse":
            row = primary.get(key)
            used_primary.add(key)
        elif source_kind == "a_pos_gripper_reuse":
            row = secondary.get(key)
            used_secondary.add(key)
        elif source_kind == "computed":
            continue
        else:
            raise ValueError(f"Unknown source_media_kind={source_kind!r}")
        if row is None:
            raise ValueError(f"Missing {source_kind} feature for key={key}")
        sample_paths = [
            str(Path(value).resolve()) for value in sample["frame_paths"]
        ]
        feature_paths = [
            str(Path(value).resolve()) for value in row["selected_frame_paths"]
        ]
        if sample_paths != feature_paths:
            raise ValueError(
                f"{sample['sample_id']}: reusable feature frame paths differ"
            )
        selected.append(row)
        counts[source_kind] += 1

    expected_counts = {
        "r_pos_reuse": expected_primary,
        "a_pos_gripper_reuse": expected_secondary,
    }
    actual_counts = {key: int(counts[key]) for key in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"Feature source counts {actual_counts} != {expected_counts}")
    if used_primary != set(primary):
        raise ValueError(
            "Primary reusable feature rows were not consumed exactly: "
            f"used={len(used_primary)}, available={len(primary)}"
        )
    keys = [
        (int(row["episode_num"]), int(row["waypoint_step"]))
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Combined reusable feature subset has duplicate keys")
    _write_jsonl(output_path, selected)
    manifest = {
        "format": "event_sae_multisource_reusable_feature_subset_v1",
        "virtual_samples_path": str(virtual_samples_path),
        "virtual_samples_sha256": sha256_file(virtual_samples_path),
        "primary_features_path": str(primary_features_path),
        "primary_features_sha256": sha256_file(primary_features_path),
        "secondary_features_path": str(secondary_features_path),
        "secondary_features_sha256": sha256_file(secondary_features_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "num_rows": len(selected),
        "source_counts": actual_counts,
        "primary_consumed_exactly": used_primary == set(primary),
        "secondary_rows_available": len(secondary),
        "secondary_rows_consumed": len(used_secondary),
        "passed": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def materialize_prompt_records_from_event_features(
    *,
    event_features_paths: list[Path],
    output_path: Path,
    expected_episodes: int,
) -> dict:
    """Materialize an exact episode-to-task map for Stage 4 task means."""

    if not event_features_paths:
        raise ValueError("At least one event feature artifact is required")
    paths = [Path(path).resolve() for path in event_features_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = Path(output_path).resolve()
    manifest_path = _require_new_outputs(output_path)

    mappings: list[dict[int, tuple[int, str, int]]] = []
    for path in paths:
        mapping: dict[int, tuple[int, str, int]] = {}
        for row in load_jsonl(path):
            episode_num = int(row["episode_num"])
            value = (
                int(row["task_id"]),
                str(row["task_description"]),
                int(row["task_episode_idx"]),
            )
            previous = mapping.setdefault(episode_num, value)
            if previous != value:
                raise ValueError(
                    f"{path}: inconsistent task mapping for episode "
                    f"{episode_num}: {previous!r} != {value!r}"
                )
        if len(mapping) != expected_episodes:
            raise ValueError(
                f"{path}: expected {expected_episodes} episodes, "
                f"found {len(mapping)}"
            )
        mappings.append(mapping)

    reference = mappings[0]
    for path, mapping in zip(paths[1:], mappings[1:], strict=True):
        if mapping != reference:
            missing = sorted(set(reference).difference(mapping))
            extra = sorted(set(mapping).difference(reference))
            differing = sorted(
                episode_num
                for episode_num in set(reference).intersection(mapping)
                if reference[episode_num] != mapping[episode_num]
            )
            raise ValueError(
                f"{path}: episode-to-task mapping differs from the first "
                f"artifact; missing={missing[:10]}, extra={extra[:10]}, "
                f"differing={differing[:10]}"
            )

    records = [
        {
            "format": "event_sae_prompt_record_v1",
            "episode_num": episode_num,
            "task_id": task_id,
            "task_description": task_description,
            "task_episode_idx": task_episode_idx,
        }
        for episode_num, (
            task_id,
            task_description,
            task_episode_idx,
        ) in sorted(reference.items())
    ]
    _write_jsonl(output_path, records)
    manifest = {
        "format": "event_sae_prompt_records_from_event_features_v1",
        "event_features": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "episodes": len(mapping),
            }
            for path, mapping in zip(paths, mappings, strict=True)
        ],
        "expected_episodes": int(expected_episodes),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_rows": len(records),
        "all_source_mappings_exact": True,
        "passed": len(records) == expected_episodes,
    }
    _write_json(manifest_path, manifest)
    return manifest
