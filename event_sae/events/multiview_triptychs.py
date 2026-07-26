"""Shared rendering primitives for synchronized Stage 3 triptychs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl

LAYOUT_ID = "synchronized_triptych_left_right_wrist_v1"
VIEW_LABELS = ("LEFT", "RIGHT", "WRIST")
VIEW_WIDTH = 256
SCENE_HEIGHT = 256
LABEL_HEIGHT = 24
VIEW_ORDER = ("left", "right", "wrist")


@dataclass(frozen=True)
class TriptychFrame:
    """One aligned three-view frame resolved by a source-specific adapter."""

    paths: Mapping[str, Path]
    position: int
    video_frame_index: int


TriptychFrameProvider = Callable[[str, str, Sequence[str]], Sequence[TriptychFrame]]


def prepare_output_directory(output_dir: Path) -> Path:
    """Resolve and create an empty output directory without overwriting artifacts."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def verify_aligned_fields(
    sample_id: str,
    rows: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
    *,
    field_label: str = "aligned field",
) -> None:
    """Require exact JSON-stable equality for fields across synchronized views."""
    for field in fields:
        values = {view: row.get(field) for view, row in rows.items()}
        stable_values = {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in values.values()
        }
        if len(stable_values) != 1:
            raise ValueError(f"{sample_id}: {field_label} {field!r} differs: {values}")


def make_triptych_from_views(
    left: Image.Image,
    right: Image.Image,
    wrist: Image.Image,
) -> Image.Image:
    """Tile three already-cropped, synchronized camera views with labels."""
    panels = [left.convert("RGB"), right.convert("RGB"), wrist.convert("RGB")]
    for label, panel in zip(VIEW_LABELS, panels, strict=True):
        if panel.size != (VIEW_WIDTH, SCENE_HEIGHT):
            raise ValueError(
                f"{label} crop size={panel.size}, expected "
                f"{(VIEW_WIDTH, SCENE_HEIGHT)}"
            )

    expected_width = VIEW_WIDTH * len(VIEW_ORDER)
    triptych = Image.new(
        "RGB",
        (expected_width, SCENE_HEIGHT + LABEL_HEIGHT),
        color=(12, 16, 24),
    )
    draw = ImageDraw.Draw(triptych)
    for view_index, (label, panel) in enumerate(
        zip(VIEW_LABELS, panels, strict=True)
    ):
        x0 = view_index * VIEW_WIDTH
        triptych.paste(panel, (x0, LABEL_HEIGHT))
        draw.text((x0 + 8, 6), label, fill=(255, 255, 255))
        if view_index:
            draw.line(
                (x0, 0, x0, SCENE_HEIGHT + LABEL_HEIGHT),
                fill=(255, 255, 255),
            )
    return triptych


def _render_triptych_frame(
    *,
    frame: TriptychFrame,
    output_path: Path,
    context: str,
) -> dict[str, str]:
    if set(frame.paths) != set(VIEW_ORDER):
        raise ValueError(
            f"{context}: frame views={sorted(frame.paths)}, expected {VIEW_ORDER}"
        )
    resolved_paths = {view: Path(frame.paths[view]).resolve() for view in VIEW_ORDER}
    for view, path in resolved_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{context}/{view}: missing frame {path}")

    with (
        Image.open(resolved_paths["left"]) as left,
        Image.open(resolved_paths["right"]) as right,
        Image.open(resolved_paths["wrist"]) as wrist,
    ):
        triptych = make_triptych_from_views(left, right, wrist)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.jpg")
    triptych.save(temporary_path, format="JPEG", quality=95, optimize=True)
    temporary_path.replace(output_path)
    return {view: str(path) for view, path in resolved_paths.items()}


def build_triptych_records(
    *,
    clusters: Sequence[Mapping[str, Any]],
    output_dir: Path,
    frame_provider: TriptychFrameProvider,
    annotation_view_names: Sequence[str],
) -> tuple[list[dict[str, Any]], int, int]:
    """Render cluster triptychs using an adapter for source-specific frame joins."""
    output_records: list[dict[str, Any]] = []
    num_sequences = 0
    num_frames = 0

    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        sample_ids = [str(value) for value in cluster["representative_sample_ids"]]
        source_frame_groups = cluster["representative_frame_paths"]
        if len(sample_ids) != len(source_frame_groups):
            raise ValueError(f"{cluster_id}: representative group count mismatch")

        multiview_groups: list[list[str]] = []
        source_view_groups: list[list[dict[str, str]]] = []
        for sample_id, source_group in zip(
            sample_ids, source_frame_groups, strict=True
        ):
            frames = frame_provider(cluster_id, sample_id, source_group)
            output_group: list[str] = []
            source_view_group: list[dict[str, str]] = []
            for frame in frames:
                output_path = (
                    output_dir
                    / "frames"
                    / cluster_id
                    / sample_id
                    / (
                        f"frame_{frame.position:02d}_"
                        f"v{frame.video_frame_index:04d}_triptych.jpg"
                    )
                )
                source_paths = _render_triptych_frame(
                    frame=frame,
                    output_path=output_path,
                    context=f"{cluster_id}/{sample_id}",
                )
                output_group.append(str(output_path))
                source_view_group.append(source_paths)
                num_frames += 1
            multiview_groups.append(output_group)
            source_view_groups.append(source_view_group)
            num_sequences += 1

        output_record = dict(cluster)
        output_record["representative_single_view_frame_paths"] = source_frame_groups
        output_record["representative_source_view_frame_paths"] = source_view_groups
        output_record["representative_frame_paths"] = multiview_groups
        output_record["annotation_media_layout"] = LAYOUT_ID
        output_record["annotation_view_names"] = list(annotation_view_names)
        output_record["annotation_triptych_size"] = [
            VIEW_WIDTH * len(VIEW_ORDER),
            SCENE_HEIGHT + LABEL_HEIGHT,
        ]
        output_records.append(output_record)

    return output_records, num_sequences, num_frames


def write_multiview_artifacts(
    *,
    output_dir: Path,
    output_records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Write cluster rows and their manifest exactly once."""
    output_clusters_path = output_dir / "clusters_multiview.jsonl"
    with output_clusters_path.open("x", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record) + "\n")

    completed_manifest = {
        **manifest,
        "output_clusters_path": str(output_clusters_path),
        "output_clusters_sha256": _sha256(output_clusters_path),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        manifest_text = json.dumps(completed_manifest, indent=2, sort_keys=True)
        handle.write(manifest_text + "\n")
    return output_clusters_path, manifest_path, completed_manifest


_VIRTUAL_MEDIA_ALIGNMENT_FIELDS = (
    "sample_id",
    "task_id",
    "task_description",
    "prompt_task_description",
    "episode_num",
    "task_episode_idx",
    "cell_id",
    "success",
    "waypoint_rank",
    "waypoint_index",
    "waypoint_step",
    "anchor_source",
    "clip_path",
    "source_trajectory_records_path",
    "source_video_relative_path",
    "source_video_sha256",
    "video_frame_indices",
    "boundary_shift_category",
    "anchor_env_step_error",
)


def _index_virtual_samples(
    path: Path,
    *,
    view: str,
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed = {str(row["sample_id"]): row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f"{view}: virtual samples must be non-empty and unique")
    wrong_views = sorted(
        {
            str(row.get("view"))
            for row in rows
            if str(row.get("view")) != view
        }
    )
    if wrong_views:
        raise ValueError(f"{view}: unexpected row view values: {wrong_views}")
    return indexed


def _verify_sample_alignment(
    sample_id: str,
    rows: dict[str, dict[str, Any]],
) -> None:
    verify_aligned_fields(
        sample_id,
        rows,
        _VIRTUAL_MEDIA_ALIGNMENT_FIELDS,
    )
    frame_counts = {
        view: len(row.get("frame_paths", [])) for view, row in rows.items()
    }
    if (
        len(set(frame_counts.values())) != 1
        or next(iter(frame_counts.values())) == 0
    ):
        raise ValueError(f"{sample_id}: frame counts differ: {frame_counts}")


def _verified_frame(path_value: str, *, sample_id: str, view: str) -> Path:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{sample_id}/{view}: missing frame {path}")
    return path


def _frame_identity(path: Path) -> tuple[int, int, int | None]:
    match = re.fullmatch(r"frame_(\d+)_v(\d+)(?:_s(\d+))?\.jpg", path.name)
    if match is None:
        raise ValueError(f"Unsupported media frame filename: {path.name}")
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)) if match.group(3) is not None else None,
    )


def build_multiview_annotation_triptychs(
    *,
    clusters_path: Path,
    left_samples_path: Path,
    right_samples_path: Path,
    wrist_samples_path: Path,
    output_dir: Path,
    min_episode_coverage: float = 0.3,
) -> dict[str, Any]:
    """Create synchronized triptychs for clusters meeting the coverage threshold."""
    clusters_path = Path(clusters_path).resolve()
    sample_paths = {
        "left": Path(left_samples_path).resolve(),
        "right": Path(right_samples_path).resolve(),
        "wrist": Path(wrist_samples_path).resolve(),
    }
    if not 0.0 <= min_episode_coverage <= 1.0:
        raise ValueError("min_episode_coverage must be between zero and one")
    output_dir = prepare_output_directory(Path(output_dir))

    clusters = [
        row
        for row in load_jsonl(clusters_path)
        if float(row["episode_coverage"]) >= min_episode_coverage
    ]
    if not clusters:
        raise ValueError("No clusters meet min_episode_coverage")
    samples = {
        view: _index_virtual_samples(path, view=view)
        for view, path in sample_paths.items()
    }

    def provide_frames(
        cluster_id: str,
        sample_id: str,
        cluster_left_paths: Sequence[str],
    ) -> list[TriptychFrame]:
        missing = [view for view in VIEW_ORDER if sample_id not in samples[view]]
        if missing:
            raise ValueError(f"{cluster_id}/{sample_id}: missing views {missing}")
        rows = {view: samples[view][sample_id] for view in VIEW_ORDER}
        _verify_sample_alignment(sample_id, rows)
        paths_by_view = {
            view: [
                _verified_frame(value, sample_id=sample_id, view=view)
                for value in rows[view]["frame_paths"]
            ]
            for view in VIEW_ORDER
        }
        resolved_cluster_left = [
            str(Path(value).resolve()) for value in cluster_left_paths
        ]
        if resolved_cluster_left != [str(path) for path in paths_by_view["left"]]:
            raise ValueError(
                f"{cluster_id}/{sample_id}: cluster frames do not match "
                "assembled left-view media"
            )

        frames: list[TriptychFrame] = []
        for frame_paths in zip(
            *(paths_by_view[view] for view in VIEW_ORDER),
            strict=True,
        ):
            identities = [_frame_identity(path) for path in frame_paths]
            if len(set(identities)) != 1:
                raise ValueError(
                    f"{cluster_id}/{sample_id}: view frame identities differ: "
                    f"{identities}"
                )
            position, video_index, _ = identities[0]
            frames.append(
                TriptychFrame(
                    paths={
                        view: path
                        for view, path in zip(VIEW_ORDER, frame_paths, strict=True)
                    },
                    position=position,
                    video_frame_index=video_index,
                )
            )
        return frames

    output_rows, num_sequences, num_frames = build_triptych_records(
        clusters=clusters,
        output_dir=output_dir,
        frame_provider=provide_frames,
        annotation_view_names=VIEW_LABELS,
    )
    manifest = {
        "format": "event_sae_v9_annotation_multiview_v1",
        "media_layout": LAYOUT_ID,
        "view_order": list(VIEW_ORDER),
        "clusters_path": str(clusters_path),
        "clusters_sha256": _sha256(clusters_path),
        "min_episode_coverage": float(min_episode_coverage),
        "virtual_samples": {
            view: {"path": str(path), "sha256": _sha256(path)}
            for view, path in sample_paths.items()
        },
        "alignment_fields": list(_VIRTUAL_MEDIA_ALIGNMENT_FIELDS),
        "num_clusters": len(output_rows),
        "num_sequences": num_sequences,
        "num_frames": num_frames,
        "passed": True,
    }
    _, _, completed_manifest = write_multiview_artifacts(
        output_dir=output_dir,
        output_records=output_rows,
        manifest=manifest,
    )
    return completed_manifest
