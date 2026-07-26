"""Render 5-frame bundle (PNGs + MP4 clip) around each AWE waypoint.

Inputs:
  - `waypoint_summary.json` from `event_sae.keyframes` (G1).
  - The corresponding `trajectory_records.jsonl` (path is stored inside the
    waypoint summary).
  - The rollout MP4s under `<run_dir>/videos/`.

Outputs (under `output_dir`):
  - `frames/<sample_id>/frame_NN_stepMMMM.png` (one image per frame)
  - `clips/<sample_id>.mp4` (a short clip spanning the frame window)
  - `samples.jsonl` (per-sample manifest, one line per waypoint bundle)
  - `skipped_samples.jsonl` (waypoints that could not be rendered)
  - `packaging_report.json`

`samples.jsonl` is the input expected by `event_sae.events.build_features`.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio

from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl, write_jsonl
from event_sae.events.video_timeline import VideoTimeline


@dataclass
class EpisodeRecords:
    episode_num: int
    task_id: int
    task_episode_idx: int
    task_description: str
    prompt_task_description: str
    success: bool
    step_indices: list[int]


def _sanitize_label(label: str, max_len: int = 80) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    if not label:
        return "unnamed"
    return label[:max_len]


def load_waypoint_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "source_trajectory_records_path" not in data or "episodes" not in data:
        raise ValueError(f"Unexpected waypoint summary format: {path}")
    return data


def load_trajectory_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"Unexpected trajectory manifest format: {path}")
    return data


def load_episode_records(path: Path) -> dict[int, EpisodeRecords]:
    grouped: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            grouped.setdefault(int(record["episode_num"]), []).append(record)

    episodes: dict[int, EpisodeRecords] = {}
    for episode_num, records in grouped.items():
        records.sort(key=lambda item: int(item["step_in_episode"]))
        first = records[0]
        episodes[episode_num] = EpisodeRecords(
            episode_num=episode_num,
            task_id=int(first["task_id"]),
            task_episode_idx=int(first["task_episode_idx"]),
            task_description=str(first["task_description"]),
            prompt_task_description=str(
                first.get("prompt_task_description", first["task_description"])
            ),
            success=bool(records[-1]["done"]),
            step_indices=[int(record["step_in_episode"]) for record in records],
        )
    return episodes


def find_episode_video(
    run_dir: Path,
    episode_num: int,
    *,
    episode_manifest: dict | None = None,
    video_root: Path | None = None,
) -> Path | None:
    """Locate an episode video.

    Manifest-backed GR00T lookup is exact and never falls back to a video from
    another experiment. Legacy OpenVLA/OpenPI runs retain their name search.
    """
    if episode_manifest is not None:
        relative_path = episode_manifest.get("source_video_relative_path")
        if not relative_path:
            raise ValueError(
                f"episode {episode_num}: manifest has no source_video_relative_path"
            )
        relative_path = Path(str(relative_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"episode {episode_num}: video path must be manifest-relative"
            )
        root = Path(video_root) if video_root is not None else run_dir
        candidate = root / relative_path
        return candidate if candidate.is_file() else None

    videos_dir = run_dir / "videos"
    if not videos_dir.is_dir():
        return None
    for pattern in (
        f"*episode={episode_num}--*.mp4",
        f"episode_{episode_num:05d}_*.mp4",
    ):
        matches = sorted(videos_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def fit_frame_window(
    *,
    waypoint_index: int,
    frame_offsets: list[int],
    num_frames: int,
) -> tuple[list[int] | None, list[int], int | None]:
    """Shift a fixed-stride frame window so it stays within [0, num_frames).

    Returns (frame_indices, requested_frame_indices, window_shift).
    `frame_indices=None` if the window span exceeds the episode length.
    """
    requested_frame_indices = [waypoint_index + offset for offset in frame_offsets]
    if not requested_frame_indices:
        return [], [], 0

    min_requested = min(requested_frame_indices)
    max_requested = max(requested_frame_indices)
    if (max_requested - min_requested) >= num_frames:
        return None, requested_frame_indices, None

    shift = 0
    if max_requested >= num_frames:
        shift -= max_requested - (num_frames - 1)
    if min_requested + shift < 0:
        shift += -(min_requested + shift)

    frame_indices = [index + shift for index in requested_frame_indices]
    return frame_indices, requested_frame_indices, shift


def _save_frame(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def _write_clip(reader, frame_start: int, frame_end: int, fps: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for frame_idx in range(frame_start, frame_end + 1):
            writer.append_data(reader.get_data(frame_idx))
    finally:
        writer.close()


def _anchor_sources_for_episode(episode: dict) -> dict[int, str]:
    anchors = episode.get("waypoint_anchors")
    if anchors is None:
        return {}
    sources: dict[int, str] = {}
    for anchor in anchors:
        index = int(anchor["waypoint_index"])
        source = str(anchor["anchor_source"])
        if index in sources:
            raise ValueError(f"Duplicate waypoint anchor index: {index}")
        sources[index] = source
    waypoint_indices = {int(index) for index in episode.get("waypoint_indices", [])}
    if set(sources) != waypoint_indices:
        raise ValueError(
            "waypoint_anchors must exactly cover waypoint_indices: "
            f"anchors={sorted(sources)}, waypoints={sorted(waypoint_indices)}"
        )
    return sources


def _write_skipped_episode_waypoints(
    skipped_file,
    *,
    reason: str,
    episode: dict,
    episode_num: int,
    video_path: Path | None = None,
    error: Exception | None = None,
) -> int:
    skipped_count = 0
    for waypoint_rank, waypoint_index in enumerate(episode.get("waypoint_indices", [])):
        record = {
            "reason": reason,
            "episode_num": episode_num,
            "waypoint_rank": waypoint_rank,
            "waypoint_index": int(waypoint_index),
        }
        if video_path is not None:
            record["video_path"] = str(video_path)
        if error is not None:
            record["error"] = repr(error)
        skipped_file.write(json.dumps(record) + "\n")
        skipped_count += 1
    return skipped_count


def _reader_frame_count(reader) -> int | None:
    count_frames = getattr(reader, "count_frames", None)
    if count_frames is None:
        return None
    try:
        return int(count_frames())
    except Exception:
        return None


def extract_keyframe_media(
    waypoint_summary_path: Path,
    output_dir: Path,
    frame_offsets: list[int],
    max_samples: int | None = None,
    *,
    trajectory_manifest_path: Path | None = None,
    video_root: Path | None = None,
    frame_anchor: str = "first",
    require_complete_videos: bool = False,
) -> dict:
    """Render frame PNGs and clips around each waypoint.

    Offsets are always applied in trajectory-record space. A manifest can map
    those records to a denser GR00T video timeline; without one, legacy
    OpenVLA/OpenPI identity mapping is used.
    """
    if frame_anchor not in {"first", "center", "last"}:
        raise ValueError(f"Unsupported frame anchor: {frame_anchor!r}")

    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    summary = load_waypoint_summary(waypoint_summary_path)

    trajectory_records_path = Path(summary["source_trajectory_records_path"]).resolve()
    if not trajectory_records_path.is_file():
        raise FileNotFoundError(f"trajectory_records.jsonl not found: {trajectory_records_path}")
    run_dir = trajectory_records_path.parent
    episode_records = load_episode_records(trajectory_records_path)

    resolved_manifest_path: Path | None
    if trajectory_manifest_path is not None:
        resolved_manifest_path = Path(trajectory_manifest_path).resolve()
        if not resolved_manifest_path.is_file():
            raise FileNotFoundError(
                f"trajectory manifest not found: {resolved_manifest_path}"
            )
    else:
        candidate = trajectory_records_path.parent / "trajectory_manifest.json"
        resolved_manifest_path = candidate if candidate.is_file() else None

    trajectory_manifest = (
        load_trajectory_manifest(resolved_manifest_path)
        if resolved_manifest_path is not None
        else None
    )
    manifest_episodes: dict[int, dict] = {}
    if trajectory_manifest is not None:
        for item in trajectory_manifest["episodes"]:
            episode_num = int(item["episode_num"])
            if episode_num in manifest_episodes:
                raise ValueError(
                    f"Duplicate episode_num={episode_num} in {resolved_manifest_path}"
                )
            manifest_episodes[episode_num] = item

    resolved_video_root: Path | None
    if video_root is not None:
        resolved_video_root = Path(video_root).resolve()
    elif trajectory_manifest is not None:
        resolved_video_root = Path(str(trajectory_manifest["source_root"])).expanduser()
    else:
        resolved_video_root = None

    if trajectory_manifest is not None:
        summary_episode_nums = {
            int(episode["episode_num"]) for episode in summary["episodes"]
        }
        missing_manifest_episodes = sorted(summary_episode_nums - set(manifest_episodes))
        if missing_manifest_episodes:
            raise ValueError(
                "Waypoint summary episodes missing from trajectory manifest: "
                f"{missing_manifest_episodes[:10]}"
            )

    if require_complete_videos:
        missing_videos: list[int] = []
        for episode in summary["episodes"]:
            episode_num = int(episode["episode_num"])
            manifest_episode = manifest_episodes.get(episode_num)
            video_path = find_episode_video(
                run_dir,
                episode_num,
                episode_manifest=manifest_episode,
                video_root=resolved_video_root,
            )
            if video_path is None:
                missing_videos.append(episode_num)
        if missing_videos:
            raise FileNotFoundError(
                f"Missing exact episode videos for {len(missing_videos)} episodes; "
                f"first episode_num={missing_videos[0]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"
    clips_root = output_dir / "clips"
    samples_path = output_dir / "samples.jsonl"
    skipped_path = output_dir / "skipped_samples.jsonl"

    sample_count = 0
    skipped_count = 0
    with samples_path.open("w", encoding="utf-8") as samples_file, skipped_path.open(
        "w", encoding="utf-8"
    ) as skipped_file:
        for episode in summary["episodes"]:
            if max_samples is not None and sample_count >= max_samples:
                break

            episode_num = int(episode["episode_num"])
            anchor_sources = _anchor_sources_for_episode(episode)
            episode_meta = episode_records.get(episode_num)
            if episode_meta is None:
                skipped_file.write(
                    json.dumps(
                        {"reason": "missing_episode_records", "episode_num": episode_num}
                    )
                    + "\n"
                )
                skipped_count += 1
                continue

            manifest_episode = manifest_episodes.get(episode_num)
            timeline = (
                VideoTimeline.from_episode_manifest(manifest_episode)
                if manifest_episode is not None
                else VideoTimeline(num_records=len(episode_meta.step_indices))
            )
            if timeline.num_records != len(episode_meta.step_indices):
                raise ValueError(
                    f"episode {episode_num}: manifest num_records={timeline.num_records}, "
                    f"JSONL records={len(episode_meta.step_indices)}"
                )

            video_path = find_episode_video(
                run_dir,
                episode_num,
                episode_manifest=manifest_episode,
                video_root=resolved_video_root,
            )
            if video_path is None:
                skipped_count += _write_skipped_episode_waypoints(
                    skipped_file,
                    reason="missing_episode_video",
                    episode=episode,
                    episode_num=episode_num,
                )
                continue

            reader = None
            try:
                reader = imageio.get_reader(video_path)
                fps = float(reader.get_meta_data().get("fps", 20.0))
            except Exception as exc:
                skipped_count += _write_skipped_episode_waypoints(
                    skipped_file,
                    reason="unreadable_episode_video",
                    episode=episode,
                    episode_num=episode_num,
                    video_path=video_path,
                    error=exc,
                )
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                continue

            try:
                actual_num_frames = _reader_frame_count(reader)
                if (
                    manifest_episode is not None
                    and actual_num_frames is not None
                    and actual_num_frames != timeline.expected_num_frames
                ):
                    skipped_count += _write_skipped_episode_waypoints(
                        skipped_file,
                        reason="video_frame_count_mismatch",
                        episode=episode,
                        episode_num=episode_num,
                        video_path=video_path,
                        error=ValueError(
                            f"actual={actual_num_frames}, "
                            f"expected={timeline.expected_num_frames}"
                        ),
                    )
                    continue

                for waypoint_rank, waypoint_index in enumerate(
                    episode["waypoint_indices"]
                ):
                    if max_samples is not None and sample_count >= max_samples:
                        break

                    waypoint_index = int(waypoint_index)
                    if not 0 <= waypoint_index < timeline.num_records:
                        skipped_file.write(
                            json.dumps(
                                {
                                    "reason": "waypoint_index_out_of_range",
                                    "episode_num": episode_num,
                                    "waypoint_rank": waypoint_rank,
                                    "waypoint_index": waypoint_index,
                                    "num_records": timeline.num_records,
                                }
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue

                    (
                        record_indices,
                        requested_record_indices,
                        record_window_shift,
                    ) = fit_frame_window(
                        waypoint_index=waypoint_index,
                        frame_offsets=frame_offsets,
                        num_frames=timeline.num_records,
                    )
                    if record_indices is None:
                        skipped_file.write(
                            json.dumps(
                                {
                                    "reason": "window_span_exceeds_episode",
                                    "episode_num": episode_num,
                                    "waypoint_rank": waypoint_rank,
                                    "waypoint_index": waypoint_index,
                                    "requested_record_indices": requested_record_indices,
                                    "num_records": timeline.num_records,
                                }
                            )
                            + "\n"
                        )
                        skipped_count += 1
                        continue

                    frame_indices = [
                        timeline.record_to_frame(index, anchor=frame_anchor)
                        for index in record_indices
                    ]
                    requested_frame_indices = [
                        (
                            timeline.record_to_frame(index, anchor=frame_anchor)
                            if 0 <= index < timeline.num_records
                            else None
                        )
                        for index in requested_record_indices
                    ]
                    waypoint_step = episode_meta.step_indices[waypoint_index]
                    sample_id = (
                        f"ep{episode_num:04d}_task{episode_meta.task_id:02d}_"
                        f"wp{waypoint_rank:02d}_step{waypoint_step:04d}_"
                        f"{_sanitize_label(episode_meta.task_description, max_len=36)}"
                    )
                    sample_frame_dir = frames_root / sample_id
                    frame_paths: list[str] = []
                    frame_steps = [
                        episode_meta.step_indices[index] for index in record_indices
                    ]
                    for frame_pos, (frame_index, frame_step) in enumerate(
                        zip(frame_indices, frame_steps, strict=True)
                    ):
                        frame = reader.get_data(frame_index)
                        frame_path = (
                            sample_frame_dir
                            / f"frame_{frame_pos:02d}_video{frame_index:04d}_"
                            f"step{frame_step:04d}.png"
                        )
                        _save_frame(frame, frame_path)
                        frame_paths.append(str(frame_path))

                    clip_path = clips_root / f"{sample_id}.mp4"
                    _write_clip(
                        reader=reader,
                        frame_start=min(frame_indices),
                        frame_end=max(frame_indices),
                        fps=fps,
                        output_path=clip_path,
                    )

                    timeline_record = timeline.to_dict()
                    timeline_record["frame_anchor"] = frame_anchor
                    sample_record = {
                        "sample_id": sample_id,
                        "run_dir": str(run_dir),
                        "source_waypoint_summary_path": str(waypoint_summary_path),
                        "source_trajectory_records_path": str(trajectory_records_path),
                        "source_trajectory_manifest_path": (
                            str(resolved_manifest_path)
                            if resolved_manifest_path is not None
                            else None
                        ),
                        "source_video_path": str(video_path),
                        "episode_num": episode_num,
                        "task_id": episode_meta.task_id,
                        "task_episode_idx": episode_meta.task_episode_idx,
                        "task_description": episode_meta.task_description,
                        "prompt_task_description": episode_meta.prompt_task_description,
                        "success": episode_meta.success,
                        "waypoint_rank": waypoint_rank,
                        "waypoint_index": waypoint_index,
                        "waypoint_step": waypoint_step,
                        "record_offsets": list(frame_offsets),
                        "requested_record_indices": requested_record_indices,
                        "record_indices": record_indices,
                        "record_window_shift": record_window_shift,
                        "video_timeline": timeline_record,
                        "requested_frame_indices": requested_frame_indices,
                        "frame_indices": frame_indices,
                        "frame_steps": frame_steps,
                        "actual_video_num_frames": actual_num_frames,
                        "frame_paths": frame_paths,
                        "clip_path": str(clip_path),
                        "frame_offsets": list(frame_offsets),
                        "window_shift": record_window_shift,
                    }
                    if anchor_sources:
                        sample_record["anchor_source"] = anchor_sources[
                            waypoint_index
                        ]
                    samples_file.write(json.dumps(sample_record) + "\n")
                    samples_file.flush()
                    sample_count += 1
            finally:
                reader.close()

    report = {
        "waypoint_summary_path": str(waypoint_summary_path),
        "source_trajectory_records_path": str(trajectory_records_path),
        "source_trajectory_manifest_path": (
            str(resolved_manifest_path)
            if resolved_manifest_path is not None
            else None
        ),
        "run_dir": str(run_dir),
        "video_root": (
            str(resolved_video_root) if resolved_video_root is not None else None
        ),
        "output_dir": str(output_dir),
        "record_offsets": list(frame_offsets),
        "frame_anchor": frame_anchor,
        "require_complete_videos": require_complete_videos,
        "num_samples": sample_count,
        "num_skipped": skipped_count,
        "samples_path": str(samples_path),
        "skipped_samples_path": str(skipped_path),
    }
    report_path = output_dir / "packaging_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


REUSABLE_ANCHOR_SOURCES = {"position", "both"}
SUPPLEMENT_ANCHOR_SOURCE = "gripper_close"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _filter_aligned_rows(
    rows: list,
    waypoint_indices: list[int],
    selected: set[int],
) -> list:
    if len(rows) != len(waypoint_indices):
        raise ValueError(
            "Waypoint-aligned field length mismatch: "
            f"rows={len(rows)}, waypoints={len(waypoint_indices)}"
        )
    return [
        copy.deepcopy(row)
        for waypoint_index, row in zip(waypoint_indices, rows, strict=True)
        if waypoint_index in selected
    ]


def build_gripper_supplement_summary(
    *,
    waypoint_summary_path: Path,
    output_path: Path,
    expected_waypoints: int | None = None,
) -> dict:
    """Write a media-only waypoint summary containing gripper-only anchors."""
    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite supplement summary: {output_path}")
    summary = json.loads(waypoint_summary_path.read_text(encoding="utf-8"))
    filtered = copy.deepcopy(summary)
    total_waypoints = 0
    episodes_with_waypoints = 0

    for source_episode, output_episode in zip(
        summary["episodes"],
        filtered["episodes"],
        strict=True,
    ):
        waypoint_indices = [
            int(value) for value in source_episode["waypoint_indices"]
        ]
        anchors = source_episode.get("waypoint_anchors")
        if not isinstance(anchors, list) or len(anchors) != len(waypoint_indices):
            raise ValueError(
                f"episode {source_episode['episode_num']}: waypoint_anchors must "
                "exactly align with waypoint_indices"
            )
        anchor_by_index = {
            int(anchor["waypoint_index"]): str(anchor["anchor_source"])
            for anchor in anchors
        }
        if set(anchor_by_index) != set(waypoint_indices):
            raise ValueError(
                f"episode {source_episode['episode_num']}: anchor key mismatch"
            )
        selected_indices = [
            index
            for index in waypoint_indices
            if anchor_by_index[index] == SUPPLEMENT_ANCHOR_SOURCE
        ]
        selected = set(selected_indices)
        output_episode["waypoint_indices"] = selected_indices
        output_episode["waypoint_anchors"] = [
            copy.deepcopy(anchor)
            for anchor in anchors
            if int(anchor["waypoint_index"]) in selected
        ]
        for field in ("waypoint_positions", "waypoint_gripper_state"):
            if field in source_episode:
                output_episode[field] = _filter_aligned_rows(
                    source_episode[field],
                    waypoint_indices,
                    selected,
                )
        output_episode["num_waypoints"] = len(selected_indices)
        output_episode["position_waypoint_indices"] = []
        output_episode["gripper_close_indices"] = selected_indices
        total_waypoints += len(selected_indices)
        episodes_with_waypoints += bool(selected_indices)

    if expected_waypoints is not None and total_waypoints != expected_waypoints:
        raise ValueError(
            f"Expected {expected_waypoints} gripper-only waypoints, found {total_waypoints}"
        )
    filtered["format"] = "event_sae_waypoint_media_subset_v1"
    filtered["media_subset"] = {
        "anchor_source": SUPPLEMENT_ANCHOR_SOURCE,
        "source_waypoint_summary_path": str(waypoint_summary_path),
        "source_waypoint_summary_sha256": _sha256(waypoint_summary_path),
        "num_episodes": len(filtered["episodes"]),
        "num_episodes_with_waypoints": episodes_with_waypoints,
        "num_waypoints": total_waypoints,
    }
    filtered["output_dir"] = str(output_path.parent)
    _write_json(output_path, filtered)
    return filtered["media_subset"]


def _index_rendered_media_samples(
    samples_path: Path,
) -> dict[tuple[int, int], dict]:
    samples_path = Path(samples_path).resolve()
    rows = load_jsonl(samples_path)
    indexed: dict[tuple[int, int], dict] = {}
    for row in rows:
        if row.get("format") != "event_sae_stage3_media_v4":
            raise ValueError(f"Unsupported media format in {samples_path}")
        key = (int(row["episode_num"]), int(row["waypoint_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate media key {key} in {samples_path}")
        indexed[key] = row
    return indexed


def _resolve_rendered_media_frames(
    samples_path: Path,
    sample: dict,
) -> list[str]:
    bundle_dir = samples_path.parent
    frames = sorted(sample["frames"], key=lambda row: int(row["position"]))
    positions = [int(row["position"]) for row in frames]
    if positions != list(range(len(frames))):
        raise ValueError(
            f"{sample['sample_id']}: non-contiguous frame positions {positions}"
        )
    paths = [(bundle_dir / str(row["path"])).resolve() for row in frames]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{sample['sample_id']}: missing frame {missing[0]}")
    for path, frame in zip(paths, frames, strict=True):
        expected_hash = frame.get("sha256")
        if expected_hash is not None and _sha256(path) != str(expected_hash):
            raise ValueError(f"{sample['sample_id']}: frame hash mismatch {path}")
    return [str(path) for path in paths]


def assemble_composite_anchor_media_view(
    *,
    waypoint_summary_path: Path,
    reusable_samples_path: Path,
    supplement_samples_path: Path,
    trajectory_records_path: Path,
    output_path: Path,
    view: str,
    expected_samples: int | None = None,
) -> dict:
    """Join reusable position anchors and gripper supplements for one view."""
    waypoint_summary_path = Path(waypoint_summary_path).resolve()
    reusable_samples_path = Path(reusable_samples_path).resolve()
    supplement_samples_path = Path(supplement_samples_path).resolve()
    trajectory_records_path = Path(trajectory_records_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite composite media output: {output_path}"
        )
    if not trajectory_records_path.is_file():
        raise FileNotFoundError(trajectory_records_path)

    summary = json.loads(waypoint_summary_path.read_text(encoding="utf-8"))
    reusable = _index_rendered_media_samples(reusable_samples_path)
    supplement = _index_rendered_media_samples(supplement_samples_path)
    records: list[dict] = []
    source_counts: Counter[str] = Counter()
    seen_keys: set[tuple[int, int]] = set()

    for episode in summary["episodes"]:
        episode_num = int(episode["episode_num"])
        anchors = episode.get("waypoint_anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"episode {episode_num}: missing waypoint_anchors")
        anchor_by_index = {
            int(anchor["waypoint_index"]): str(anchor["anchor_source"])
            for anchor in anchors
        }
        waypoint_indices = [int(value) for value in episode["waypoint_indices"]]
        if set(anchor_by_index) != set(waypoint_indices):
            raise ValueError(f"episode {episode_num}: waypoint anchor mismatch")

        for waypoint_rank, waypoint_index in enumerate(waypoint_indices):
            key = (episode_num, waypoint_index)
            if key in seen_keys:
                raise ValueError(f"Duplicate composite media key: {key}")
            seen_keys.add(key)
            anchor_source = anchor_by_index[waypoint_index]
            if anchor_source in REUSABLE_ANCHOR_SOURCES:
                source_kind = "abs_position_reuse"
                source_samples_path = reusable_samples_path
                source_sample = reusable.get(key)
            elif anchor_source == SUPPLEMENT_ANCHOR_SOURCE:
                source_kind = "gripper_close_supplement"
                source_samples_path = supplement_samples_path
                source_sample = supplement.get(key)
            else:
                raise ValueError(f"Unsupported anchor_source={anchor_source!r}")
            if source_sample is None:
                raise ValueError(
                    f"Missing {source_kind} media for episode={episode_num}, "
                    f"waypoint={waypoint_index}"
                )
            frame_paths = _resolve_rendered_media_frames(
                source_samples_path,
                source_sample,
            )
            sample_id = (
                f"ep{episode_num:04d}_wp{waypoint_rank:03d}_r{waypoint_index:04d}"
            )
            record = {
                "sample_id": sample_id,
                "task_id": int(episode["task_id"]),
                "task_description": str(episode["task_description"]),
                "prompt_task_description": str(
                    episode.get(
                        "prompt_task_description",
                        episode["task_description"],
                    )
                ),
                "episode_num": episode_num,
                "task_episode_idx": int(episode["task_episode_idx"]),
                "cell_id": source_sample.get("cell_id"),
                "success": bool(episode["success"]),
                "waypoint_rank": waypoint_rank,
                "waypoint_index": waypoint_index,
                "waypoint_step": waypoint_index,
                "anchor_source": anchor_source,
                "frame_paths": frame_paths,
                "clip_path": str(
                    source_sample.get("source_video_relative_path", "")
                ),
                "source_trajectory_records_path": str(trajectory_records_path),
                "source_media_format": source_sample["format"],
                "source_media_kind": source_kind,
                "source_media_samples_path": str(source_samples_path),
                "source_media_samples_sha256": _sha256(source_samples_path),
                "source_media_sample_id": str(source_sample["sample_id"]),
                "source_video_relative_path": source_sample.get(
                    "source_video_relative_path"
                ),
                "source_video_sha256": source_sample.get("source_video_sha256"),
                "video_frame_indices": source_sample.get("video_frame_indices"),
                "boundary_shift_category": source_sample.get(
                    "boundary_shift_category"
                ),
                "anchor_env_step_error": source_sample.get(
                    "anchor_env_step_error"
                ),
                "view": view,
            }
            records.append(record)
            source_counts[source_kind] += 1

    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} composite samples, "
            f"found {len(records)}"
        )
    if source_counts["abs_position_reuse"] != len(reusable):
        raise ValueError(
            "Reusable abs media does not exactly match position/both anchors: "
            f"used={source_counts['abs_position_reuse']}, available={len(reusable)}"
        )
    if source_counts["gripper_close_supplement"] != len(supplement):
        raise ValueError(
            "Supplement media does not exactly match gripper-only anchors: "
            f"used={source_counts['gripper_close_supplement']}, "
            f"available={len(supplement)}"
        )

    write_jsonl(output_path, records)
    manifest = {
        "format": "event_sae_v9_virtual_media_v1",
        "view": view,
        "waypoint_summary_path": str(waypoint_summary_path),
        "waypoint_summary_sha256": _sha256(waypoint_summary_path),
        "trajectory_records_path": str(trajectory_records_path),
        "trajectory_records_sha256": _sha256(trajectory_records_path),
        "reusable_samples_path": str(reusable_samples_path),
        "reusable_samples_sha256": _sha256(reusable_samples_path),
        "supplement_samples_path": str(supplement_samples_path),
        "supplement_samples_sha256": _sha256(supplement_samples_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "num_samples": len(records),
        "frames_per_sample": len(records[0]["frame_paths"]) if records else 0,
        "source_counts": dict(sorted(source_counts.items())),
        "copies_frame_files": False,
        "passed": True,
    }
    _write_json(manifest_path, manifest)
    return manifest
