"""Materialize phase-contained rollout frames for oracle keyframe descriptors.

The oracle phase extractor operates at simulator-state resolution, while the
saved numerical robot state is available once per policy inference record.
Media samples produced here therefore use two explicit clocks:

* ``waypoint_step`` / ``descriptor_record_index`` select the saved state used
  by :func:`event_sae.events.build_features.build_event_features`.
* ``state_env_step_index`` identifies the simulator-oracle keyframe.

Rendered frames follow the exporter contract
``frame f -> state env_step 1 + f * steps_per_render``.  For even-indexed states the
first post-transition frame can be one simulator step late.  The exact signed
error and phase-containment status are serialized for every sample.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from event_sae import sha256_file
from event_sae.events.io import load_jsonl, write_jsonl


FORMAT = "event_sae_oracle_phase_media_v1"
SAMPLES_NAME = "samples.jsonl"
REPORT_NAME = "packaging_report.json"

FIRST_VIDEO_FRAME_ENV_STEP = 1
VIEW_LAYOUT = {
    "left": (0, "robot0_agentview_left"),
    "right": (1, "robot0_agentview_right"),
    "wrist": (2, "robot0_eye_in_hand"),
}


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def select_phase_frame_indices(
    *,
    state_env_step_index: int,
    phase_segment_stop: int,
    steps_per_render: int,
    video_num_frames: int,
    frames_per_sample: int,
) -> tuple[list[int], list[int], bool]:
    """Choose forward rendered frames contained in one oracle phase segment.

    The segment is half-open: ``[state_env_step_index, phase_segment_stop)``.
    If it contains fewer frames than requested, the last valid frame is
    repeated.  A sub-render phase may have no rendered frame of its own; in that
    case the nearest available frame is used and the returned phase-match flag
    is false.
    """

    if state_env_step_index < 0:
        raise ValueError("state_env_step_index must be non-negative")
    if phase_segment_stop <= state_env_step_index:
        raise ValueError("phase_segment_stop must be after the keyframe state")
    if steps_per_render <= 0:
        raise ValueError("steps_per_render must be positive")
    if video_num_frames <= 0:
        raise ValueError("video_num_frames must be positive")
    if frames_per_sample <= 0:
        raise ValueError("frames_per_sample must be positive")

    first_frame = _ceil_div(
        state_env_step_index - FIRST_VIDEO_FRAME_ENV_STEP,
        steps_per_render,
    )
    first_frame = max(first_frame, 0)
    last_phase_frame = (
        phase_segment_stop - 1 - FIRST_VIDEO_FRAME_ENV_STEP
    ) // steps_per_render
    last_video_frame = video_num_frames - 1
    valid_last = min(last_phase_frame, last_video_frame)
    candidates = (
        list(range(first_frame, valid_last + 1))
        if first_frame <= valid_last
        else []
    )
    phase_match = bool(candidates)
    if not candidates:
        relative_step = state_env_step_index - FIRST_VIDEO_FRAME_ENV_STEP
        if relative_step <= 0:
            nearest = 0
        else:
            nearest, remainder = divmod(relative_step, steps_per_render)
            if remainder * 2 >= steps_per_render:
                nearest += 1
            nearest = min(nearest, last_video_frame)
        candidates = [nearest]

    frame_indices = candidates[:frames_per_sample]
    frame_indices.extend(
        [frame_indices[-1]] * (frames_per_sample - len(frame_indices))
    )
    represented_env_steps = [
        FIRST_VIDEO_FRAME_ENV_STEP + frame_index * steps_per_render
        for frame_index in frame_indices
    ]
    return frame_indices, represented_env_steps, phase_match


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Trajectory manifest has no episodes: {path}")
    return payload


def _episode_index(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for episode in manifest["episodes"]:
        episode_num = int(episode["episode_num"])
        if episode_num in output:
            raise ValueError(
                f"Duplicate episode_num={episode_num} in trajectory manifest"
            )
        output[episode_num] = episode
    return output


def _safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path


def _video_frame_count(reader: Any) -> int:
    count_frames = getattr(reader, "count_frames", None)
    if count_frames is None:
        raise ValueError("Video reader does not expose count_frames()")
    frame_count = int(count_frames())
    if frame_count <= 0:
        raise ValueError(f"Invalid video frame count: {frame_count}")
    return frame_count


def _phase_segment_stops(
    episode_events: list[dict[str, Any]],
) -> dict[str, int]:
    """Infer the next phase-entry state for every selected event."""

    ordered = sorted(
        episode_events,
        key=lambda row: (
            int(row["state_env_step_index"]),
            int(row["waypoint_rank"]),
        ),
    )
    transitions = [
        row for row in ordered if bool(row.get("is_phase_transition"))
    ]
    if not transitions or int(transitions[0]["state_env_step_index"]) != 0:
        raise ValueError(
            f"episode {ordered[0]['episode_num']}: oracle media requires "
            "phase-entry anchors including state s0"
        )
    transition_steps = [
        int(row["state_env_step_index"]) for row in transitions
    ]
    if transition_steps != sorted(set(transition_steps)):
        raise ValueError(
            f"episode {ordered[0]['episode_num']}: duplicate phase transitions"
        )

    num_steps = int(ordered[0]["num_steps"])
    stops: dict[str, int] = {}
    for event in ordered:
        step = int(event["state_env_step_index"])
        following = [
            transition_step
            for transition_step in transition_steps
            if transition_step > step
        ]
        stops[str(event["sample_id"])] = (
            following[0] if following else num_steps + 1
        )
    return stops


def materialize_oracle_phase_media(
    *,
    oracle_events_path: Path,
    trajectory_manifest_path: Path,
    output_dir: Path,
    video_root: Path | None = None,
    trajectory_records_path: Path | None = None,
    frames_per_sample: int = 5,
    view_name: str = "left",
    scene_height_pixels: int = 256,
    view_width_pixels: int = 256,
    jpeg_quality: int = 90,
    expected_samples: int | None = None,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Decode phase-contained keyframes and emit feature-builder samples."""

    oracle_events_path = Path(oracle_events_path).resolve()
    trajectory_manifest_path = Path(trajectory_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    for label, path in (
        ("oracle events", oracle_events_path),
        ("trajectory manifest", trajectory_manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty media directory: {output_dir}"
        )

    if view_name not in VIEW_LAYOUT:
        raise ValueError(
            f"view_name must be one of {sorted(VIEW_LAYOUT)}, got {view_name!r}"
        )
    if scene_height_pixels <= 0 or view_width_pixels <= 0:
        raise ValueError("view crop dimensions must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")

    manifest = _load_manifest(trajectory_manifest_path)
    manifest_episodes = _episode_index(manifest)
    events = load_jsonl(oracle_events_path)
    if expected_samples is not None and len(events) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} oracle events, found {len(events)}"
        )
    sample_ids = [str(row["sample_id"]) for row in events]
    if not events or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Oracle events must be non-empty with unique sample_id values"
        )
    if any(row.get("oracle_upper_bound") is not True for row in events):
        raise ValueError("Every media event must be a simulator-oracle label")

    if video_root is None:
        video_root = Path(str(manifest["source_root"])).expanduser()
    video_root = Path(video_root).resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    if trajectory_records_path is None:
        records_file = manifest.get("trajectory_records_file")
        if not records_file:
            raise ValueError(
                "trajectory_records_path is required when the manifest has no "
                "trajectory_records_file"
            )
        trajectory_records_path = (
            trajectory_manifest_path.parent
            / _safe_relative_path(
                records_file,
                label="trajectory_records_file",
            )
        )
    trajectory_records_path = Path(trajectory_records_path).resolve()
    if not trajectory_records_path.is_file():
        raise FileNotFoundError(
            f"Trajectory records not found: {trajectory_records_path}"
        )

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        episode_num = int(event["episode_num"])
        if episode_num not in manifest_episodes:
            raise ValueError(
                f"episode_num={episode_num} missing from trajectory manifest"
            )
        by_episode[episode_num].append(event)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"
    samples: list[dict[str, Any]] = []
    mismatch_samples: list[str] = []
    padded_samples: list[str] = []
    for episode_index, (episode_num, episode_events) in enumerate(
        sorted(by_episode.items()),
        start=1,
    ):
        episode = manifest_episodes[episode_num]
        for event in episode_events:
            for field in (
                "task_id",
                "task_episode_idx",
                "task_description",
            ):
                if str(event[field]) != str(episode[field]):
                    raise ValueError(
                        f"{event['sample_id']}: {field} mismatch between "
                        "oracle event and trajectory manifest"
                    )

        video_relative_path = _safe_relative_path(
            episode["source_video_relative_path"],
            label=f"episode {episode_num} source_video_relative_path",
        )
        video_path = video_root / video_relative_path
        if not video_path.is_file():
            raise FileNotFoundError(
                f"episode {episode_num}: exact video not found: {video_path}"
            )
        steps_per_render = int(episode["steps_per_render"])
        if steps_per_render <= 0:
            raise ValueError(
                f"episode {episode_num}: steps_per_render must be positive"
            )
        segment_stops = _phase_segment_stops(episode_events)
        reader = imageio.get_reader(video_path)
        try:
            video_num_frames = _video_frame_count(reader)
            expected_video_frames = int(episode["expected_video_frames"])
            if video_num_frames != expected_video_frames:
                raise ValueError(
                    f"episode {episode_num}: video frames={video_num_frames} "
                    f"!= manifest expected={expected_video_frames}"
                )
            frame_cache: dict[int, Any] = {}
            for event in sorted(
                episode_events,
                key=lambda row: (
                    int(row["state_env_step_index"]),
                    int(row["waypoint_rank"]),
                ),
            ):
                sample_id = str(event["sample_id"])
                state_env_step_index = int(event["state_env_step_index"])
                frame_indices, frame_env_steps, phase_match = (
                    select_phase_frame_indices(
                        state_env_step_index=state_env_step_index,
                        phase_segment_stop=segment_stops[sample_id],
                        steps_per_render=steps_per_render,
                        video_num_frames=video_num_frames,
                        frames_per_sample=frames_per_sample,
                    )
                )
                if not phase_match:
                    mismatch_samples.append(sample_id)
                if len(set(frame_indices)) < len(frame_indices):
                    padded_samples.append(sample_id)

                frame_paths: list[str] = []
                sample_frame_dir = frames_root / sample_id
                for frame_position, (
                    frame_index,
                    frame_env_step,
                ) in enumerate(
                    zip(frame_indices, frame_env_steps, strict=True)
                ):
                    if frame_index not in frame_cache:
                        frame_cache[frame_index] = reader.get_data(frame_index)
                    frame_path = sample_frame_dir / (
                        f"frame_{frame_position:02d}_video{frame_index:04d}_"
                        f"env{frame_env_step:04d}.jpg"
                    )
                    image = Image.fromarray(frame_cache[frame_index]).convert("RGB")
                    source_width, source_height = image.size
                    expected_width = view_width_pixels * len(VIEW_LAYOUT)
                    if source_width != expected_width:
                        raise ValueError(
                            f"{sample_id}: source width={source_width} "
                            f"!= expected montage width={expected_width}"
                        )
                    if source_height < scene_height_pixels:
                        raise ValueError(
                            f"{sample_id}: source height={source_height} "
                            f"< scene height={scene_height_pixels}"
                        )
                    view_index, source_camera = VIEW_LAYOUT[view_name]
                    crop_top_pixels = source_height - scene_height_pixels
                    crop_left_pixels = view_index * view_width_pixels
                    image = image.crop(
                        (
                            crop_left_pixels,
                            crop_top_pixels,
                            crop_left_pixels + view_width_pixels,
                            source_height,
                        )
                    )
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path = frame_path.with_suffix(".tmp.jpg")
                    image.save(
                        temporary_path,
                        format="JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                    )
                    temporary_path.replace(frame_path)
                    frame_paths.append(str(frame_path))

                anchor_error = frame_env_steps[0] - state_env_step_index
                descriptor_record_index = int(
                    event["activation_record_index"]
                )
                samples.append(
                    {
                        "format": FORMAT,
                        "sample_id": sample_id,
                        "source_oracle_events_path": str(
                            oracle_events_path
                        ),
                        "source_trajectory_manifest_path": str(
                            trajectory_manifest_path
                        ),
                        "source_trajectory_records_path": str(
                            trajectory_records_path
                        ),
                        "source_video_path": str(video_path),
                        "source_video_relative_path": (
                            video_relative_path.as_posix()
                        ),
                        "episode_num": episode_num,
                        "task_id": int(event["task_id"]),
                        "task_episode_idx": int(
                            event["task_episode_idx"]
                        ),
                        "task_description": str(
                            event["task_description"]
                        ),
                        "prompt_task_description": str(
                            event["prompt_task_description"]
                        ),
                        "cell_id": event.get("cell_id"),
                        "success": bool(event["success"]),
                        "waypoint_rank": int(event["waypoint_rank"]),
                        # Existing feature-builder clock.
                        "waypoint_step": descriptor_record_index,
                        "waypoint_index": descriptor_record_index,
                        "descriptor_record_index": descriptor_record_index,
                        # Oracle/scorer clock retained as provenance.
                        "state_env_step_index": state_env_step_index,
                        "activation_env_step_index": int(
                            event["activation_env_step_index"]
                        ),
                        "action_token_offset": int(
                            event["action_token_offset"]
                        ),
                        "phase": str(event["phase"]),
                        "phase_scheme": str(event["phase_scheme"]),
                        "oracle_upper_bound": True,
                        "anchor_source": str(event["anchor_source"]),
                        "phase_segment_stop": int(
                            segment_stops[sample_id]
                        ),
                        "steps_per_render": steps_per_render,
                        "video_num_frames": video_num_frames,
                        "frame_indices": frame_indices,
                        "frame_env_step_indices": frame_env_steps,
                        "frame_paths": frame_paths,
                        "clip_path": str(video_path),
                        "view_name": view_name,
                        "view_index": int(view_index),
                        "source_camera": str(source_camera),
                        "source_frame_width": int(source_width),
                        "source_frame_height": int(source_height),
                        "crop_top_pixels": int(crop_top_pixels),
                        "crop_left_pixels": int(crop_left_pixels),
                        "crop_right_pixels": int(
                            source_width - crop_left_pixels - view_width_pixels
                        ),
                        "anchor_frame_env_step_index": frame_env_steps[0],
                        "anchor_env_step_error": anchor_error,
                        "frames_within_oracle_phase": phase_match,
                        "phase_frame_padding_count": (
                            len(frame_indices) - len(set(frame_indices))
                        ),
                    }
                )
        finally:
            reader.close()

        if progress_every > 0 and (
            episode_index % progress_every == 0
            or episode_index == len(by_episode)
        ):
            print(
                f"[oracle-media] episodes={episode_index}/{len(by_episode)} "
                f"samples={len(samples)}",
                flush=True,
            )

    samples.sort(
        key=lambda row: (
            int(row["episode_num"]),
            int(row["state_env_step_index"]),
            str(row["sample_id"]),
        )
    )
    samples_path = output_dir / SAMPLES_NAME
    write_jsonl(samples_path, samples)
    report = {
        "format": FORMAT,
        "oracle_events_path": str(oracle_events_path),
        "oracle_events_sha256": sha256_file(oracle_events_path),
        "trajectory_manifest_path": str(trajectory_manifest_path),
        "trajectory_manifest_sha256": sha256_file(
            trajectory_manifest_path
        ),
        "trajectory_records_path": str(trajectory_records_path),
        "trajectory_records_sha256": sha256_file(
            trajectory_records_path
        ),
        "video_root": str(video_root),
        "samples_path": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "num_episodes": len(by_episode),
        "num_samples": len(samples),
        "frames_per_sample": frames_per_sample,
        "view_name": view_name,
        "view_index": int(VIEW_LAYOUT[view_name][0]),
        "source_camera": str(VIEW_LAYOUT[view_name][1]),
        "scene_height_pixels": scene_height_pixels,
        "view_width_pixels": view_width_pixels,
        "jpeg_quality": jpeg_quality,
        "first_video_frame_env_step": FIRST_VIDEO_FRAME_ENV_STEP,
        "num_phase_mismatch_samples": len(mismatch_samples),
        "phase_mismatch_sample_ids": mismatch_samples,
        "num_padded_samples": len(padded_samples),
        "alignment": (
            "frame f represents state env_step=1+f*steps_per_render; "
            "first forward frame inside oracle phase; nearest ties later"
        ),
        "passed": True,
    }
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
