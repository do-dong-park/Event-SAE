import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_extract_media():
    repo = Path(__file__).resolve().parents[1]
    timeline_name = "event_sae.events.video_timeline"
    media_name = "standalone_extract_media"

    saved = {
        key: sys.modules.get(key)
        for key in (
            "event_sae",
            "event_sae.events",
            timeline_name,
            "imageio",
            "imageio.v2",
            media_name,
        )
    }
    root_module = saved["event_sae"] or types.ModuleType("event_sae")
    events_module = types.ModuleType("event_sae.events")
    events_module.__path__ = []
    imageio_module = types.ModuleType("imageio")
    imageio_module.__path__ = []
    imageio_v2 = types.ModuleType("imageio.v2")

    try:
        sys.modules["event_sae"] = root_module
        sys.modules["event_sae.events"] = events_module
        setattr(root_module, "events", events_module)
        timeline = _load_module(
            repo / "event_sae" / "events" / "video_timeline.py",
            timeline_name,
        )
        setattr(events_module, "video_timeline", timeline)
        sys.modules["imageio"] = imageio_module
        sys.modules["imageio.v2"] = imageio_v2
        setattr(imageio_module, "v2", imageio_v2)
        return _load_module(
            repo / "event_sae" / "events" / "extract_media.py",
            media_name,
        )
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


MEDIA = _load_extract_media()


class _FakeReader:
    def get_meta_data(self):
        return {"fps": 20.0}

    def count_frames(self):
        return 13

    def get_data(self, frame_index):
        return np.full((2, 2, 3), frame_index, dtype=np.uint8)

    def close(self):
        return None


def test_manifest_timeline_maps_record_offsets_before_video_frames(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "trajectory_records.jsonl"
    records = []
    for step in range(5):
        records.append(
            {
                "episode_num": 0,
                "task_id": 8,
                "task_episode_idx": 0,
                "task_description": "Open the left drawer.",
                "prompt_task_description": "Open the left drawer.",
                "step_in_episode": step,
                "eef_pos": [float(step), 0.0, 0.0],
                "done": step == 4,
            }
        )
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary_path = tmp_path / "waypoint_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_trajectory_records_path": str(records_path),
                "episodes": [
                    {
                        "episode_num": 0,
                        "waypoint_indices": [2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    relative_video = Path(
        "OpenDrawer/pq3_drawer_left/task8--ep0--succ1.mp4"
    )
    video_root = tmp_path / "videos"
    video_path = video_root / relative_video
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    manifest_path = tmp_path / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_root": "/remote/not/used",
                "episodes": [
                    {
                        "episode_num": 0,
                        "source_video_relative_path": relative_video.as_posix(),
                        "num_records": 5,
                        "n_action_steps": 5,
                        "steps_per_render": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    MEDIA.imageio.get_reader = lambda _: _FakeReader()
    MEDIA._save_frame = lambda frame, path: None
    MEDIA._write_clip = lambda **kwargs: None

    output = tmp_path / "media"
    report = MEDIA.extract_keyframe_media(
        waypoint_summary_path=summary_path,
        output_dir=output,
        frame_offsets=[-1, 0, 1],
        trajectory_manifest_path=manifest_path,
        video_root=video_root,
        frame_anchor="first",
        require_complete_videos=True,
    )

    sample = json.loads(
        (output / "samples.jsonl").read_text(encoding="utf-8").strip()
    )
    assert report["num_samples"] == 1
    assert sample["requested_record_indices"] == [1, 2, 3]
    assert sample["record_indices"] == [1, 2, 3]
    assert sample["frame_indices"] == [3, 5, 8]
    assert sample["video_timeline"]["expected_num_frames"] == 13


def test_manifest_video_lookup_never_falls_back_to_legacy_match(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    legacy_dir = run_dir / "videos"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "episode_00000_wrong_experiment.mp4").touch()

    result = MEDIA.find_episode_video(
        run_dir,
        0,
        episode_manifest={
            "source_video_relative_path": (
                "OpenDrawer/pq3_drawer_left/task8--ep0--succ1.mp4"
            )
        },
        video_root=tmp_path / "exact_root",
    )
    assert result is None

    with pytest.raises(ValueError, match="manifest-relative"):
        MEDIA.find_episode_video(
            run_dir,
            0,
            episode_manifest={"source_video_relative_path": "/other/run.mp4"},
            video_root=tmp_path / "exact_root",
        )
