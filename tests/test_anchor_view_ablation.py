import json
from pathlib import Path

from PIL import Image

from event_sae.events.io import load_jsonl
from event_sae.groot.anchor_view_ablation import (
    build_multisource_reusable_features,
    materialize_position_virtual_media,
    materialize_prompt_records_from_event_features,
    video_frame_window,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _media_row(root: Path, *, episode: int, step: int, sample_id: str) -> dict:
    frames = []
    for position in range(5):
        relative = (
            Path("frames") / sample_id / f"frame_{position:02d}_v{position:04d}.jpg"
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), (position, 0, 0)).save(path)
        frames.append({"position": position, "path": str(relative)})
    return {
        "format": "event_sae_stage3_media_v4",
        "sample_id": sample_id,
        "episode_num": episode,
        "waypoint_index": step,
        "waypoint_step": step,
        "task_id": 1,
        "task_episode_idx": episode,
        "task_description": "Open the drawer.",
        "prompt_task_description": "Open the drawer.",
        "cell_id": "cell",
        "success": False,
        "source_video_relative_path": "task/episode.mp4",
        "source_video_sha256": "video",
        "video_frame_indices": list(range(5)),
        "frames": frames,
    }


def test_materialize_position_virtual_media_and_reuse_subset(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_num": 0,
                        "task_id": 1,
                        "task_episode_idx": 0,
                        "task_description": "Open the drawer.",
                        "success": False,
                        "waypoint_indices": [2],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    media_root = tmp_path / "media"
    samples = media_root / "samples.jsonl"
    row = _media_row(media_root, episode=0, step=2, sample_id="sample")
    _write_jsonl(samples, [row])
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text("{}\n", encoding="utf-8")
    virtual = tmp_path / "virtual" / "samples.jsonl"

    report = materialize_position_virtual_media(
        waypoint_summary_path=summary,
        source_samples_path=samples,
        trajectory_records_path=trajectory,
        output_path=virtual,
        view="left",
        expected_samples=1,
    )

    virtual_row = load_jsonl(virtual)[0]
    assert virtual_row["anchor_source"] == "position"
    assert virtual_row["source_media_kind"] == "r_pos_reuse"
    assert report["passed"] is True

    primary = tmp_path / "primary.jsonl"
    _write_jsonl(
        primary,
        [
            {
                "sample_id": "old",
                "episode_num": 0,
                "waypoint_step": 2,
                "selected_frame_paths": virtual_row["frame_paths"],
                "vision_embedding": [1.0],
            }
        ],
    )
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        secondary,
        [
            {
                "sample_id": "unused",
                "episode_num": 1,
                "waypoint_step": 3,
                "selected_frame_paths": virtual_row["frame_paths"],
                "vision_embedding": [2.0],
            }
        ],
    )
    output = tmp_path / "reuse.jsonl"
    reuse_report = build_multisource_reusable_features(
        virtual_samples_path=virtual,
        primary_features_path=primary,
        secondary_features_path=secondary,
        output_path=output,
        expected_primary=1,
        expected_secondary=0,
    )
    assert reuse_report["num_rows"] == 1


def test_video_frame_window_matches_existing_bundle_contract() -> None:
    timeline = {
        "n_action_steps": 5,
        "first_video_frame_env_step": 1,
        "steps_per_render": 2,
        "expected_num_frames": 360,
    }
    actual, requested, category, center = video_frame_window(
        waypoint_index=2,
        timeline=timeline,
    )
    assert actual == [3, 4, 5, 6, 7]
    assert requested == actual
    assert category == "interior"
    assert center == 5


def test_materialize_prompt_records_requires_exact_source_mappings(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        {
            "sample_id": f"sample_{episode_num}",
            "episode_num": episode_num,
            "task_id": 7,
            "task_description": "Open the drawer.",
            "task_episode_idx": episode_num,
        }
        for episode_num in range(2)
    ]
    _write_jsonl(first, rows + [{**rows[0], "sample_id": "duplicate_event"}])
    _write_jsonl(second, list(reversed(rows)))
    output = tmp_path / "prompt_records.jsonl"

    report = materialize_prompt_records_from_event_features(
        event_features_paths=[first, second],
        output_path=output,
        expected_episodes=2,
    )

    assert [row["episode_num"] for row in load_jsonl(output)] == [0, 1]
    assert report["all_source_mappings_exact"] is True
    assert report["output_rows"] == 2
