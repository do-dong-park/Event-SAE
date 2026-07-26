import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from event_sae import sha256_file
from event_sae.events.extract_media import (
    assemble_composite_anchor_media_view,
    build_gripper_supplement_summary,
)
from event_sae.events.multiview_triptychs import (
    build_multiview_annotation_triptychs,
    make_triptych_from_views,
)
from event_sae.events.review import render_contact_sheets


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_v4_media_bundle(
    root: Path,
    *,
    episode: int,
    waypoint: int,
    sample_id: str,
) -> Path:
    frames = []
    for position in range(5):
        path = root / "frames" / sample_id / f"frame_{position}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{sample_id}-{position}".encode())
        frames.append(
            {
                "position": position,
                "path": path.relative_to(root).as_posix(),
            }
        )
    return _write_jsonl(
        root / "samples.jsonl",
        [
            {
                "format": "event_sae_stage3_media_v4",
                "sample_id": sample_id,
                "episode_num": episode,
                "waypoint_index": waypoint,
                "cell_id": "cell",
                "source_video_relative_path": "cell/video.mp4",
                "frames": frames,
            }
        ],
    )


def _write_composite_waypoint_summary(path: Path) -> Path:
    episode = {
        "episode_num": 0,
        "task_id": 1,
        "task_episode_idx": 0,
        "task_description": "Pick and place.",
        "prompt_task_description": "Pick and place.",
        "success": True,
        "num_steps": 5,
        "num_waypoints": 2,
        "position_waypoint_indices": [1],
        "gripper_close_indices": [3],
        "waypoint_indices": [1, 3],
        "waypoint_anchors": [
            {"waypoint_index": 1, "anchor_source": "position"},
            {"waypoint_index": 3, "anchor_source": "gripper_close"},
        ],
        "waypoint_positions": [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        "waypoint_gripper_state": [
            {"waypoint_index": 1, "normalized_aperture": 1.0},
            {"waypoint_index": 3, "normalized_aperture": 0.0},
        ],
    }
    path.write_text(
        json.dumps(
            {
                "format": "event_sae_waypoint_summary_v3",
                "episodes": [episode],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_view_samples(
    root: Path,
    *,
    view: str,
    waypoint_step: int = 7,
) -> Path:
    frame_paths = []
    for position in range(2):
        path = (
            root
            / view
            / f"frame_{position:02d}_v{position + 3:04d}_s0001.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (256, 256),
            color=(position * 20, 10, 30),
        ).save(path)
        frame_paths.append(str(path))
    return _write_jsonl(
        root / f"{view}.jsonl",
        [
            {
                "sample_id": "sample-1",
                "task_id": 1,
                "task_description": "Open the left drawer.",
                "prompt_task_description": "Open the left drawer.",
                "episode_num": 2,
                "task_episode_idx": 0,
                "cell_id": "cell",
                "success": True,
                "waypoint_rank": 0,
                "waypoint_index": waypoint_step,
                "waypoint_step": waypoint_step,
                "anchor_source": "gripper_close",
                "clip_path": "cell/video.mp4",
                "source_trajectory_records_path": "/tmp/records.jsonl",
                "source_video_relative_path": "cell/video.mp4",
                "source_video_sha256": "abc",
                "video_frame_indices": [3, 4],
                "boundary_shift_category": "exact",
                "anchor_env_step_error": 0,
                "frame_paths": frame_paths,
                "view": view,
            }
        ],
    )


def _write_triptych_clusters(
    path: Path,
    left_samples_path: Path,
) -> Path:
    left = _read_jsonl(left_samples_path)[0]
    return _write_jsonl(
        path,
        [
            {
                "cluster_id": "drawer_cluster_00",
                "task_description": "Open the left drawer.",
                "episode_coverage": 0.5,
                "representative_sample_ids": ["sample-1"],
                "representative_frame_paths": [left["frame_paths"]],
            }
        ],
    )


def _write_multiview_inputs(
    root: Path,
    *,
    wrist_waypoint_step: int = 7,
) -> tuple[Path, dict[str, Path]]:
    samples = {
        view: _write_view_samples(
            root,
            view=view,
            waypoint_step=(
                wrist_waypoint_step if view == "wrist" else 7
            ),
        )
        for view in ("left", "right", "wrist")
    }
    clusters = _write_triptych_clusters(
        root / "clusters.jsonl",
        samples["left"],
    )
    return clusters, samples


def test_render_contact_sheets_and_refuse_overwrite(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), "red").save(frame_path)
    clusters_path = _write_jsonl(
        tmp_path / "clusters.jsonl",
        [
            {
                "cluster_id": "a_long_task_name_cluster_03",
                "task_description": "A long task name.",
                "num_members": 18,
                "episode_coverage": 0.6,
                "cluster_mean_progress_percent": 0.4,
                "meets_min_coverage": True,
                "representative_frame_paths": [[str(frame_path)] * 5],
            }
        ],
    )
    output_dir = tmp_path / "contact_sheets"

    outputs = render_contact_sheets(
        clusters_path,
        output_dir,
        tile_size=16,
    )

    assert len(outputs) == 1
    assert outputs[0].is_file()
    manifest = json.loads(
        (output_dir / "contact_sheet_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["num_clusters"] == 1

    with pytest.raises(
        FileExistsError,
        match="non-empty contact-sheet directory",
    ):
        render_contact_sheets(clusters_path, output_dir, tile_size=16)


def test_make_triptych_from_views_preserves_crop_order() -> None:
    left = np.full((256, 256, 3), (210, 10, 10), dtype=np.uint8)
    right = np.full((256, 256, 3), (10, 210, 10), dtype=np.uint8)
    wrist = np.full((256, 256, 3), (10, 10, 210), dtype=np.uint8)

    triptych = make_triptych_from_views(
        Image.fromarray(left),
        Image.fromarray(right),
        Image.fromarray(wrist),
    )

    assert triptych.size == (768, 280)
    assert triptych.getpixel((100, 40)) == (210, 10, 10)
    assert triptych.getpixel((356, 40)) == (10, 210, 10)
    assert triptych.getpixel((612, 40)) == (10, 10, 210)


def test_make_triptych_from_views_rejects_wrong_crop_size() -> None:
    good = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    bad = Image.fromarray(np.zeros((255, 256, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="WRIST crop size"):
        make_triptych_from_views(good, good, bad)


def test_build_gripper_supplement_summary_filters_aligned_fields(
    tmp_path: Path,
) -> None:
    source = _write_composite_waypoint_summary(tmp_path / "waypoints.json")
    output = tmp_path / "supplement.json"

    result = build_gripper_supplement_summary(
        waypoint_summary_path=source,
        output_path=output,
        expected_waypoints=1,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    assert result["num_waypoints"] == 1
    assert episode["waypoint_indices"] == [3]
    assert episode["waypoint_anchors"] == [
        {"waypoint_index": 3, "anchor_source": "gripper_close"}
    ]
    assert episode["waypoint_positions"] == [[3.0, 0.0, 0.0]]
    assert episode["waypoint_gripper_state"] == [
        {"waypoint_index": 3, "normalized_aperture": 0.0}
    ]


def test_composite_media_reuses_position_and_supplements_gripper(
    tmp_path: Path,
) -> None:
    summary = _write_composite_waypoint_summary(tmp_path / "waypoints.json")
    reusable = _write_v4_media_bundle(
        tmp_path / "reusable",
        episode=0,
        waypoint=1,
        sample_id="old_position",
    )
    supplement = _write_v4_media_bundle(
        tmp_path / "supplement",
        episode=0,
        waypoint=3,
        sample_id="new_gripper",
    )
    trajectory = tmp_path / "trajectory_records.jsonl"
    trajectory.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "virtual" / "samples.jsonl"

    manifest = assemble_composite_anchor_media_view(
        waypoint_summary_path=summary,
        reusable_samples_path=reusable,
        supplement_samples_path=supplement,
        trajectory_records_path=trajectory,
        output_path=output,
        view="left",
        expected_samples=2,
    )

    rows = _read_jsonl(output)
    assert [row["sample_id"] for row in rows] == [
        "ep0000_wp000_r0001",
        "ep0000_wp001_r0003",
    ]
    assert [row["anchor_source"] for row in rows] == [
        "position",
        "gripper_close",
    ]
    assert [row["source_media_kind"] for row in rows] == [
        "abs_position_reuse",
        "gripper_close_supplement",
    ]
    assert all(
        Path(path).is_file()
        for row in rows
        for path in row["frame_paths"]
    )
    assert manifest["source_counts"] == {
        "abs_position_reuse": 1,
        "gripper_close_supplement": 1,
    }
    assert manifest["copies_frame_files"] is False


def test_multiview_triptychs_exact_join_virtual_views(
    tmp_path: Path,
) -> None:
    clusters, samples = _write_multiview_inputs(tmp_path)
    output = tmp_path / "output"

    manifest = build_multiview_annotation_triptychs(
        clusters_path=clusters,
        left_samples_path=samples["left"],
        right_samples_path=samples["right"],
        wrist_samples_path=samples["wrist"],
        output_dir=output,
    )

    rows = _read_jsonl(output / "clusters_multiview.jsonl")
    assert manifest["passed"] is True
    assert manifest["num_clusters"] == 1
    assert manifest["num_frames"] == 2
    assert manifest["output_clusters_sha256"] == sha256_file(
        output / "clusters_multiview.jsonl"
    )
    assert json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    ) == manifest
    assert rows[0]["annotation_media_layout"] == (
        "synchronized_triptych_left_right_wrist_v1"
    )
    source_paths = rows[0][
        "representative_source_view_frame_paths"
    ][0][0]
    assert list(source_paths) == ["left", "right", "wrist"]
    assert len(
        rows[0]["representative_source_view_frame_paths"]
    ) == 1
    assert all(Path(path).is_file() for path in source_paths.values())
    assert all(
        Path(path).is_file()
        for path in rows[0]["representative_frame_paths"][0]
    )
    with Image.open(
        rows[0]["representative_frame_paths"][0][0]
    ) as image:
        assert image.size == (768, 280)


def test_multiview_triptychs_reject_cross_view_misalignment(
    tmp_path: Path,
) -> None:
    clusters, samples = _write_multiview_inputs(
        tmp_path,
        wrist_waypoint_step=8,
    )

    with pytest.raises(ValueError, match="waypoint_index"):
        build_multiview_annotation_triptychs(
            clusters_path=clusters,
            left_samples_path=samples["left"],
            right_samples_path=samples["right"],
            wrist_samples_path=samples["wrist"],
            output_dir=tmp_path / "output",
        )


def test_multiview_triptychs_refuse_to_overwrite_outputs(
    tmp_path: Path,
) -> None:
    clusters, samples = _write_multiview_inputs(tmp_path)
    kwargs = {
        "clusters_path": clusters,
        "left_samples_path": samples["left"],
        "right_samples_path": samples["right"],
        "wrist_samples_path": samples["wrist"],
        "output_dir": tmp_path / "output",
    }

    build_multiview_annotation_triptychs(**kwargs)
    with pytest.raises(FileExistsError, match="not empty"):
        build_multiview_annotation_triptychs(**kwargs)
