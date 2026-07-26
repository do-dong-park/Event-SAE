import json
from pathlib import Path

import numpy as np
import pytest

import event_sae.events.build_features as build_features
from event_sae.events.build_features import (
    EQUAL_VIEW_CONCAT_FUSION_ID,
    combine_multiview_event_features,
)


def _write_v4_sample(root: Path) -> Path:
    frames = []
    for position in range(5):
        frame_path = root / "frames" / f"frame_{position}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"jpeg")
        frames.append(
            {
                "position": position,
                "path": frame_path.relative_to(root).as_posix(),
            }
        )
    sample = {
        "format": "event_sae_stage3_media_v4",
        "sample_id": "ep0000_wp000_r0002",
        "task_id": 8,
        "task_description": "Open the left drawer.",
        "prompt_task_description": "Open the left drawer.",
        "episode_num": 0,
        "task_episode_idx": 0,
        "cell_id": "pq3_drawer_left",
        "success": False,
        "waypoint_rank": 0,
        "waypoint_index": 2,
        "source_video_relative_path": "OpenDrawer/example.mp4",
        "frames": frames,
        "boundary_shift_category": "interior",
        "anchor_env_step_error": 1,
    }
    samples_path = root / "samples.jsonl"
    samples_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return samples_path


def _write_trajectory(root: Path) -> Path:
    path = root / "trajectory_records.jsonl"
    records = [
        {
            "episode_num": 0,
            "step_in_episode": step,
            "eef_pos": [float(step), 0.0, 1.0],
        }
        for step in range(5)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_normalize_versioned_media_sample(tmp_path: Path) -> None:
    trajectory_path = _write_trajectory(tmp_path)
    samples_path = _write_v4_sample(tmp_path)

    samples = build_features.normalize_media_samples(
        samples_path,
        trajectory_records_path=trajectory_path,
    )

    assert len(samples) == 1
    assert samples[0]["waypoint_step"] == 2
    assert samples[0]["clip_path"] == "OpenDrawer/example.mp4"
    assert all(Path(path).is_file() for path in samples[0]["frame_paths"])
    assert samples[0]["source_trajectory_records_path"] == str(
        trajectory_path.resolve()
    )


def test_build_event_features_from_versioned_media(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trajectory_path = _write_trajectory(tmp_path)
    samples_path = _write_v4_sample(tmp_path)

    class FakeEmbedder:
        def __init__(self, model_name_or_path, device, revision=None):
            self.model_name_or_path = model_name_or_path

        def encode(self, frame_paths):
            assert len(frame_paths) == 5
            return np.asarray([0.6, 0.8], dtype=np.float32)

    monkeypatch.setattr(build_features, "VisionEmbedder", FakeEmbedder)
    output_path = tmp_path / "event_features.jsonl"
    manifest = build_features.build_event_features(
        samples_path=samples_path,
        output_path=output_path,
        trajectory_records_path=trajectory_path,
        expected_samples=1,
        device="cpu",
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["sample_id"] == "ep0000_wp000_r0002"
    assert record["waypoint_step"] == 2
    assert record["state_vector"] == [2.0, 0.0, 1.0]
    assert record["progress_percent"] == 0.5
    assert record["success"] is False
    assert record["vision_embedding"] == pytest.approx([0.6, 0.8])
    assert manifest["num_samples"] == 1
    assert manifest["vision_dimension"] == 2
    assert manifest["passed"] is True
    assert manifest["trajectory_record_sources"] == [
        {
            "path": str(trajectory_path.resolve()),
            "sha256": manifest["trajectory_records_sha256"],
        }
    ]
    assert manifest["trajectory_records_path"] == str(trajectory_path.resolve())
    assert (tmp_path / "event_features_manifest.json").is_file()


def test_build_event_features_rejects_provider_provenance_collision(
    tmp_path: Path,
) -> None:
    trajectory_path = _write_trajectory(tmp_path)
    samples_path = _write_v4_sample(tmp_path)

    class CollidingProvider:
        def encode(self, *, sample, selected_frame_paths):
            return build_features.VisionEmbeddingResult(
                embedding=np.asarray([0.6, 0.8], dtype=np.float32),
                provenance={"vision_embedding": [1.0, 0.0]},
            )

        def finalize(self):
            raise AssertionError("collision must fail before provider finalization")

    output_path = tmp_path / "event_features.jsonl"
    with pytest.raises(
        ValueError,
        match=r"provenance collides with core fields: \['vision_embedding'\]",
    ):
        build_features.build_event_features(
            samples_path=samples_path,
            output_path=output_path,
            trajectory_records_path=trajectory_path,
            expected_samples=1,
            device="cpu",
            frame_positions=(0, 1, 2, 3, 4),
            vision_embedding_provider=CollidingProvider(),
        )

    assert not output_path.exists()
    assert not (tmp_path / "event_features_manifest.json").exists()


def test_build_event_features_rejects_provider_manifest_collision(
    tmp_path: Path,
) -> None:
    trajectory_path = _write_trajectory(tmp_path)
    samples_path = _write_v4_sample(tmp_path)

    class CollidingProvider:
        def encode(self, *, sample, selected_frame_paths):
            return build_features.VisionEmbeddingResult(
                embedding=np.asarray([0.6, 0.8], dtype=np.float32),
            )

        def finalize(self):
            return {"format": "provider-must-not-replace-artifact-format"}

    output_path = tmp_path / "event_features.jsonl"
    with pytest.raises(
        ValueError,
        match=r"manifest provenance collides with core fields: \['format'\]",
    ):
        build_features.build_event_features(
            samples_path=samples_path,
            output_path=output_path,
            trajectory_records_path=trajectory_path,
            expected_samples=1,
            device="cpu",
            frame_positions=(0, 1, 2, 3, 4),
            vision_embedding_provider=CollidingProvider(),
        )

    assert not output_path.exists()
    assert not (tmp_path / "event_features_manifest.json").exists()


def _write_frames_and_sample(root: Path) -> Path:
    frames = []
    for position in range(5):
        frame_path = root / "frames" / f"frame_{position}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"jpeg")
        frames.append(
            {
                "position": position,
                "path": frame_path.relative_to(root).as_posix(),
            }
        )
    sample = {
        "format": "event_sae_stage3_media_v4",
        "sample_id": "ep0000_wp000_r0001",
        "task_id": 1,
        "task_description": "Pick the object and place it in the cabinet.",
        "prompt_task_description": "Pick the object and place it in the cabinet.",
        "episode_num": 0,
        "task_episode_idx": 0,
        "cell_id": "cell",
        "success": False,
        "waypoint_rank": 0,
        "waypoint_index": 1,
        "source_video_relative_path": "video.mp4",
        "frames": frames,
    }
    path = root / "samples.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return path


def _write_dual_frame_trajectory(root: Path) -> Path:
    task = "Pick the object and place it in the cabinet."
    records = []
    qpos_by_episode = {
        0: ([0.5, -0.5], [0.25, -0.25]),
        1: ([0.0, 0.0], [0.0, 0.0]),
    }
    for episode_num, qpos_rows in qpos_by_episode.items():
        for step, qpos in enumerate(qpos_rows):
            records.append(
                {
                    "episode_num": episode_num,
                    "task_id": 1,
                    "task_episode_idx": episode_num,
                    "task_description": task,
                    "step_in_episode": step,
                    "eef_pos": [float(step), 0.0, 0.0],
                    "eef_pos_rel": [float(step), 0.0, 0.0],
                    "eef_pos_abs": [10.0 + step, 1.0, 2.0],
                    "gripper_qpos": list(qpos),
                    "done": step == len(qpos_rows) - 1,
                }
            )
    path = root / "trajectory_records.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_build_event_features_uses_abs_qpos_state_shared_with_waypoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    samples_path = _write_frames_and_sample(tmp_path)
    trajectory_path = _write_dual_frame_trajectory(tmp_path)

    class FakeEmbedder:
        def __init__(self, model_name_or_path, device, revision=None):
            pass

        def encode(self, frame_paths):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(build_features, "VisionEmbedder", FakeEmbedder)
    output_path = tmp_path / "event_features.jsonl"
    manifest = build_features.build_event_features(
        samples_path=samples_path,
        output_path=output_path,
        trajectory_records_path=trajectory_path,
        state_position_frame="abs",
        gripper_state_mode="qpos_aperture_delta",
        expected_samples=1,
        device="cpu",
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["state_feature_names"] == [
        "eef_pos_abs_x",
        "eef_pos_abs_y",
        "eef_pos_abs_z",
        "gripper_aperture_normalized",
        "gripper_aperture_delta",
    ]
    assert record["state_vector"] == pytest.approx(
        [11.0, 1.0, 2.0, 0.5, -0.5]
    )
    assert record["gripper_state"] == pytest.approx(
        {
            "aperture": 0.5,
            "normalized_aperture": 0.5,
            "aperture_delta": -0.5,
            "normalization_min": 0.0,
            "normalization_max": 1.0,
        }
    )
    assert manifest["state_position_frame"] == "abs"
    assert manifest["gripper_state_mode"] == "qpos_aperture_delta"


def test_abs_qpos_state_fails_closed_when_abs_position_is_missing(
    tmp_path: Path,
) -> None:
    samples_path = _write_frames_and_sample(tmp_path)
    trajectory_path = _write_dual_frame_trajectory(tmp_path)
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1].pop("eef_pos_abs")
    trajectory_path.write_text(
        "".join(json.dumps(record) + "\n" for record in rows),
        encoding="utf-8",
    )
    samples = build_features.normalize_media_samples(
        samples_path,
        trajectory_records_path=trajectory_path,
    )

    with pytest.raises(KeyError, match="eef_pos_abs"):
        build_features.build_episode_state_index(
            samples,
            state_position_frame="abs",
            gripper_state_mode="qpos_aperture_delta",
        )


def _multiview_record(
    sample_id: str,
    embedding: list[float],
    frame_prefix: str,
) -> dict:
    sample_index = int(sample_id.rsplit("_", maxsplit=1)[-1])
    return {
        "sample_id": sample_id,
        "source_format": "event_sae_stage3_media_v4",
        "task_id": 8,
        "task_description": "Open the left drawer.",
        "prompt_task_description": "Open the left drawer.",
        "episode_num": sample_index,
        "task_episode_idx": sample_index,
        "cell_id": "pq3_drawer_left",
        "success": False,
        "anchor_source": "gripper_close",
        "waypoint_rank": 0,
        "waypoint_step": 2,
        "clip_path": "OpenDrawer/example.mp4",
        "frame_paths": [f"{frame_prefix}_{index}.jpg" for index in range(5)],
        "selected_frame_paths": [
            f"{frame_prefix}_{index}.jpg" for index in range(5)
        ],
        "source_trajectory_records_path": "/tmp/trajectory_records.jsonl",
        "vision_model_name_or_path": "google/siglip-base-patch16-224",
        "vision_model_revision": "test-revision",
        "vision_frame_positions": [0, 1, 2, 3, 4],
        "vision_embedding": embedding,
        "state_vector": [0.1, 0.2, 0.3],
        "state_feature_names": ["eef_pos_x", "eef_pos_y", "eef_pos_z"],
        "progress_percent": 0.5,
        "num_steps": 5,
        "boundary_shift_category": "interior",
        "anchor_env_step_error": 0,
    }


def _write_features(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_combine_multiview_features_normalizes_each_view_equally(
    tmp_path: Path,
) -> None:
    paths = {
        view: tmp_path / f"{view}.jsonl"
        for view in ("left", "right", "wrist")
    }
    embeddings = {
        "left": [3.0, 4.0],
        "right": [0.0, 2.0],
        "wrist": [1.0, 0.0],
    }
    for view, path in paths.items():
        _write_features(
            path,
            [
                _multiview_record("sample_0", embeddings[view], view),
                _multiview_record("sample_1", embeddings[view], view),
            ],
        )

    output_path = tmp_path / "multiview.jsonl"
    manifest = combine_multiview_event_features(
        left_features_path=paths["left"],
        right_features_path=paths["right"],
        wrist_features_path=paths["wrist"],
        output_path=output_path,
        expected_samples=2,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    expected = np.asarray([0.6, 0.8, 0.0, 1.0, 1.0, 0.0]) / np.sqrt(3.0)
    assert np.asarray(rows[0]["vision_embedding"]) == pytest.approx(expected)
    assert rows[0]["vision_fusion"] == EQUAL_VIEW_CONCAT_FUSION_ID
    assert rows[0]["vision_view_order"] == ["left", "right", "wrist"]
    assert rows[0]["anchor_source"] == "gripper_close"
    assert rows[0]["view_frame_paths"]["right"][0] == "right_0.jpg"
    assert manifest["vision_dimension"] == 6
    assert manifest["num_samples"] == 2
    assert manifest["passed"] is True
    assert (tmp_path / "multiview_manifest.json").is_file()


def test_combine_multiview_features_rejects_alignment_mismatch(
    tmp_path: Path,
) -> None:
    paths = {
        view: tmp_path / f"{view}.jsonl"
        for view in ("left", "right", "wrist")
    }
    for view, path in paths.items():
        record = _multiview_record("sample_0", [1.0, 0.0], view)
        if view == "right":
            record["waypoint_step"] = 3
        _write_features(path, [record])

    with pytest.raises(ValueError, match="waypoint_step"):
        combine_multiview_event_features(
            left_features_path=paths["left"],
            right_features_path=paths["right"],
            wrist_features_path=paths["wrist"],
            output_path=tmp_path / "multiview.jsonl",
            expected_samples=1,
        )


def test_combine_multiview_features_rejects_sample_coverage_mismatch(
    tmp_path: Path,
) -> None:
    paths = {
        view: tmp_path / f"{view}.jsonl"
        for view in ("left", "right", "wrist")
    }
    _write_features(
        paths["left"],
        [_multiview_record("sample_0", [1.0, 0.0], "left")],
    )
    _write_features(
        paths["right"],
        [_multiview_record("sample_1", [1.0, 0.0], "right")],
    )
    _write_features(
        paths["wrist"],
        [_multiview_record("sample_0", [1.0, 0.0], "wrist")],
    )

    with pytest.raises(ValueError, match="sample coverage differs"):
        combine_multiview_event_features(
            left_features_path=paths["left"],
            right_features_path=paths["right"],
            wrist_features_path=paths["wrist"],
            output_path=tmp_path / "multiview.jsonl",
        )


def _write_reused_vision_inputs(root: Path) -> tuple[Path, Path, Path]:
    frames = []
    for waypoint in (1, 3):
        paths = []
        for position in range(5):
            path = root / "frames" / f"wp{waypoint}_frame{position}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{waypoint}-{position}".encode())
            paths.append(str(path))
        frames.append(paths)
    task = "Pick and place."
    samples = [
        {
            "sample_id": "ep0000_wp000_r0001",
            "task_id": 1,
            "task_description": task,
            "prompt_task_description": task,
            "episode_num": 0,
            "task_episode_idx": 0,
            "success": True,
            "waypoint_rank": 0,
            "waypoint_step": 1,
            "anchor_source": "position",
            "frame_paths": frames[0],
        },
        {
            "sample_id": "ep0000_wp001_r0003",
            "task_id": 1,
            "task_description": task,
            "prompt_task_description": task,
            "episode_num": 0,
            "task_episode_idx": 0,
            "success": True,
            "waypoint_rank": 1,
            "waypoint_step": 3,
            "anchor_source": "gripper_close",
            "frame_paths": frames[1],
        },
    ]
    samples_path = root / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row) + "\n" for row in samples),
        encoding="utf-8",
    )
    trajectory_path = root / "trajectory_records.jsonl"
    trajectory_rows = []
    for step, aperture in enumerate((1.0, 0.8, 0.4, 0.0)):
        trajectory_rows.append(
            {
                "episode_num": 0,
                "task_id": 1,
                "task_episode_idx": 0,
                "task_description": task,
                "step_in_episode": step,
                "eef_pos_abs": [10.0 + step, 1.0, 2.0],
                "gripper_qpos": [aperture / 2.0, -aperture / 2.0],
                "done": step == 3,
            }
        )
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory_rows),
        encoding="utf-8",
    )
    reuse_path = root / "abs_features.jsonl"
    reuse_path.write_text(
        json.dumps(
            {
                "sample_id": "old_abs_position",
                "episode_num": 0,
                "waypoint_step": 1,
                "vision_model_name_or_path": "google/siglip-base-patch16-224",
                "vision_model_revision": None,
                "vision_frame_positions": [0, 1, 2, 3, 4],
                "selected_frame_paths": frames[0],
                "vision_embedding": [0.6, 0.8],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return samples_path, trajectory_path, reuse_path


def _build_reused_vision_features(
    *,
    samples_path: Path,
    reusable_vision_features_path: Path,
    trajectory_records_path: Path,
    output_path: Path,
    expected_samples: int,
) -> dict:
    model = "google/siglip-base-patch16-224"
    frame_positions = [0, 1, 2, 3, 4]
    provider = build_features.ExactReuseVisionEmbeddingProvider(
        reusable_features_path=reusable_vision_features_path,
        vision_model_name_or_path=model,
        vision_model_revision=None,
        frame_positions=frame_positions,
        device="cpu",
        embedder_factory=build_features.VisionEmbedder,
    )
    return build_features.build_event_features(
        samples_path=samples_path,
        output_path=output_path,
        vision_model_name_or_path=model,
        device="cpu",
        frame_positions=frame_positions,
        trajectory_records_path=trajectory_records_path,
        expected_samples=expected_samples,
        state_position_frame="abs",
        gripper_state_mode="qpos_aperture_delta",
        vision_embedding_provider=provider,
        artifact_spec=build_features.EventFeatureArtifactSpec(
            manifest_format="event_sae_v9_event_features_v1",
            record_source_format="event_sae_v9_virtual_media_v1",
            include_trajectory_record_sources=False,
            sort_manifest_keys=True,
            manifest_trailing_newline=True,
        ),
    )


def test_reused_vision_features_encode_position_and_gripper_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    samples, trajectory, reuse = _write_reused_vision_inputs(tmp_path)

    class CountingEmbedder:
        calls = 0

        def __init__(self, model_name_or_path, device, revision=None):
            pass

        def encode(self, frame_paths):
            self.__class__.calls += 1
            return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(build_features, "VisionEmbedder", CountingEmbedder)
    output = tmp_path / "reused_vision_features.jsonl"
    manifest = _build_reused_vision_features(
        samples_path=samples,
        reusable_vision_features_path=reuse,
        trajectory_records_path=trajectory,
        output_path=output,
        expected_samples=2,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["vision_embedding_source"] for row in rows] == [
        "reused",
        "computed",
    ]
    assert rows[0]["vision_embedding"] == pytest.approx([0.6, 0.8])
    assert rows[1]["vision_embedding"] == pytest.approx([1.0, 0.0])
    assert rows[0]["source_format"] == "event_sae_v9_virtual_media_v1"
    assert rows[0]["reused_vision_sample_id"] == "old_abs_position"
    assert rows[1]["reused_vision_sample_id"] is None
    assert rows[0]["state_feature_names"] == [
        "eef_pos_abs_x",
        "eef_pos_abs_y",
        "eef_pos_abs_z",
        "gripper_aperture_normalized",
        "gripper_aperture_delta",
    ]
    assert rows[0]["state_vector"] == pytest.approx(
        [11.0, 1.0, 2.0, 0.8, -0.2]
    )
    assert CountingEmbedder.calls == 1
    assert manifest["num_reused_vision_embeddings"] == 1
    assert manifest["num_computed_vision_embeddings"] == 1
    assert manifest["format"] == "event_sae_v9_event_features_v1"
    assert manifest["reusable_vision_features_path"] == str(reuse.resolve())
    assert "trajectory_record_sources" not in manifest
    manifest_path = tmp_path / "reused_vision_features_manifest.json"
    assert manifest_path.read_text(encoding="utf-8") == (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_reused_vision_features_fail_when_reuse_is_not_exhaustive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    samples, trajectory, reuse = _write_reused_vision_inputs(tmp_path)
    reusable_record = json.loads(reuse.read_text(encoding="utf-8"))
    reusable_record["waypoint_step"] = 2
    reuse.write_text(json.dumps(reusable_record) + "\n", encoding="utf-8")

    class FakeEmbedder:
        def __init__(self, model_name_or_path, device, revision=None):
            pass

        def encode(self, frame_paths):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(build_features, "VisionEmbedder", FakeEmbedder)
    output = tmp_path / "reused_vision_features.jsonl"
    with pytest.raises(
        ValueError,
        match=r"exactly match current samples: used=0, available=1",
    ):
        _build_reused_vision_features(
            samples_path=samples,
            reusable_vision_features_path=reuse,
            trajectory_records_path=trajectory,
            output_path=output,
            expected_samples=2,
        )

    assert not output.exists()
    assert not (
        tmp_path / "reused_vision_features_manifest.json"
    ).exists()


def test_reused_vision_features_fail_on_reusable_frame_mismatch(
    tmp_path: Path,
) -> None:
    samples, trajectory, reuse = _write_reused_vision_inputs(tmp_path)
    reusable_record = json.loads(reuse.read_text(encoding="utf-8"))
    reusable_record["selected_frame_paths"][0] = str(
        tmp_path / "different_frame.jpg"
    )
    reuse.write_text(json.dumps(reusable_record) + "\n", encoding="utf-8")

    output = tmp_path / "reused_vision_features.jsonl"
    with pytest.raises(ValueError, match="frames do not exactly match"):
        _build_reused_vision_features(
            samples_path=samples,
            reusable_vision_features_path=reuse,
            trajectory_records_path=trajectory,
            output_path=output,
            expected_samples=2,
        )

    assert not output.exists()
