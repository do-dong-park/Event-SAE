import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_exporter_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "groot"
        / "export_pq3_trajectories.py"
    )
    spec = importlib.util.spec_from_file_location("groot_trajectory_exporter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXPORTER = _load_exporter_module()


def _payload(*, success: bool = True) -> dict:
    states = []
    actions = []
    for step in range(2):
        states.append(
            {
                "observation.state.eef_pos_rel": np.asarray(
                    [step, step + 0.1, step + 0.2], dtype=np.float32
                ),
                "observation.state.eef_quat_rel": np.asarray(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                ),
                "observation.state.gripper_qpos": np.asarray(
                    [0.01, 0.02], dtype=np.float32
                ),
            }
        )
        actions.append(
            {
                "action.gripper_close": np.zeros((16, 1), dtype=np.float32),
            }
        )
    return {
        "states": states,
        "actions": actions,
        "task_id": 8,
        "episode_idx": 0,
        "episode_success": int(success),
        "cell_index": 8,
        "cell_id": "pq3_drawer_left",
        "robocasa_task": "OpenDrawer",
        "task_description": "Open the left drawer.",
        "canonical_instruction": "Open the left drawer.",
        "n_action_steps": 5,
        "steps_per_render": 2,
        "video_fps": 20,
        "event_steps": {"near:handle": 1},
        "grasp_steps": [],
        "drop_steps": [],
    }


def _write_source(root: Path, *, success: bool = True) -> Path:
    cell = root / "OpenDrawer" / "pq3_drawer_left"
    cell.mkdir(parents=True)
    stem = f"task8--ep0--succ{int(success)}"
    pkl_path = cell / f"{stem}.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(_payload(success=success), handle)
    (cell / f"{stem}.csv").write_text("record\n", encoding="utf-8")
    (cell / f"{stem}.mp4").touch()
    return pkl_path


def _export_args(root: Path, output: Path, *, trust_pkl: bool = True):
    return SimpleNamespace(
        input_dir=root,
        output_dir=output,
        trust_pkl=trust_pkl,
        allow_partial_inventory=True,
        progress_every=0,
    )


def test_export_preserves_pose_provenance_and_video_timing(tmp_path: Path) -> None:
    root = tmp_path / "raw_rollouts"
    _write_source(root)
    output = tmp_path / "export"

    EXPORTER.export_trajectories(_export_args(root, output))

    records = [
        json.loads(line)
        for line in (output / EXPORTER.RECORDS_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["step_in_episode"] for record in records] == [0, 1]
    assert [record["done"] for record in records] == [False, True]
    assert records[0]["eef_pos"] == pytest.approx([0.0, 0.1, 0.2])
    assert records[0]["eef_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert records[0]["source_file"] == (
        "OpenDrawer/pq3_drawer_left/task8--ep0--succ1.pkl"
    )

    manifest = json.loads(
        (output / EXPORTER.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    episode = manifest["episodes"][0]
    assert manifest["num_records"] == 2
    assert episode["num_records"] == 2
    assert episode["n_action_steps"] == 5
    assert episode["steps_per_render"] == 2
    assert episode["expected_video_frames"] == 5
    assert episode["video_present_at_export"] is True
    assert episode["source_video_relative_path"].endswith("succ1.mp4")


def test_audit_accepts_exact_video_root_and_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "raw_rollouts"
    _write_source(root)
    output = tmp_path / "export"
    EXPORTER.export_trajectories(_export_args(root, output))

    report_path = output / "audit.json"
    report = EXPORTER.audit_trajectories(
        SimpleNamespace(
            trajectory_records_path=output / EXPORTER.RECORDS_NAME,
            manifest=output / EXPORTER.MANIFEST_NAME,
            video_root=root,
            require_complete_videos=True,
            output=report_path,
        )
    )

    assert report["schema_finite_step_audit_passed"] is True
    assert report["num_episodes"] == 1
    assert report["num_records"] == 2
    assert report["num_videos_present"] == 1
    assert report["num_videos_missing"] == 0
    assert report_path.is_file()


def test_export_requires_explicit_pickle_trust(tmp_path: Path) -> None:
    root = tmp_path / "raw_rollouts"
    _write_source(root)

    with pytest.raises(ValueError, match="trust-pkl"):
        EXPORTER.export_trajectories(
            _export_args(root, tmp_path / "export", trust_pkl=False)
        )


def test_export_rejects_nonfinite_pose(tmp_path: Path) -> None:
    root = tmp_path / "raw_rollouts"
    pkl_path = _write_source(root)
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["states"][1]["observation.state.eef_pos_rel"][0] = np.nan
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="non-finite"):
        EXPORTER.export_trajectories(
            _export_args(root, tmp_path / "export")
        )
