import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_audit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "groot"
        / "audit_pq3_keyframes.py"
    )
    spec = importlib.util.spec_from_file_location("groot_waypoint_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()
POSITIONS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 1.0, 0.0],
    [3.0, 0.0, 0.0],
    [4.0, 0.0, 0.0],
]
ABS_POSITIONS = [[position[0] + 10.0, *position[1:]] for position in POSITIONS]


def _write_inputs(root: Path) -> tuple[Path, Path]:
    records_path = root / "trajectory_records.jsonl"
    records = []
    for step, position in enumerate(POSITIONS):
        records.append(
            {
                "episode_num": 0,
                "task_id": 8,
                "task_episode_idx": 0,
                "task_description": "Open the left drawer.",
                "step_in_episode": step,
                "eef_pos": position,
                "eef_pos_rel": position,
                "eef_pos_abs": ABS_POSITIONS[step],
                "done": step == len(POSITIONS) - 1,
                "cell_id": "pq3_drawer_left",
            }
        )
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = root / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_num": 0,
                        "event_steps": {"near:handle": 2},
                        "grasp_steps": [],
                        "drop_steps": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return records_path, manifest_path


def _write_summary(
    root: Path,
    *,
    threshold: float,
    waypoint_indices: list[int],
    eef_position_frame: str = "rel",
) -> Path:
    tag = str(threshold).replace(".", "p")
    frame_tag = "" if eef_position_frame == "rel" else f"_{eef_position_frame}"
    path = root / f"dp_pos_only{frame_tag}_err{tag}" / "waypoint_summary.json"
    path.parent.mkdir(parents=True)
    positions = POSITIONS if eef_position_frame == "rel" else ABS_POSITIONS
    path.write_text(
        json.dumps(
            {
                "waypoint_mode": "pos_only",
                "eef_position_frame": eef_position_frame,
                "dp_implementation": "exact_pos_only",
                "err_threshold": threshold,
                "episodes": [
                    {
                        "episode_num": 0,
                        "num_steps": len(POSITIONS),
                        "waypoint_indices": waypoint_indices,
                        "num_waypoints": len(waypoint_indices),
                        "waypoint_positions": [
                            positions[index] for index in waypoint_indices
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_waypoint_audit_computes_reconstruction_and_monotonicity(
    tmp_path: Path,
) -> None:
    records_path, manifest_path = _write_inputs(tmp_path)
    _write_summary(
        tmp_path,
        threshold=0.02,
        waypoint_indices=[0, 1, 2, 3, 4],
    )
    _write_summary(
        tmp_path,
        threshold=0.1,
        waypoint_indices=[0, 4],
    )
    output = tmp_path / "waypoint_audit.json"

    report = AUDIT.audit_waypoints(
        SimpleNamespace(
            trajectory_records_path=records_path,
            trajectory_manifest=manifest_path,
            waypoint_root=tmp_path,
            waypoint_summary=None,
            event_tolerance=0,
            output=output,
        )
    )

    assert [run["err_threshold"] for run in report["runs"]] == [0.02, 0.1]
    tight, coarse = report["runs"]
    assert tight["dp_implementation"] == "exact_pos_only"
    assert tight["aggregate"]["global_max_error"] == pytest.approx(0.0)
    assert tight["aggregate"]["num_awe_threshold_violations"] == 0
    assert coarse["aggregate"]["global_max_error"] == pytest.approx(1.0)
    assert coarse["aggregate"]["global_awe_geometric_max_error"] == pytest.approx(1.0)
    assert coarse["aggregate"]["num_awe_threshold_violations"] == 1
    assert tight["aggregate"]["micro_event_recall"] == pytest.approx(1.0)
    assert coarse["aggregate"]["micro_event_recall"] == pytest.approx(0.0)
    assert report["threshold_monotonicity"]["passed"] is True
    assert report["threshold_contract"]["passed"] is False
    assert report["threshold_contract"]["num_violations"] == 1
    assert output.is_file()


def test_waypoint_audit_reports_threshold_count_violation(tmp_path: Path) -> None:
    records_path, manifest_path = _write_inputs(tmp_path)
    _write_summary(
        tmp_path,
        threshold=0.02,
        waypoint_indices=[0, 4],
    )
    _write_summary(
        tmp_path,
        threshold=0.1,
        waypoint_indices=[0, 2, 4],
    )

    report = AUDIT.audit_waypoints(
        SimpleNamespace(
            trajectory_records_path=records_path,
            trajectory_manifest=manifest_path,
            waypoint_root=tmp_path,
            waypoint_summary=None,
            event_tolerance=2,
            output=tmp_path / "audit.json",
        )
    )

    monotonicity = report["threshold_monotonicity"]
    assert monotonicity["passed"] is False
    assert monotonicity["num_violations"] == 1
    assert monotonicity["violations"][0]["waypoint_counts"] == [2, 3]


def test_waypoint_audit_uses_summary_position_frame(tmp_path: Path) -> None:
    records_path, manifest_path = _write_inputs(tmp_path)
    summary_path = _write_summary(
        tmp_path,
        threshold=0.05,
        waypoint_indices=[0, 1, 2, 3, 4],
        eef_position_frame="abs",
    )

    report = AUDIT.audit_waypoints(
        SimpleNamespace(
            trajectory_records_path=records_path,
            trajectory_manifest=manifest_path,
            waypoint_root=None,
            waypoint_summary=[summary_path],
            event_tolerance=0,
            output=tmp_path / "abs_audit.json",
        )
    )

    assert report["eef_position_frame"] == "abs"
    assert report["runs"][0]["eef_position_frame"] == "abs"
    assert report["runs"][0]["aggregate"]["global_max_error"] == pytest.approx(
        0.0
    )


def test_waypoint_audit_refuses_to_overwrite_output_before_loading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "waypoint_audit.json"
    sentinel = b"existing waypoint audit"
    output.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        AUDIT.audit_waypoints(
            SimpleNamespace(
                trajectory_records_path=tmp_path / "missing_records.jsonl",
                trajectory_manifest=None,
                waypoint_root=tmp_path / "missing_waypoints",
                waypoint_summary=None,
                event_tolerance=0,
                output=output,
            )
        )

    assert output.read_bytes() == sentinel


def test_composite_audit_separates_position_reconstruction_from_event_recall(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "trajectory_records.jsonl"
    records = [
        {
            "episode_num": 0,
            "task_id": 8,
            "task_episode_idx": 0,
            "task_description": "Open the left drawer.",
            "step_in_episode": step,
            "eef_pos": position,
            "eef_pos_rel": position,
            "done": step == len(POSITIONS) - 1,
            "cell_id": "pq3_drawer_left",
        }
        for step, position in enumerate(POSITIONS)
    ]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_num": 0,
                        "event_steps": {"near:handle": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "waypoint_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "waypoint_mode": "pos_gripper_close",
                "eef_position_frame": "rel",
                "dp_implementation": "exact_pos_only",
                "err_threshold": 1.1,
                "episodes": [
                    {
                        "episode_num": 0,
                        "num_steps": len(POSITIONS),
                        "waypoint_indices": [0, 2, 4],
                        "position_waypoint_indices": [0, 4],
                        "num_waypoints": 3,
                        "waypoint_positions": [
                            POSITIONS[index] for index in (0, 2, 4)
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = AUDIT.audit_waypoints(
        SimpleNamespace(
            trajectory_records_path=records_path,
            trajectory_manifest=manifest_path,
            waypoint_root=None,
            waypoint_summary=[summary_path],
            event_tolerance=0,
            output=tmp_path / "audit.json",
        )
    )

    aggregate = report["runs"][0]["aggregate"]
    episode = report["runs"][0]["episodes"][0]
    assert aggregate["total_waypoints"] == 3
    assert aggregate["total_position_waypoints"] == 2
    assert episode["rmse"] == pytest.approx(np.sqrt(1.0 / 5.0))
    assert episode["event_recall"] == pytest.approx(1.0)
    assert report["threshold_contract"]["passed"] is True
