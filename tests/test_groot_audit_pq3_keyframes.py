import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
) -> Path:
    tag = str(threshold).replace(".", "p")
    path = root / f"dp_pos_only_err{tag}" / "waypoint_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "waypoint_mode": "pos_only",
                "dp_implementation": "exact_pos_only",
                "err_threshold": threshold,
                "episodes": [
                    {
                        "episode_num": 0,
                        "num_steps": len(POSITIONS),
                        "waypoint_indices": waypoint_indices,
                        "num_waypoints": len(waypoint_indices),
                        "waypoint_positions": [
                            POSITIONS[index] for index in waypoint_indices
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
