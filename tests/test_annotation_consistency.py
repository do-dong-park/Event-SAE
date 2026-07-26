from __future__ import annotations

import hashlib
import json
from pathlib import Path

from event_sae.groot.annotation_consistency import (
    CLAIM_SCOPE,
    analyze_annotation_consistency,
    recompute_strict_consensus,
)


MODEL = "gemini-3.1-pro-preview"
DRAWER_TASK = "Open the left drawer."
PICK_TASK = "Pick the bread and place it in the cabinet."


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _annotation(
    representative_index: int,
    phase: str | None,
    *,
    attempts: int = 1,
    api_error: str | None = None,
    parse_error: str | None = None,
    visibility: str | None = "clear",
    model_version: str | None = MODEL,
) -> dict:
    return {
        "representative_index": representative_index,
        "phrase": phase,
        "phase": phase,
        "visibility": visibility,
        "request_attempts": attempts,
        "api_error": api_error,
        "parse_error": parse_error,
        "model_version": model_version,
    }


def _consensus(annotations: list[dict]) -> dict:
    recomputed = recompute_strict_consensus(annotations)
    return {
        "status": recomputed["status"],
        "phase": recomputed["phase"],
        "phase_counts": recomputed["phase_counts"],
        "num_representatives_evaluated": (
            recomputed["representatives_evaluated"]
        ),
        "num_usable_votes": recomputed["usable_votes"],
        "dominant_votes": recomputed["dominant_votes"],
    }


def _row(
    cluster_id: str,
    task_description: str,
    annotations: list[dict],
    *,
    representative_set_sha256: str = "same-representatives",
) -> dict:
    consensus = _consensus(annotations)
    return {
        "cluster_id": cluster_id,
        "task_description": task_description,
        "model": MODEL,
        "representative_set_sha256": representative_set_sha256,
        "status": consensus["status"],
        "phase": consensus["phase"],
        "consensus": consensus,
        "representative_annotations": annotations,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    conditions: list[dict],
) -> Path:
    contract = {
        "format": "event_sae_e0_e4_clean_annotation_v12r3",
        "model": MODEL,
        "prompt_id": "prompt-v12r3",
        "response_schema_id": "schema-v12r3",
        "expected_total_cluster_rows": sum(
            int(condition["selected_cluster_rows"])
            for condition in conditions
        ),
        "conditions": conditions,
    }
    manifest = {
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "frozen_at_utc": "2026-07-24T00:00:00+00:00",
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_partial_run_reports_consensus_pairing_and_terminal_quota(
    tmp_path: Path,
) -> None:
    indices = [1, 4, 5, 2, 3]
    e0_shared = [
        _annotation(index, phase)
        for index, phase in zip(
            indices,
            [
                "reach-to-handle",
                "reach-to-handle",
                "reach-to-handle",
                "grasp-handle",
                "grasp-handle",
            ],
            strict=True,
        )
    ]
    e1_shared = [
        _annotation(index, phase)
        for index, phase in zip(
            indices,
            [
                "reach-to-handle",
                "grasp-handle",
                "reach-to-handle",
                "grasp-handle",
                "grasp-handle",
            ],
            strict=True,
        )
    ]
    e0_accepted = [
        _annotation(1, "pull"),
        _annotation(4, "pull"),
        _annotation(5, "pull"),
    ]

    e0_output = tmp_path / "e0" / "adaptive_annotations.jsonl"
    e1_output = tmp_path / "e1" / "adaptive_annotations.jsonl"
    _write_jsonl(
        e0_output,
        [
            _row("shared", DRAWER_TASK, e0_shared),
            _row("accepted", DRAWER_TASK, e0_accepted),
        ],
    )
    _write_jsonl(
        e1_output,
        [_row("shared", DRAWER_TASK, e1_shared)],
    )
    inflight = {
        "cluster_id": "pending",
        "task_description": DRAWER_TASK,
        "representative_annotations": [
            _annotation(1, "grasp-handle", attempts=2),
            _annotation(
                4,
                None,
                attempts=3,
                api_error="429 RESOURCE_EXHAUSTED quota",
                visibility=None,
                model_version=None,
            ),
        ],
    }
    e1_output.with_name("adaptive_annotations.inflight.json").write_text(
        json.dumps(inflight),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "run_manifest.json",
        [
            {
                "experiment_id": "E0",
                "condition_id": "e0",
                "selected_cluster_rows": 2,
                "output_path": str(e0_output),
            },
            {
                "experiment_id": "E1",
                "condition_id": "e1",
                "selected_cluster_rows": 2,
                "output_path": str(e1_output),
            },
        ],
    )

    report = analyze_annotation_consistency(manifest)

    assert report["claim_scope"]["statement"] == CLAIM_SCOPE
    assert report["claim_scope"]["accuracy_evaluated"] is False
    assert report["progress"] == {
        "expected_final_clusters": 4,
        "finalized_clusters": 3,
        "remaining_final_clusters": 1,
        "active_inflight_clusters": 1,
        "completion_fraction": 0.75,
        "complete_conditions": 1,
        "partial_conditions": 1,
        "not_started_conditions": 0,
        "overcomplete_conditions": 0,
    }
    strict = report["strict_accepted_consensus"]
    assert strict["accepted_clusters"] == 1
    assert strict["recomputed_status_counts"] == {
        "consensus-3-of-3": 1,
        "mixed": 2,
    }
    dominant = report["dominant_expanded"]
    assert dominant["expanded_clusters"] == 2
    assert dominant["five_usable_vote_clusters"] == 2
    assert dominant["dominant_at_least_3_clusters"] == 2
    assert dominant["exactly_3_of_5_strictly_rejected_clusters"] == 2
    all_spans = report["main_chain_span"][
        "all_supported_task_families"
    ]
    assert all_spans["span_le_1_clusters"] == 3
    assert all_spans["span_le_2_clusters"] == 3
    assert all_spans["adjacent_disagreement_clusters"] == 2

    pairing = report["e0_e1_same_cluster_representative_agreement"]
    assert pairing["common_finalized_clusters"] == 1
    assert pairing["paired_representative_indices"] == 5
    assert pairing["comparable_phase_pairs"] == 5
    assert pairing["exact_phase_agreements"] == 4
    assert pairing["exact_phase_agreement_rate"] == 0.8
    assert pairing["phase_transitions"] == [
        {
            "e0_phase": "grasp-handle",
            "e1_phase": "grasp-handle",
            "count": 2,
        },
        {
            "e0_phase": "reach-to-handle",
            "e1_phase": "reach-to-handle",
            "count": 2,
        },
        {
            "e0_phase": "reach-to-handle",
            "e1_phase": "grasp-handle",
            "count": 1,
        },
    ]

    attempts = report["attempts_errors_model_versions"]
    assert attempts["representative_records"] == 15
    assert attempts["request_attempts_total"] == 18
    assert attempts["retried_representative_records"] == 2
    assert attempts["retry_excess_attempts"] == 3
    assert attempts["terminal_api_error_records"] == 1
    assert attempts["terminal_api_error_categories"] == {"quota": 1}
    assert attempts["model_version_counts"] == {
        "<missing>": 1,
        MODEL: 14,
    }
    assert report["integrity"]["passed"] is True


def test_main_chain_span_reports_thresholds_and_excludes_detours(
    tmp_path: Path,
) -> None:
    drawer_annotations = [
        _annotation(1, "reach-to-handle"),
        _annotation(4, "pull"),
        _annotation(5, "reach-to-handle"),
        _annotation(2, "pull"),
        _annotation(3, "pull"),
    ]
    drawer_detour = [
        _annotation(1, "grasp-handle"),
        _annotation(4, "disengage"),
        _annotation(5, "grasp-handle"),
        _annotation(2, "disengage"),
        _annotation(3, "grasp-handle"),
    ]
    pick_within = [
        _annotation(1, "reach-to-object"),
        _annotation(4, "transport"),
        _annotation(5, "transport"),
        _annotation(2, "reach-to-object"),
        _annotation(3, "transport"),
    ]
    pick_outside = [
        _annotation(1, "reach-to-object"),
        _annotation(4, "place"),
        _annotation(5, "place"),
        _annotation(2, "reach-to-object"),
        _annotation(3, "place"),
    ]
    pick_off_main = [
        _annotation(1, "wrong-grasp"),
        _annotation(4, "grasp"),
        _annotation(5, "grasp"),
        _annotation(2, "wrong-grasp"),
        _annotation(3, "grasp"),
    ]
    output = tmp_path / "e2" / "adaptive_annotations.jsonl"
    _write_jsonl(
        output,
        [
            _row("drawer-wide", DRAWER_TASK, drawer_annotations),
            _row("drawer-detour", DRAWER_TASK, drawer_detour),
            _row("pick-within", PICK_TASK, pick_within),
            _row("pick-outside", PICK_TASK, pick_outside),
            _row("pick-off-main", PICK_TASK, pick_off_main),
        ],
    )
    manifest = _write_manifest(
        tmp_path / "run_manifest.json",
        [
            {
                "experiment_id": "E2",
                "condition_id": "e2",
                "selected_cluster_rows": 5,
                "output_path": str(output),
            }
        ],
    )

    report = analyze_annotation_consistency(manifest)
    spans = report["main_chain_span"]

    assert spans["drawer"]["clusters"] == 2
    assert spans["drawer"]["evaluable_clusters"] == 1
    assert spans["drawer"]["off_main_phase_clusters"] == 1
    assert spans["drawer"]["span_le_1_clusters"] == 0
    assert spans["drawer"]["span_le_2_clusters"] == 1
    assert spans["drawer"]["span_counts"] == {"2": 1}
    assert spans["pick_place"]["clusters"] == 3
    assert spans["pick_place"]["evaluable_clusters"] == 2
    assert spans["pick_place"]["span_le_1_clusters"] == 0
    assert spans["pick_place"]["span_le_2_clusters"] == 1
    assert spans["pick_place"]["off_main_phase_clusters"] == 1
    assert spans["pick_place"]["span_counts"] == {"2": 1, "3": 1}
    combined = spans["all_supported_task_families"]
    assert combined["clusters"] == 5
    assert combined["evaluable_clusters"] == 3
    assert combined["span_le_1_clusters"] == 0
    assert combined["span_le_2_clusters"] == 2
    assert combined["adjacent_disagreement_clusters"] == 0


def test_stored_consensus_mismatch_and_manifest_hash_are_reported(
    tmp_path: Path,
) -> None:
    annotations = [
        _annotation(1, "grasp-handle"),
        _annotation(4, "grasp-handle"),
        _annotation(5, "reach-to-handle"),
        _annotation(2, "reach-to-handle"),
        _annotation(3, "reach-to-handle"),
    ]
    row = _row("cluster", DRAWER_TASK, annotations)
    row["status"] = "consensus-4-of-5"
    row["phase"] = "grasp-handle"
    row["consensus"]["status"] = "consensus-4-of-5"
    row["consensus"]["phase"] = "grasp-handle"
    output = tmp_path / "e0" / "adaptive_annotations.jsonl"
    _write_jsonl(output, [row])
    manifest = _write_manifest(
        tmp_path / "run_manifest.json",
        [
            {
                "experiment_id": "E0",
                "condition_id": "e0",
                "selected_cluster_rows": 1,
                "output_path": str(output),
            }
        ],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = analyze_annotation_consistency(manifest)

    assert report["manifest"]["embedded_contract_hash_matches"] is False
    assert report["strict_accepted_consensus"]["accepted_clusters"] == 0
    assert report["integrity"]["passed"] is False
    joined = "\n".join(report["integrity"]["issues"])
    assert "contract_sha256 mismatch" in joined
    assert "status mismatch" in joined
    assert "phase mismatch" in joined
