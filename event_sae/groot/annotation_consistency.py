"""Read-only consistency diagnostics for representative annotations.

The analyzer deliberately does not import the annotation runtime.  It reads a
frozen run manifest, finalized JSONL rows, and an optional in-flight checkpoint,
then recomputes agreement statistics from stored responses.  These statistics
measure internal consistency only.  They do not measure phase accuracy because
no oracle or independently reviewed phase labels are joined.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ANNOTATION_CONSISTENCY_FORMAT = "event_sae_v12r3_annotation_consistency_v1"
CLAIM_SCOPE = "internal consistency—not accuracy"

DRAWER_MAIN_CHAIN = (
    "reach-to-handle",
    "grasp-handle",
    "pull",
    "open-done",
)
PICK_PLACE_MAIN_CHAIN = (
    "reach-to-object",
    "grasp",
    "transport",
    "place",
    "insert-settle",
    "terminal",
)
MAIN_CHAIN_CONFIG = {
    "drawer": {
        "phases": DRAWER_MAIN_CHAIN,
        "phase_ranks": {
            phase: rank
            for rank, phase in enumerate(DRAWER_MAIN_CHAIN)
        },
        "off_main_phases": (
            "push-back",
            "disengage",
            "wrong-grasp",
        ),
    },
    "pick_place": {
        "phases": PICK_PLACE_MAIN_CHAIN,
        "phase_ranks": {
            phase: rank
            for rank, phase in enumerate(PICK_PLACE_MAIN_CHAIN)
        },
        "off_main_phases": ("wrong-grasp",),
    },
}


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(
    path: Path,
    *,
    issues: list[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                f"{path}: line {line_number}: invalid JSON: {exc.msg}"
            )
            continue
        if not isinstance(value, dict):
            issues.append(
                f"{path}: line {line_number}: expected a JSON object"
            )
            continue
        rows.append(value)
    return rows


def _resolve_artifact_path(
    value: str | Path,
    *,
    manifest_path: Path,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _inflight_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.inflight.json")


def _task_family(task_description: str) -> str | None:
    lowered = task_description.lower()
    if "drawer" in lowered:
        return "drawer"
    if "pick" in lowered and "place" in lowered:
        return "pick_place"
    return None


def _usable_phase(annotation: Mapping[str, Any]) -> str | None:
    if annotation.get("api_error") is not None:
        return None
    if annotation.get("parse_error") is not None:
        return None
    if annotation.get("visibility") == "insufficient":
        return None
    phase = annotation.get("phase")
    if not isinstance(phase, str) or not phase or phase == "unresolved":
        return None
    return phase


def recompute_strict_consensus(
    representative_annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the frozen 3-of-3 / 4-of-5 acceptance rule.

    This is intentionally an independent diagnostic implementation.  Exactly
    three usable unanimous votes stop early.  An expanded five-representative
    row is accepted only with at least four votes for one phase.
    """

    annotations = list(representative_annotations)
    phases = [
        phase
        for annotation in annotations
        if (phase := _usable_phase(annotation)) is not None
    ]
    phase_counts = Counter(phases)
    ordered = sorted(
        phase_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_phase = ordered[0][0] if ordered else None
    dominant_votes = ordered[0][1] if ordered else 0

    if (
        len(annotations) == 3
        and len(phases) == 3
        and dominant_votes == 3
    ):
        status = "consensus-3-of-3"
    elif len(annotations) == 5 and dominant_votes >= 4:
        status = f"consensus-{dominant_votes}-of-5"
    elif len(phases) < 4:
        status = "insufficient"
    else:
        status = "mixed"

    accepted = status.startswith("consensus-")
    return {
        "status": status,
        "accepted": accepted,
        "phase": dominant_phase if accepted else None,
        "phase_counts": dict(sorted(phase_counts.items())),
        "dominant_phase": dominant_phase,
        "dominant_votes": dominant_votes,
        "representatives_evaluated": len(annotations),
        "usable_votes": len(phases),
    }


def _annotation_list(
    record: Mapping[str, Any],
    *,
    record_label: str,
    issues: list[str],
) -> list[dict[str, Any]]:
    raw_annotations = record.get("representative_annotations")
    if not isinstance(raw_annotations, list):
        issues.append(
            f"{record_label}: representative_annotations is not a list"
        )
        return []
    annotations: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for position, raw_annotation in enumerate(raw_annotations, start=1):
        if not isinstance(raw_annotation, dict):
            issues.append(
                f"{record_label}: representative annotation {position} "
                "is not an object"
            )
            continue
        annotation = dict(raw_annotation)
        raw_index = annotation.get("representative_index")
        try:
            representative_index = int(raw_index)
        except (TypeError, ValueError):
            issues.append(
                f"{record_label}: invalid representative_index {raw_index!r}"
            )
            continue
        if representative_index in seen_indices:
            issues.append(
                f"{record_label}: duplicate representative_index "
                f"{representative_index}"
            )
            continue
        seen_indices.add(representative_index)
        annotation["representative_index"] = representative_index
        annotations.append(annotation)
    return annotations


def _stored_consensus_issues(
    row: Mapping[str, Any],
    *,
    record_label: str,
    recomputed: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    expected_pairs = (
        ("status", row.get("status"), recomputed["status"]),
        ("phase", row.get("phase"), recomputed["phase"]),
    )
    stored_consensus = row.get("consensus")
    if not isinstance(stored_consensus, dict):
        issues.append(f"{record_label}: consensus is not an object")
        return issues
    expected_pairs += (
        (
            "consensus.status",
            stored_consensus.get("status"),
            recomputed["status"],
        ),
        (
            "consensus.phase",
            stored_consensus.get("phase"),
            recomputed["phase"],
        ),
        (
            "consensus.phase_counts",
            stored_consensus.get("phase_counts"),
            recomputed["phase_counts"],
        ),
        (
            "consensus.num_representatives_evaluated",
            stored_consensus.get("num_representatives_evaluated"),
            recomputed["representatives_evaluated"],
        ),
        (
            "consensus.num_usable_votes",
            stored_consensus.get("num_usable_votes"),
            recomputed["usable_votes"],
        ),
        (
            "consensus.dominant_votes",
            stored_consensus.get("dominant_votes"),
            recomputed["dominant_votes"],
        ),
    )
    for field, actual, expected in expected_pairs:
        if actual != expected:
            issues.append(
                f"{record_label}: {field} mismatch "
                f"(stored={actual!r}, recomputed={expected!r})"
            )
    return issues


def _main_chain_evaluation(
    *,
    task_description: str,
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family = _task_family(task_description)
    usable_phases = [
        phase
        for annotation in annotations
        if (phase := _usable_phase(annotation)) is not None
    ]
    if family is None:
        return {
            "family": None,
            "evaluable": False,
            "reason": "unsupported-task-family",
            "usable_votes": len(usable_phases),
        }
    config = MAIN_CHAIN_CONFIG[family]
    index = dict(config["phase_ranks"])
    off_main = sorted(
        phase for phase in set(usable_phases) if phase not in index
    )
    if not usable_phases:
        return {
            "family": family,
            "evaluable": False,
            "reason": "no-usable-phase-votes",
            "usable_votes": 0,
            "off_main_phases": [],
        }
    if off_main:
        return {
            "family": family,
            "evaluable": False,
            "reason": "off-main-phase-vote",
            "usable_votes": len(usable_phases),
            "off_main_phases": off_main,
        }
    positions = [index[phase] for phase in usable_phases]
    span = max(positions) - min(positions)
    distinct_phases = len(set(usable_phases))
    return {
        "family": family,
        "evaluable": True,
        "reason": None,
        "usable_votes": len(usable_phases),
        "off_main_phases": [],
        "span": span,
        "distinct_phases": distinct_phases,
        "span_le_1": span <= 1,
        "span_le_2": span <= 2,
        "adjacent_disagreement": (
            distinct_phases == 2 and span == 1
        ),
    }


def _evaluate_final_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_id: str,
    issues: list[str],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    seen_cluster_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        cluster_id = str(row.get("cluster_id", "")).strip()
        record_label = (
            f"{condition_id}: row {row_number}"
            if not cluster_id
            else f"{condition_id}: {cluster_id}"
        )
        if not cluster_id:
            issues.append(f"{record_label}: missing cluster_id")
            cluster_id = f"<missing:{row_number}>"
        elif cluster_id in seen_cluster_ids:
            issues.append(
                f"{condition_id}: duplicate finalized cluster_id {cluster_id}"
            )
        seen_cluster_ids.add(cluster_id)

        task_description = str(row.get("task_description", "")).strip()
        if not task_description:
            issues.append(f"{record_label}: missing task_description")
        annotations = _annotation_list(
            row,
            record_label=record_label,
            issues=issues,
        )
        recomputed = recompute_strict_consensus(annotations)
        issues.extend(
            _stored_consensus_issues(
                row,
                record_label=record_label,
                recomputed=recomputed,
            )
        )
        if len(annotations) not in (3, 5):
            issues.append(
                f"{record_label}: finalized row has "
                f"{len(annotations)} representative records"
            )
        evaluations.append(
            {
                "cluster_id": cluster_id,
                "task_description": task_description,
                "row": row,
                "annotations": annotations,
                "strict_consensus": recomputed,
                "main_chain": _main_chain_evaluation(
                    task_description=task_description,
                    annotations=annotations,
                ),
            }
        )
    return evaluations


def _strict_consensus_summary(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(
        str(evaluation["strict_consensus"]["status"])
        for evaluation in evaluations
    )
    accepted = sum(
        bool(evaluation["strict_consensus"]["accepted"])
        for evaluation in evaluations
    )
    return {
        "finalized_clusters": len(evaluations),
        "accepted_clusters": accepted,
        "accepted_rate": _rate(accepted, len(evaluations)),
        "recomputed_status_counts": dict(sorted(statuses.items())),
        "accepted_phase_counts": dict(
            sorted(
                Counter(
                    str(evaluation["strict_consensus"]["phase"])
                    for evaluation in evaluations
                    if evaluation["strict_consensus"]["accepted"]
                ).items()
            )
        ),
    }


def _dominant_expanded_summary(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expanded = [
        evaluation
        for evaluation in evaluations
        if evaluation["strict_consensus"]["representatives_evaluated"] == 5
    ]
    five_vote = [
        evaluation
        for evaluation in expanded
        if evaluation["strict_consensus"]["usable_votes"] == 5
    ]
    dominant = [
        evaluation
        for evaluation in five_vote
        if evaluation["strict_consensus"]["dominant_votes"] >= 3
    ]
    exactly_three = [
        evaluation
        for evaluation in dominant
        if evaluation["strict_consensus"]["dominant_votes"] == 3
    ]
    return {
        "definition": "exactly 5 usable votes and maximum phase count >= 3",
        "expanded_clusters": len(expanded),
        "five_usable_vote_clusters": len(five_vote),
        "dominant_at_least_3_clusters": len(dominant),
        "dominant_at_least_3_rate_among_five_vote": _rate(
            len(dominant),
            len(five_vote),
        ),
        "exactly_3_of_5_clusters": len(exactly_three),
        "exactly_3_of_5_strictly_rejected_clusters": sum(
            not evaluation["strict_consensus"]["accepted"]
            for evaluation in exactly_three
        ),
    }


def _main_chain_summary(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family, config in MAIN_CHAIN_CONFIG.items():
        selected = [
            evaluation["main_chain"]
            for evaluation in evaluations
            if evaluation["main_chain"]["family"] == family
        ]
        evaluable = [
            evaluation
            for evaluation in selected
            if evaluation["evaluable"]
        ]
        result[family] = {
            "main_chain": list(config["phases"]),
            "phase_ranks": dict(config["phase_ranks"]),
            "clusters": len(selected),
            "evaluable_clusters": len(evaluable),
            "off_main_phase_clusters": sum(
                evaluation.get("reason") == "off-main-phase-vote"
                for evaluation in selected
            ),
            "no_usable_phase_clusters": sum(
                evaluation.get("reason") == "no-usable-phase-votes"
                for evaluation in selected
            ),
            "span_le_1_clusters": sum(
                evaluation["span_le_1"] for evaluation in evaluable
            ),
            "span_le_2_clusters": sum(
                evaluation["span_le_2"] for evaluation in evaluable
            ),
            "adjacent_disagreement_clusters": sum(
                evaluation["adjacent_disagreement"]
                for evaluation in evaluable
            ),
            "span_counts": dict(
                sorted(
                    Counter(
                        str(evaluation["span"])
                        for evaluation in evaluable
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            ),
        }
    supported = [
        evaluation["main_chain"]
        for evaluation in evaluations
        if evaluation["main_chain"]["family"] is not None
    ]
    evaluable_supported = [
        evaluation
        for evaluation in supported
        if evaluation["evaluable"]
    ]
    span_le_1 = sum(
        evaluation["span_le_1"]
        for evaluation in evaluable_supported
    )
    span_le_2 = sum(
        evaluation["span_le_2"]
        for evaluation in evaluable_supported
    )
    adjacent_disagreement = sum(
        evaluation["adjacent_disagreement"]
        for evaluation in evaluable_supported
    )
    result["all_supported_task_families"] = {
        "definition": (
            "all usable phases must lie on the task-family main chain"
        ),
        "clusters": len(supported),
        "evaluable_clusters": len(evaluable_supported),
        "off_main_or_no_vote_clusters": (
            len(supported) - len(evaluable_supported)
        ),
        "span_le_1_clusters": span_le_1,
        "span_le_1_rate_among_all_clusters": _rate(
            span_le_1,
            len(supported),
        ),
        "span_le_1_rate_among_evaluable_clusters": _rate(
            span_le_1,
            len(evaluable_supported),
        ),
        "span_le_2_clusters": span_le_2,
        "span_le_2_rate_among_all_clusters": _rate(
            span_le_2,
            len(supported),
        ),
        "span_le_2_rate_among_evaluable_clusters": _rate(
            span_le_2,
            len(evaluable_supported),
        ),
        "adjacent_disagreement_definition": (
            "exactly two distinct main-chain phases with rank span 1; "
            "unanimous rows are excluded"
        ),
        "adjacent_disagreement_clusters": adjacent_disagreement,
        "adjacent_disagreement_rate_among_all_clusters": _rate(
            adjacent_disagreement,
            len(supported),
        ),
        "adjacent_disagreement_rate_among_evaluable_clusters": _rate(
            adjacent_disagreement,
            len(evaluable_supported),
        ),
    }
    result["unsupported_task_family_clusters"] = sum(
        evaluation["main_chain"]["family"] is None
        for evaluation in evaluations
    )
    return result


def _task_breakdown(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[str(evaluation["task_description"])].append(evaluation)
    result: dict[str, Any] = {}
    for task_description in sorted(grouped):
        selected = grouped[task_description]
        vote_counts = Counter(
            phase
            for evaluation in selected
            for annotation in evaluation["annotations"]
            if (phase := _usable_phase(annotation)) is not None
        )
        result[task_description] = {
            "finalized_clusters": len(selected),
            "strict_accepted_consensus": _strict_consensus_summary(selected),
            "dominant_expanded": _dominant_expanded_summary(selected),
            "main_chain_span": _main_chain_summary(selected),
            "representative_usable_phase_counts": dict(
                sorted(vote_counts.items())
            ),
        }
    return result


def _error_category(error: Any) -> str | None:
    if error is None:
        return None
    text = str(error).lower()
    if (
        "resource_exhausted" in text
        or "quota" in text
        or "429" in text
    ):
        return "quota"
    if (
        "timeout" in text
        or "timed out" in text
        or "deadline" in text
    ):
        return "timeout"
    return "other"


def _attempt_summary(
    records: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    expected_model: str | None,
) -> dict[str, Any]:
    annotations = [
        (origin, annotation)
        for origin, origin_annotations in records
        for annotation in origin_annotations
    ]
    total_attempts = 0
    unknown_attempts = 0
    retried_records = 0
    retry_excess = 0
    model_versions: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    terminal_api_errors = 0
    terminal_parse_errors = 0
    insufficient = 0
    unexpected_model_versions = 0
    missing_model_versions = 0

    for origin, annotation in annotations:
        origins[origin] += 1
        raw_attempts = annotation.get("request_attempts")
        try:
            attempts = int(raw_attempts)
        except (TypeError, ValueError):
            attempts = 0
        if attempts < 1:
            unknown_attempts += 1
        else:
            total_attempts += attempts
            if attempts > 1:
                retried_records += 1
                retry_excess += attempts - 1

        api_error = annotation.get("api_error")
        if api_error is not None:
            terminal_api_errors += 1
            category = _error_category(api_error)
            if category is not None:
                error_categories[category] += 1
        if annotation.get("parse_error") is not None:
            terminal_parse_errors += 1
        if annotation.get("visibility") == "insufficient":
            insufficient += 1

        model_version = annotation.get("model_version")
        if isinstance(model_version, str) and model_version:
            model_versions[model_version] += 1
            if expected_model is not None and model_version != expected_model:
                unexpected_model_versions += 1
        else:
            model_versions["<missing>"] += 1
            missing_model_versions += 1

    return {
        "representative_records": len(annotations),
        "records_by_origin": dict(sorted(origins.items())),
        "request_attempts_total": total_attempts,
        "records_with_unknown_attempt_count": unknown_attempts,
        "retried_representative_records": retried_records,
        "retry_excess_attempts": retry_excess,
        "terminal_api_error_records": terminal_api_errors,
        "terminal_parse_error_records": terminal_parse_errors,
        "terminal_api_error_categories": dict(
            sorted(error_categories.items())
        ),
        "insufficient_visibility_records": insufficient,
        "model_version_counts": dict(sorted(model_versions.items())),
        "expected_model_version": expected_model,
        "unexpected_model_version_records": unexpected_model_versions,
        "missing_model_version_records": missing_model_versions,
        "note": (
            "A successful record can have request_attempts > 1; only the "
            "terminal error fields are retained per representative record."
        ),
    }


def _find_experiment_state(
    states: Sequence[Mapping[str, Any]],
    experiment_id: str,
) -> Mapping[str, Any] | None:
    matches = [
        state
        for state in states
        if str(state["experiment_id"]).upper() == experiment_id
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _annotation_index(
    evaluation: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    return {
        int(annotation["representative_index"]): annotation
        for annotation in evaluation["annotations"]
    }


def _compare_annotation_view_conditions(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_label_condition = _find_experiment_state(states, "E0")
    multiview_label_condition = _find_experiment_state(states, "E1")
    if (
        left_label_condition is None
        or multiview_label_condition is None
    ):
        return {
            "available": False,
            "reason": "exactly one E0 and one E1 condition are required",
        }

    left_label_rows = {
        evaluation["cluster_id"]: evaluation
        for evaluation in left_label_condition["evaluations"]
    }
    multiview_label_rows = {
        evaluation["cluster_id"]: evaluation
        for evaluation in multiview_label_condition["evaluations"]
    }
    common_clusters = sorted(
        set(left_label_rows).intersection(multiview_label_rows)
    )
    transitions: Counter[tuple[str, str]] = Counter()
    paired_indices = 0
    comparable = 0
    agreements = 0
    missing_left_label_indices = 0
    missing_multiview_label_indices = 0
    noncomparable = 0
    representative_set_hash_matches = 0
    representative_set_hash_mismatches = 0
    accepted_cluster_pairs = 0
    accepted_cluster_phase_agreements = 0

    for cluster_id in common_clusters:
        left_label_evaluation = left_label_rows[cluster_id]
        multiview_label_evaluation = multiview_label_rows[cluster_id]
        left_representative_hash = left_label_evaluation["row"].get(
            "representative_set_sha256"
        )
        multiview_representative_hash = multiview_label_evaluation["row"].get(
            "representative_set_sha256"
        )
        if (
            left_representative_hash is not None
            and left_representative_hash == multiview_representative_hash
        ):
            representative_set_hash_matches += 1
        else:
            representative_set_hash_mismatches += 1

        left_label_consensus = left_label_evaluation["strict_consensus"]
        multiview_label_consensus = multiview_label_evaluation[
            "strict_consensus"
        ]
        if (
            left_label_consensus["accepted"]
            and multiview_label_consensus["accepted"]
        ):
            accepted_cluster_pairs += 1
            if (
                left_label_consensus["phase"]
                == multiview_label_consensus["phase"]
            ):
                accepted_cluster_phase_agreements += 1

        left_label_annotations = _annotation_index(left_label_evaluation)
        multiview_label_annotations = _annotation_index(
            multiview_label_evaluation
        )
        all_indices = sorted(
            set(left_label_annotations).union(multiview_label_annotations)
        )
        for representative_index in all_indices:
            if representative_index not in left_label_annotations:
                missing_left_label_indices += 1
                continue
            if representative_index not in multiview_label_annotations:
                missing_multiview_label_indices += 1
                continue
            paired_indices += 1
            left_label_phase = _usable_phase(
                left_label_annotations[representative_index]
            )
            multiview_label_phase = _usable_phase(
                multiview_label_annotations[representative_index]
            )
            if left_label_phase is None or multiview_label_phase is None:
                noncomparable += 1
                continue
            comparable += 1
            transitions[(left_label_phase, multiview_label_phase)] += 1
            if left_label_phase == multiview_label_phase:
                agreements += 1

    return {
        "available": True,
        "e0_condition_id": left_label_condition["condition_id"],
        "e1_condition_id": multiview_label_condition["condition_id"],
        "e0_finalized_clusters": len(left_label_rows),
        "e1_finalized_clusters": len(multiview_label_rows),
        "common_finalized_clusters": len(common_clusters),
        "e0_only_finalized_clusters": len(
            set(left_label_rows).difference(multiview_label_rows)
        ),
        "e1_only_finalized_clusters": len(
            set(multiview_label_rows).difference(left_label_rows)
        ),
        "representative_set_hash_matches": representative_set_hash_matches,
        "representative_set_hash_mismatches": (
            representative_set_hash_mismatches
        ),
        "paired_representative_indices": paired_indices,
        "comparable_phase_pairs": comparable,
        "exact_phase_agreements": agreements,
        "exact_phase_agreement_rate": _rate(agreements, comparable),
        "noncomparable_phase_pairs": noncomparable,
        "missing_e0_representative_indices": missing_left_label_indices,
        "missing_e1_representative_indices": (
            missing_multiview_label_indices
        ),
        "phase_transitions": [
            {
                "e0_phase": left_label_phase,
                "e1_phase": multiview_label_phase,
                "count": count,
            }
            for (left_label_phase, multiview_label_phase), count in sorted(
                transitions.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "strict_accepted_cluster_pairs": accepted_cluster_pairs,
        "strict_accepted_cluster_phase_agreements": (
            accepted_cluster_phase_agreements
        ),
        "strict_accepted_cluster_phase_agreement_rate": _rate(
            accepted_cluster_phase_agreements,
            accepted_cluster_pairs,
        ),
    }


def _condition_state(
    condition: Mapping[str, Any],
    *,
    manifest_path: Path,
    expected_model: str | None,
    issues: list[str],
) -> dict[str, Any]:
    condition_id = str(condition.get("condition_id", "")).strip()
    experiment_id = str(condition.get("experiment_id", "")).strip().upper()
    if not condition_id:
        raise ValueError("Manifest condition is missing condition_id")
    raw_output_path = condition.get("output_path")
    if raw_output_path is None:
        output_path = (
            manifest_path.parent
            / condition_id
            / "adaptive_annotations.jsonl"
        ).resolve()
    else:
        output_path = _resolve_artifact_path(
            str(raw_output_path),
            manifest_path=manifest_path,
        )

    rows = _read_jsonl(output_path, issues=issues)
    evaluations = _evaluate_final_rows(
        rows,
        condition_id=condition_id,
        issues=issues,
    )
    finalized_cluster_ids = {
        evaluation["cluster_id"] for evaluation in evaluations
    }
    inflight_path = _inflight_path(output_path)
    inflight_record: dict[str, Any] | None = None
    inflight_annotations: list[dict[str, Any]] = []
    inflight_cluster_id: str | None = None
    active_inflight = False
    if inflight_path.is_file():
        try:
            raw_inflight = _read_json(inflight_path)
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"{condition_id}: invalid inflight JSON: {exc}")
        else:
            if not isinstance(raw_inflight, dict):
                issues.append(
                    f"{condition_id}: inflight checkpoint is not an object"
                )
            else:
                inflight_record = raw_inflight
                inflight_cluster_id = str(
                    raw_inflight.get("cluster_id", "")
                ).strip() or None
                inflight_annotations = _annotation_list(
                    raw_inflight,
                    record_label=(
                        f"{condition_id}: inflight "
                        f"{inflight_cluster_id or '<missing-cluster>'}"
                    ),
                    issues=issues,
                )
                if inflight_cluster_id is None:
                    issues.append(
                        f"{condition_id}: inflight checkpoint has no cluster_id"
                    )
                elif inflight_cluster_id in finalized_cluster_ids:
                    issues.append(
                        f"{condition_id}: stale inflight checkpoint for "
                        f"finalized cluster {inflight_cluster_id}"
                    )
                else:
                    active_inflight = True

    try:
        expected_clusters = int(condition["selected_cluster_rows"])
    except (KeyError, TypeError, ValueError):
        expected_clusters = 0
        issues.append(
            f"{condition_id}: invalid selected_cluster_rows in manifest"
        )
    finalized_clusters = len(evaluations)
    if finalized_clusters > expected_clusters:
        issues.append(
            f"{condition_id}: finalized cluster count {finalized_clusters} "
            f"exceeds expected {expected_clusters}"
        )
    if active_inflight and finalized_clusters >= expected_clusters:
        issues.append(
            f"{condition_id}: active inflight cluster exists with no "
            "remaining manifest slot"
        )

    if finalized_clusters > expected_clusters:
        progress_status = "overcomplete"
    elif finalized_clusters == expected_clusters and not active_inflight:
        progress_status = "complete"
    elif finalized_clusters or active_inflight:
        progress_status = "partial"
    else:
        progress_status = "not-started"

    final_attempt_groups = [
        (
            f"{condition_id}:final:{evaluation['cluster_id']}",
            evaluation["annotations"],
        )
        for evaluation in evaluations
    ]
    attempt_groups = list(final_attempt_groups)
    if active_inflight:
        attempt_groups.append(
            (
                f"{condition_id}:inflight:{inflight_cluster_id}",
                inflight_annotations,
            )
        )

    condition_output = {
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "output_path": str(output_path),
        "expected_clusters": expected_clusters,
        "finalized_clusters": finalized_clusters,
        "remaining_clusters": max(expected_clusters - finalized_clusters, 0),
        "completion_fraction": _rate(
            min(finalized_clusters, expected_clusters),
            expected_clusters,
        ),
        "progress_status": progress_status,
        "inflight": {
            "present": inflight_record is not None,
            "active": active_inflight,
            "cluster_id": inflight_cluster_id,
            "representative_records": len(inflight_annotations),
            "api_parse_success_records": sum(
                annotation.get("api_error") is None
                and annotation.get("parse_error") is None
                for annotation in inflight_annotations
            ),
            "usable_vote_records": sum(
                _usable_phase(annotation) is not None
                for annotation in inflight_annotations
            ),
            "terminal_api_error_records": sum(
                annotation.get("api_error") is not None
                for annotation in inflight_annotations
            ),
            "terminal_parse_error_records": sum(
                annotation.get("parse_error") is not None
                for annotation in inflight_annotations
            ),
        },
        "strict_accepted_consensus": _strict_consensus_summary(evaluations),
        "dominant_expanded": _dominant_expanded_summary(evaluations),
        "main_chain_span": _main_chain_summary(evaluations),
        "task_breakdown": _task_breakdown(evaluations),
        "attempts_errors_model_versions": _attempt_summary(
            attempt_groups,
            expected_model=expected_model,
        ),
    }
    return {
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "evaluations": evaluations,
        "active_inflight": active_inflight,
        "inflight_annotations": inflight_annotations,
        "attempt_groups": attempt_groups,
        "output": condition_output,
    }


def analyze_annotation_consistency(
    run_manifest_path: str | Path,
) -> dict[str, Any]:
    """Analyze stored V12r3 responses without writing or calling an API."""

    manifest_path = Path(run_manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Run manifest has no contract object")
    raw_conditions = contract.get("conditions")
    if not isinstance(raw_conditions, list):
        raise ValueError("Run manifest contract has no condition list")

    issues: list[str] = []
    declared_contract_hash = manifest.get("contract_sha256")
    recomputed_contract_hash = _canonical_sha256(contract)
    if declared_contract_hash != recomputed_contract_hash:
        issues.append(
            "run_manifest: contract_sha256 mismatch "
            f"(declared={declared_contract_hash!r}, "
            f"recomputed={recomputed_contract_hash!r})"
        )

    expected_model_value = contract.get("model")
    expected_model = (
        str(expected_model_value)
        if isinstance(expected_model_value, str)
        and expected_model_value
        else None
    )
    states = [
        _condition_state(
            condition,
            manifest_path=manifest_path,
            expected_model=expected_model,
            issues=issues,
        )
        for condition in raw_conditions
        if isinstance(condition, dict)
    ]
    if len(states) != len(raw_conditions):
        issues.append("run_manifest: one or more conditions are not objects")

    all_evaluations = [
        evaluation
        for state in states
        for evaluation in state["evaluations"]
    ]
    all_attempt_groups = [
        group
        for state in states
        for group in state["attempt_groups"]
    ]
    expected_clusters = sum(
        int(state["output"]["expected_clusters"]) for state in states
    )
    finalized_clusters = len(all_evaluations)
    active_inflight_clusters = sum(
        bool(state["active_inflight"]) for state in states
    )
    manifest_expected = contract.get("expected_total_cluster_rows")
    if (
        isinstance(manifest_expected, int)
        and manifest_expected != expected_clusters
    ):
        issues.append(
            "run_manifest: expected_total_cluster_rows does not equal "
            "the sum of condition selected_cluster_rows"
        )

    return {
        "format": ANNOTATION_CONSISTENCY_FORMAT,
        "claim_scope": {
            "statement": CLAIM_SCOPE,
            "claim_strength": "diagnostic evidence",
            "oracle_or_human_reference_labels_used": False,
            "accuracy_evaluated": False,
            "causal_or_policy_effect_evaluated": False,
            "interpretation": (
                "Agreement, phase-span, and consensus metrics describe only "
                "consistency among stored Gemini representative judgments."
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "declared_contract_sha256": declared_contract_hash,
            "recomputed_embedded_contract_sha256": (
                recomputed_contract_hash
            ),
            "embedded_contract_hash_matches": (
                declared_contract_hash == recomputed_contract_hash
            ),
            "hash_check_scope": (
                "manifest internal consistency only; current runtime, "
                "package, cluster, and media bytes are not rehashed here"
            ),
            "model": expected_model,
            "prompt_id": contract.get("prompt_id"),
            "response_schema_id": contract.get("response_schema_id"),
            "frozen_at_utc": manifest.get("frozen_at_utc"),
        },
        "progress": {
            "expected_final_clusters": expected_clusters,
            "finalized_clusters": finalized_clusters,
            "remaining_final_clusters": max(
                expected_clusters - finalized_clusters,
                0,
            ),
            "active_inflight_clusters": active_inflight_clusters,
            "completion_fraction": _rate(
                min(finalized_clusters, expected_clusters),
                expected_clusters,
            ),
            "complete_conditions": sum(
                state["output"]["progress_status"] == "complete"
                for state in states
            ),
            "partial_conditions": sum(
                state["output"]["progress_status"] == "partial"
                for state in states
            ),
            "not_started_conditions": sum(
                state["output"]["progress_status"] == "not-started"
                for state in states
            ),
            "overcomplete_conditions": sum(
                state["output"]["progress_status"] == "overcomplete"
                for state in states
            ),
        },
        "strict_accepted_consensus": _strict_consensus_summary(
            all_evaluations
        ),
        "dominant_expanded": _dominant_expanded_summary(all_evaluations),
        "main_chain_span": _main_chain_summary(all_evaluations),
        "task_breakdown": _task_breakdown(all_evaluations),
        "e0_e1_same_cluster_representative_agreement": (
            _compare_annotation_view_conditions(states)
        ),
        "attempts_errors_model_versions": _attempt_summary(
            all_attempt_groups,
            expected_model=expected_model,
        ),
        "conditions": [state["output"] for state in states],
        "integrity": {
            "passed": not issues,
            "issue_count": len(issues),
            "issues": issues,
        },
    }


__all__ = [
    "CLAIM_SCOPE",
    "ANNOTATION_CONSISTENCY_FORMAT",
    "DRAWER_MAIN_CHAIN",
    "MAIN_CHAIN_CONFIG",
    "PICK_PLACE_MAIN_CHAIN",
    "analyze_annotation_consistency",
    "recompute_strict_consensus",
]
