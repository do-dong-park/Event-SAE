#!/usr/bin/env python3
"""Run fixed or adaptive centroid Gemini Batch annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from event_sae.groot.cluster_annotation import (
    CONSENSUS_POLICIES,
    audit_consensus_derivation,
    audit_results,
    audit_user_phase_override,
    derive_consensus_annotations,
    derive_user_phase_override,
    exclusive_run_lock,
    freeze_or_validate_manifest,
    materialize,
)
from event_sae.groot.gemini_batch import (
    audit_transport,
    collect_plan,
    make_client,
    prepare_plan,
    submit_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Explicit new output root. Historical annotation roots are "
            "read-only and rejected by mutating commands."
        ),
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=None,
        help=(
            "Read the Gemini API key from this file. Defaults to "
            "GEMINI_API_KEY; the secret is never persisted."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "freeze",
        help="Freeze or validate the output-root-specific run contract.",
    )
    subparsers.add_parser(
        "prepare",
        help="Compile the next immutable plan for all missing responses.",
    )
    submit = subparsers.add_parser(
        "submit",
        help="Submit every unsubmitted chunk in one immutable plan.",
    )
    submit.add_argument("--plan", type=Path, required=True)
    collect = subparsers.add_parser(
        "collect",
        help="Poll and collect provider jobs for one immutable plan.",
    )
    collect.add_argument("--plan", type=Path, required=True)
    subparsers.add_parser(
        "materialize",
        help="Build condition outputs after every required response succeeds.",
    )
    subparsers.add_parser(
        "audit",
        help="Replay and verify the completed result and transport artifacts.",
    )
    derive = subparsers.add_parser(
        "derive",
        help="Apply an offline consensus policy to a completed source run.",
    )
    derive.add_argument("--source-root", type=Path, required=True)
    derive.add_argument(
        "--policy",
        choices=tuple(CONSENSUS_POLICIES),
        default="unique-plurality",
    )
    subparsers.add_parser(
        "audit-derivation",
        help="Read-only replay audit for a v2 consensus derivation.",
    )
    override = subparsers.add_parser(
        "derive-phase-override",
        help="Derive one explicit user-directed phase override offline.",
    )
    override.add_argument("--source-root", type=Path, required=True)
    override.add_argument("--condition-id", required=True)
    override.add_argument("--cluster-id", required=True)
    override.add_argument("--phase", required=True)
    override.add_argument("--phrase", required=True)
    override.add_argument("--reason", required=True)
    override.add_argument("--excluded-run-root", type=Path, default=None)
    subparsers.add_parser(
        "audit-phase-override",
        help="Replay-audit a user-directed phase override.",
    )
    return parser


def _plan_summary(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {
        "plan_path": str(path),
        "plan_id": plan["plan_id"],
        "logical_request_count": plan["logical_request_count"],
        "chunk_count": len(plan["chunks"]),
        "serialized_payload_bytes": plan["serialized_payload_bytes"],
    }


def _manifest_summary(manifest: dict) -> dict:
    contract = manifest["contract"]
    if contract.get("run_kind") == "adaptive-plurality":
        return {
            "contract_sha256": manifest["contract_sha256"],
            "run_kind": contract["run_kind"],
            "expected_source_rows": contract["expected_source_rows"],
            "expected_target_rows": contract["expected_target_rows"],
            "initial_fresh_requests": contract[
                "initial_fresh_requests"
            ],
            "maximum_fresh_requests": contract[
                "maximum_fresh_requests"
            ],
            "conditions": [
                {
                    "experiment_id": condition["experiment_id"],
                    "condition_id": condition["condition_id"],
                    "source_rows": condition[
                        "source_annotation_rows"
                    ],
                    "targets": condition["target_cluster_rows"],
                    "maximum_fresh_requests": condition[
                        "maximum_fresh_requests"
                    ],
                }
                for condition in contract["conditions"]
            ],
        }
    return {
        "contract_sha256": manifest["contract_sha256"],
        "expected_total_cluster_rows": contract[
            "expected_total_cluster_rows"
        ],
        "expected_logical_requests": contract[
            "expected_logical_requests"
        ],
        "response_reuse_allowed": contract[
            "response_reuse"
        ]["allowed"],
        "conditions": [
            {
                "experiment_id": condition["experiment_id"],
                "condition_id": condition["condition_id"],
                "clusters": condition["selected_cluster_rows"],
                "representatives": condition["representative_count"],
                "images": condition["image_reference_count"],
            }
            for condition in contract["conditions"]
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    output_root = args.output_root.resolve()
    if args.command == "derive":
        result = derive_consensus_annotations(
            source_root=args.source_root,
            output_root=output_root,
            policy=CONSENSUS_POLICIES[args.policy],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if args.command == "audit-derivation":
        result = audit_consensus_derivation(output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if args.command == "derive-phase-override":
        result = derive_user_phase_override(
            source_root=args.source_root,
            output_root=output_root,
            condition_id=args.condition_id,
            cluster_id=args.cluster_id,
            phase=args.phase,
            phrase=args.phrase,
            reason=args.reason,
            excluded_run_root=args.excluded_run_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if args.command == "audit-phase-override":
        result = audit_user_phase_override(output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    with exclusive_run_lock(output_root):
        if args.command == "freeze":
            result = _manifest_summary(
                freeze_or_validate_manifest(output_root)
            )
            result["manifest_path"] = str(
                output_root / "run_manifest.json"
            )
        elif args.command == "prepare":
            result = _plan_summary(prepare_plan(output_root))
        elif args.command == "submit":
            result = submit_plan(
                output_root,
                plan_path=args.plan,
                client=make_client(api_key_path=args.api_key_path),
            )
        elif args.command == "collect":
            result = collect_plan(
                output_root,
                plan_path=args.plan,
                client=make_client(api_key_path=args.api_key_path),
            )
        elif args.command == "materialize":
            result = materialize(output_root)
        elif args.command == "audit":
            results = audit_results(output_root)
            transport = audit_transport(output_root)
            result = {
                "format": "event_sae_cluster_annotation_full_audit_v2",
                "complete": bool(
                    results["complete"] and transport["complete"]
                ),
                "results": results,
                "transport": transport,
            }
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
