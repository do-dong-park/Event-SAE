"""Gemini Batch transport for centroid-five cluster annotation.

The semantic run contract, request construction, response ledger, and final
materialization belong to :mod:`event_sae.groot.cluster_annotation`.  This
module owns only the paid, non-idempotent Gemini Batch lifecycle:

* immutable request plans,
* receipt-before-create submission,
* terminal-job collection,
* exact response normalization, and
* provider-artifact replay audits.

Keeping the boundary explicit prevents provider retries from changing the
centroid-selection or consensus contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import google.genai as genai
from google.genai import types

from event_sae.events.annotate import load_api_key
from event_sae.groot import cluster_annotation as domain
from event_sae.groot.cluster_annotation import (
    atomic_write_json,
    canonical_sha256,
    load_json_object,
    serialize_inlined_request,
    sha256_bytes,
    utc_now,
    serialize_batch_payload,
    write_json_exclusive,
)


PLAN_FORMAT = "event_sae_cluster_annotation_request_plan_v2"
RECEIPT_FORMAT = "event_sae_cluster_annotation_job_receipt_v2"
COLLECTION_FORMAT = "event_sae_cluster_annotation_collection_v2"
TRANSPORT_AUDIT_FORMAT = "event_sae_cluster_annotation_transport_audit_v2"

REQUEST_TIMEOUT_SECONDS = 300.0
TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
RESULT_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _chunk_requests(
    requests: Sequence[tuple[dict[str, Any], types.InlinedRequest]],
    *,
    max_serialized_bytes: int,
) -> list[dict[str, Any]]:
    if max_serialized_bytes < 2:
        raise ValueError("Serialized chunk cap is too small")
    chunks: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    inlined_requests: list[types.InlinedRequest] = []

    def finish_chunk() -> None:
        if not entries:
            return
        payload = serialize_batch_payload(inlined_requests)
        chunks.append(
            {
                "chunk_id": f"chunk-{len(chunks) + 1:03d}",
                "entries": [dict(entry) for entry in entries],
                "request_count": len(entries),
                "serialized_payload_size_bytes": len(payload),
                "serialized_payload_sha256": sha256_bytes(payload),
            }
        )

    for entry, request in requests:
        candidate = [*inlined_requests, request]
        if (
            len(serialize_batch_payload(candidate)) > max_serialized_bytes
            and inlined_requests
        ):
            finish_chunk()
            entries = []
            inlined_requests = []
            candidate = [request]
        if len(serialize_batch_payload(candidate)) > max_serialized_bytes:
            raise ValueError(
                f"Single Batch request exceeds chunk cap: "
                f"{entry['logical_id']}"
            )
        entries.append(entry)
        inlined_requests.append(request)
    finish_chunk()
    return chunks


def _state_value(state: Any) -> str:
    if state is None:
        return "JOB_STATE_UNSPECIFIED"
    return str(getattr(state, "value", state))


def _normalized_model(value: Any) -> str:
    return str(value or "").removeprefix("models/")


def _job_summary(job: Any) -> dict[str, Any]:
    return {
        "name": getattr(job, "name", None),
        "display_name": getattr(job, "display_name", None),
        "state": _state_value(getattr(job, "state", None)),
        "model": getattr(job, "model", None),
        "create_time": str(getattr(job, "create_time", None) or ""),
        "start_time": str(getattr(job, "start_time", None) or ""),
        "update_time": str(getattr(job, "update_time", None) or ""),
        "end_time": str(getattr(job, "end_time", None) or ""),
    }


def _job_error_text(job: Any) -> str:
    error = getattr(job, "error", None)
    if error is None:
        return "Batch job ended without per-request responses"
    if hasattr(error, "model_dump"):
        error = error.model_dump(mode="json", exclude_none=True)
    return domain.redact_sensitive_error(
        error if isinstance(error, str) else str(error)
    )


def _responses_by_request_key(
    *,
    job: Any,
    job_name: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    destination = getattr(job, "dest", None)
    responses = (
        getattr(destination, "inlined_responses", None)
        if destination is not None
        else None
    )
    if not isinstance(responses, list):
        raise ValueError(
            f"Batch result job has no inline responses: {job_name}"
        )
    if len(responses) != len(entries):
        raise ValueError(
            f"Batch response count mismatch for {job_name}: "
            f"responses={len(responses)} requests={len(entries)}"
        )
    result: dict[str, Any] = {}
    for response in responses:
        metadata = getattr(response, "metadata", None)
        key = metadata.get("key") if isinstance(metadata, Mapping) else None
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(
                f"Batch response has an invalid request key: {job_name}"
            )
        result[key] = response
    expected = {str(entry["request_key"]) for entry in entries}
    if set(result) != expected:
        raise ValueError(
            f"Batch response key set mismatch for {job_name}"
        )
    return result


def _plan_body(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"created_at_utc", "plan_sha256", "plan_id"}
    }


def _plans_root(output_root: Path) -> Path:
    return Path(output_root).resolve() / "plans"


def _all_plan_paths(output_root: Path) -> list[Path]:
    root = _plans_root(output_root)
    return sorted(root.glob("*.json")) if root.exists() else []


def _collection_path(output_root: Path, plan_id: str) -> Path:
    return (
        Path(output_root).resolve()
        / "collections"
        / f"{plan_id}.json"
    )


def _receipt_path(
    output_root: Path,
    plan_id: str,
    chunk_id: str,
) -> Path:
    return (
        Path(output_root).resolve()
        / "jobs"
        / plan_id
        / f"{chunk_id}.json"
    )


def _outstanding_plan(output_root: Path) -> Path | None:
    outstanding = [
        path
        for path in _all_plan_paths(output_root)
        if not _collection_path(output_root, path.stem).exists()
    ]
    if len(outstanding) > 1:
        raise ValueError(
            f"Multiple outstanding centroid Batch plans: {outstanding}"
        )
    return outstanding[0] if outstanding else None


def _compiled_requirements(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[Any],
) -> list[tuple[dict[str, Any], types.InlinedRequest]]:
    """Normalize the domain's outstanding-request descriptions.

    The preferred domain interface returns ``(entry, request)`` pairs.  A
    four-item semantic requirement is also accepted when the domain exposes a
    ``request_descriptor`` adapter.  Supporting the latter keeps transport
    independent of the concrete condition-context type.
    """

    compiled: list[tuple[dict[str, Any], types.InlinedRequest]] = []
    for requirement in domain.required_requests(output_root, contexts):
        if (
            isinstance(requirement, Sequence)
            and not isinstance(requirement, (str, bytes))
            and len(requirement) == 2
            and isinstance(requirement[0], Mapping)
        ):
            entry = dict(requirement[0])
            request = requirement[1]
        elif (
            isinstance(requirement, Sequence)
            and not isinstance(requirement, (str, bytes))
            and len(requirement) == 4
            and callable(getattr(domain, "request_descriptor", None))
        ):
            entry, request = domain.request_descriptor(*requirement)
            entry = dict(entry)
        else:
            raise TypeError(
                "required_requests must return (entry, InlinedRequest) pairs "
                "or four-item requirements backed by request_descriptor"
            )
        if not isinstance(request, types.InlinedRequest):
            raise TypeError(
                f"{entry.get('logical_id')}: expected InlinedRequest"
            )
        _, _, regenerated = domain.validate_plan_entry(
            output_root,
            manifest,
            contexts,
            entry,
        )
        if not isinstance(regenerated, types.InlinedRequest):
            raise TypeError(
                f"{entry.get('logical_id')}: regenerated request is invalid"
            )
        if serialize_inlined_request(regenerated) != serialize_inlined_request(request):
            raise ValueError(
                f"{entry.get('logical_id')}: domain request regeneration "
                "changed before plan freeze"
            )
        compiled.append((entry, request))
    return compiled


def prepare_plan(output_root: Path) -> Path:
    """Freeze the next immutable fixed-wave request plan.

    A completed plan is returned as an idempotent no-op when every logical
    request already succeeded.  Failed attempts result in a new plan containing
    only the next attempts supplied by the domain.
    """

    output_root = Path(output_root).resolve()
    manifest = domain.freeze_or_validate_manifest(output_root)
    outstanding = _outstanding_plan(output_root)
    if outstanding is not None:
        return outstanding
    contexts = domain.load_contexts(manifest)
    compiled = _compiled_requirements(
        output_root=output_root,
        manifest=manifest,
        contexts=contexts,
    )
    if not compiled:
        previous = _all_plan_paths(output_root)
        if previous:
            return previous[-1]
        raise RuntimeError(
            "Centroid run has no requests and no prior immutable plan"
        )

    chunks = _chunk_requests(
        compiled,
        max_serialized_bytes=domain.MAX_SERIALIZED_CHUNK_BYTES,
    )
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "run_contract_sha256": manifest["contract_sha256"],
        "wave": domain.batch_wave(manifest),
        "model": domain.MODEL,
        "expected_model_version": domain.EXPECTED_MODEL_VERSION,
        "max_serialized_chunk_bytes": (
            domain.MAX_SERIALIZED_CHUNK_BYTES
        ),
        "logical_request_count": len(compiled),
        "serialized_payload_bytes": sum(
            int(chunk["serialized_payload_size_bytes"])
            for chunk in chunks
        ),
        "chunks": chunks,
    }
    plan_sha = canonical_sha256(_plan_body(plan))
    plan_id = f"centroid-{plan_sha[:20]}"
    plan.update(
        {
            "plan_id": plan_id,
            "plan_sha256": plan_sha,
            "created_at_utc": utc_now(),
        }
    )
    plan_path = _plans_root(output_root) / f"{plan_id}.json"
    if plan_path.exists():
        existing = load_json_object(plan_path)
        if (
            _plan_body(existing) != _plan_body(plan)
            or existing.get("plan_sha256") != plan_sha
            or existing.get("plan_id") != plan_id
        ):
            raise FileExistsError(f"Centroid plan collision: {plan_path}")
    else:
        write_json_exclusive(plan_path, plan)
    return plan_path


def _regenerate_chunk_requests(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    contexts: Sequence[Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> list[types.InlinedRequest]:
    requests: list[types.InlinedRequest] = []
    for entry in chunk["entries"]:
        _, _, request = domain.validate_plan_entry(
            output_root,
            manifest,
            contexts,
            entry,
        )
        if not isinstance(request, types.InlinedRequest):
            raise TypeError(
                f"{entry.get('logical_id')}: regenerated request is invalid"
            )
        requests.append(request)
    payload = serialize_batch_payload(requests)
    if (
        len(payload) != int(chunk["serialized_payload_size_bytes"])
        or sha256_bytes(payload)
        != chunk["serialized_payload_sha256"]
    ):
        raise ValueError(
            "Centroid Batch chunk payload drifted: "
            f"{plan['plan_id']}/{chunk['chunk_id']}"
        )
    if len(payload) > domain.MAX_SERIALIZED_CHUNK_BYTES:
        raise ValueError(
            "Centroid Batch chunk exceeds the frozen byte cap: "
            f"{plan['plan_id']}/{chunk['chunk_id']}"
        )
    return requests


def _load_plan(
    output_root: Path,
    plan_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[Any],
]:
    output_root = Path(output_root).resolve()
    manifest = domain.freeze_or_validate_manifest(output_root)
    contexts = domain.load_contexts(manifest)
    plan_path = Path(plan_path).resolve()
    if plan_path.parent != _plans_root(output_root).resolve():
        raise ValueError(f"Plan is outside this centroid run: {plan_path}")
    plan = load_json_object(plan_path)
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"Invalid centroid Batch plan format: {plan_path}")
    if plan.get("run_contract_sha256") != manifest["contract_sha256"]:
        raise ValueError(f"Centroid Batch plan contract drifted: {plan_path}")
    recomputed_sha = canonical_sha256(_plan_body(plan))
    if (
        plan.get("plan_sha256") != recomputed_sha
        or plan.get("plan_id") != f"centroid-{recomputed_sha[:20]}"
        or plan_path.stem != plan.get("plan_id")
    ):
        raise ValueError(f"Centroid Batch plan hash is invalid: {plan_path}")
    scalar_expectations = {
        "wave": domain.batch_wave(manifest),
        "model": domain.MODEL,
        "expected_model_version": domain.EXPECTED_MODEL_VERSION,
        "max_serialized_chunk_bytes": (
            domain.MAX_SERIALIZED_CHUNK_BYTES
        ),
    }
    mismatched = [
        field
        for field, value in scalar_expectations.items()
        if plan.get(field) != value
    ]
    if mismatched:
        raise ValueError(
            f"Centroid Batch plan settings drifted: {mismatched}"
        )

    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Centroid Batch plan has no chunks: {plan_path}")
    expected_chunk_ids = [
        f"chunk-{index:03d}"
        for index in range(1, len(chunks) + 1)
    ]
    actual_chunk_ids = [
        str(chunk.get("chunk_id", ""))
        if isinstance(chunk, Mapping)
        else ""
        for chunk in chunks
    ]
    if actual_chunk_ids != expected_chunk_ids:
        raise ValueError(f"Centroid Batch chunk IDs drifted: {plan_path}")

    entry_keys: set[tuple[str, int]] = set()
    logical_request_count = 0
    serialized_payload_bytes = 0
    for chunk in chunks:
        entries = chunk.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"Centroid Batch chunk entries are invalid: {plan_path}"
            )
        if int(chunk.get("request_count", -1)) != len(entries):
            raise ValueError(
                f"Centroid Batch chunk count drifted: {plan_path}"
            )
        payload_size = int(
            chunk.get("serialized_payload_size_bytes", -1)
        )
        if (
            payload_size < 0
            or payload_size > domain.MAX_SERIALIZED_CHUNK_BYTES
        ):
            raise ValueError(
                f"Centroid Batch chunk size is invalid: {plan_path}"
            )
        logical_request_count += len(entries)
        serialized_payload_bytes += payload_size
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Centroid Batch plan entry is invalid: {plan_path}"
                )
            representative_index = int(
                entry.get("representative_index", -1)
            )
            attempt = int(entry.get("attempt", -1))
            logical_id = str(entry.get("logical_id", ""))
            if representative_index not in (
                domain.requestable_representative_indices(manifest)
            ):
                raise ValueError(
                    f"Centroid representative index is invalid: {logical_id}"
                )
            if attempt < 1 or attempt > domain.MAX_REQUEST_ATTEMPTS:
                raise ValueError(
                    f"Centroid request attempt is invalid: {logical_id}"
                )
            entry_key = (logical_id, attempt)
            if entry_key in entry_keys:
                raise ValueError(
                    f"Duplicate centroid plan entry: {entry_key}"
                )
            entry_keys.add(entry_key)
        _regenerate_chunk_requests(
            output_root=output_root,
            manifest=manifest,
            contexts=contexts,
            plan=plan,
            chunk=chunk,
        )
    if int(plan.get("logical_request_count", -1)) != logical_request_count:
        raise ValueError(
            f"Centroid Batch logical request count drifted: {plan_path}"
        )
    if (
        int(plan.get("serialized_payload_bytes", -1))
        != serialized_payload_bytes
    ):
        raise ValueError(
            f"Centroid Batch serialized bytes drifted: {plan_path}"
        )
    return plan, manifest, list(contexts)


def _display_name(plan_id: str, chunk_id: str) -> str:
    digest = sha256_bytes(f"{plan_id}|{chunk_id}".encode("utf-8"))
    return f"event-sae-centroid-annotation-{digest[:24]}"


def _receipt_contract(
    *,
    run_contract_sha256: str,
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    chunk_id = str(chunk["chunk_id"])
    return {
        "format": RECEIPT_FORMAT,
        "run_contract_sha256": run_contract_sha256,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "chunk_id": chunk_id,
        "chunk_payload_sha256": chunk[
            "serialized_payload_sha256"
        ],
        "request_count": int(chunk["request_count"]),
        "request_keys_sha256": canonical_sha256(
            [
                str(entry["request_key"])
                for entry in chunk["entries"]
            ]
        ),
        "display_name": _display_name(
            str(plan["plan_id"]),
            chunk_id,
        ),
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    receipt_path: Path,
) -> None:
    expected = _receipt_contract(
        run_contract_sha256=run_contract_sha256,
        plan=plan,
        chunk=chunk,
    )
    mismatched = [
        field
        for field, value in expected.items()
        if receipt.get(field) != value
    ]
    if mismatched:
        raise ValueError(
            f"Centroid Batch receipt drifted at {receipt_path}: "
            f"{mismatched}"
        )


def _matching_provider_jobs(
    client: genai.Client,
    display_name: str,
) -> list[Any]:
    return [
        job
        for job in client.batches.list()
        if getattr(job, "display_name", None) == display_name
    ]


def _verify_provider_job(
    job: Any,
    *,
    display_name: str,
) -> None:
    if getattr(job, "display_name", None) != display_name:
        raise ValueError("Provider Batch display name drifted")
    if _normalized_model(getattr(job, "model", None)) != domain.MODEL:
        raise ValueError(
            f"Provider Batch model drifted: {getattr(job, 'model', None)}"
        )
    name = getattr(job, "name", None)
    if not isinstance(name, str) or not name.startswith("batches/"):
        raise ValueError(f"Provider Batch has invalid name: {name!r}")


def _adopt_ambiguous_create(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    client: genai.Client,
) -> dict[str, Any]:
    matches = _matching_provider_jobs(
        client,
        str(receipt["display_name"]),
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Ambiguous non-idempotent centroid Batch create cannot be "
            "retried automatically. Provider jobs with deterministic "
            f"display name={receipt['display_name']!r}: {len(matches)}"
        )
    job = matches[0]
    _verify_provider_job(
        job,
        display_name=str(receipt["display_name"]),
    )
    updated = {
        **dict(receipt),
        "state": "submitted",
        "adopted_after_ambiguous_create": True,
        "submitted_at_utc": utc_now(),
        "job": _job_summary(job),
    }
    atomic_write_json(receipt_path, updated)
    return updated


def submit_plan(
    output_root: Path,
    *,
    plan_path: Path,
    client: genai.Client,
) -> dict[str, Any]:
    """Submit every unsubmitted immutable chunk exactly once."""

    output_root = Path(output_root).resolve()
    plan, manifest, contexts = _load_plan(output_root, plan_path)
    submitted = 0
    existing_jobs = 0
    jobs = []
    for chunk in plan["chunks"]:
        chunk_id = str(chunk["chunk_id"])
        receipt_path = _receipt_path(
            output_root,
            str(plan["plan_id"]),
            chunk_id,
        )
        display_name = _display_name(str(plan["plan_id"]), chunk_id)
        if receipt_path.exists():
            receipt = load_json_object(receipt_path)
            _validate_receipt(
                receipt,
                run_contract_sha256=str(
                    manifest["contract_sha256"]
                ),
                plan=plan,
                chunk=chunk,
                receipt_path=receipt_path,
            )
            if receipt.get("state") == "create_started":
                receipt = _adopt_ambiguous_create(
                    receipt=receipt,
                    receipt_path=receipt_path,
                    client=client,
                )
            if not isinstance(receipt.get("job"), Mapping):
                raise ValueError(
                    f"Centroid Batch receipt has no job: {receipt_path}"
                )
            existing_jobs += 1
            jobs.append(receipt["job"])
            continue

        requests = _regenerate_chunk_requests(
            output_root=output_root,
            manifest=manifest,
            contexts=contexts,
            plan=plan,
            chunk=chunk,
        )
        unexpected = _matching_provider_jobs(client, display_name)
        if unexpected:
            raise RuntimeError(
                "Provider already has a job for an unrecorded centroid "
                f"Batch chunk: display_name={display_name!r}, "
                f"count={len(unexpected)}"
            )
        receipt = {
            **_receipt_contract(
                run_contract_sha256=str(
                    manifest["contract_sha256"]
                ),
                plan=plan,
                chunk=chunk,
            ),
            "state": "create_started",
            "create_started_at_utc": utc_now(),
            "adopted_after_ambiguous_create": False,
        }
        write_json_exclusive(receipt_path, receipt)
        job = client.batches.create(
            model=domain.MODEL,
            src=requests,
            config=types.CreateBatchJobConfig(
                display_name=display_name,
            ),
        )
        _verify_provider_job(job, display_name=display_name)
        receipt.update(
            {
                "state": "submitted",
                "submitted_at_utc": utc_now(),
                "job": _job_summary(job),
            }
        )
        atomic_write_json(receipt_path, receipt)
        submitted += 1
        jobs.append(receipt["job"])
    return {
        "plan_id": plan["plan_id"],
        "logical_request_count": int(plan["logical_request_count"]),
        "chunk_count": len(plan["chunks"]),
        "new_jobs_submitted": submitted,
        "existing_jobs_reused": existing_jobs,
        "jobs": jobs,
    }


def _refresh_submitted_job(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    client: genai.Client,
) -> tuple[Any, str, str]:
    chunk_id = str(chunk["chunk_id"])
    receipt_path = _receipt_path(
        output_root,
        str(plan["plan_id"]),
        chunk_id,
    )
    if not receipt_path.exists():
        raise FileNotFoundError(
            f"Centroid Batch chunk is not submitted: {receipt_path}"
        )
    receipt = load_json_object(receipt_path)
    _validate_receipt(
        receipt,
        run_contract_sha256=str(manifest["contract_sha256"]),
        plan=plan,
        chunk=chunk,
        receipt_path=receipt_path,
    )
    if receipt.get("state") == "create_started":
        receipt = _adopt_ambiguous_create(
            receipt=receipt,
            receipt_path=receipt_path,
            client=client,
        )
    job_summary = receipt.get("job")
    if not isinstance(job_summary, Mapping):
        raise ValueError(f"Centroid receipt has no provider job: {receipt_path}")
    job_name = str(job_summary["name"])
    job = client.batches.get(name=job_name)
    _verify_provider_job(
        job,
        display_name=str(receipt["display_name"]),
    )
    current_summary = _job_summary(job)
    state = str(current_summary["state"])
    receipt.update(
        {
            "state": (
                "terminal"
                if state in TERMINAL_JOB_STATES
                else "submitted"
            ),
            "last_checked_at_utc": utc_now(),
            "job": current_summary,
        }
    )
    atomic_write_json(receipt_path, receipt)
    return job, job_name, state


def _record_job_failure(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    entry: Mapping[str, Any],
    job_name: str,
    state: str,
    error: str,
) -> bool:
    return domain.record_job_failure(
        output_root=output_root,
        manifest=manifest,
        plan=plan,
        chunk=chunk,
        entry=entry,
        job_name=job_name,
        state=state,
        error=error,
    )


def _context_lookup(
    contexts: Sequence[Any],
) -> dict[tuple[str, str], tuple[Any, Mapping[str, Any]]]:
    lookup: dict[
        tuple[str, str],
        tuple[Any, Mapping[str, Any]],
    ] = {}
    for context in contexts:
        for cluster in context.targeted_rows:
            key = (
                str(context.condition_id),
                str(cluster["cluster_id"]),
            )
            if key in lookup:
                raise ValueError(f"Duplicate centroid condition/cluster: {key}")
            lookup[key] = (context, cluster)
    return lookup


def _collect_inline_entry(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    entry: Mapping[str, Any],
    inline_response: Any,
    lookup: Mapping[
        tuple[str, str],
        tuple[Any, Mapping[str, Any]],
    ],
    job_name: str,
) -> str | None:
    key = (
        str(entry["condition_id"]),
        str(entry["cluster_id"]),
    )
    if key not in lookup:
        raise ValueError(f"Response points to an unknown centroid cluster: {key}")
    context, cluster = lookup[key]
    record, success = domain.build_inline_attempt_record(
        manifest=manifest,
        plan=plan,
        chunk=chunk,
        entry=entry,
        context=context,
        cluster=cluster,
        inline_response=inline_response,
        job_name=job_name,
    )
    created = domain.write_attempt_record(
        output_root,
        entry,
        record,
    )
    if not created:
        return None
    return "success" if success else "failure"


def collect_plan(
    output_root: Path,
    *,
    plan_path: Path,
    client: genai.Client,
) -> dict[str, Any]:
    """Collect terminal chunks and persist normalized response attempts."""

    output_root = Path(output_root).resolve()
    plan, manifest, contexts = _load_plan(output_root, plan_path)
    lookup = _context_lookup(contexts)
    pending = []
    terminal = []
    new_attempts = 0
    success_attempts = 0
    failed_attempts = 0
    for chunk in plan["chunks"]:
        chunk_id = str(chunk["chunk_id"])
        job, job_name, state = _refresh_submitted_job(
            output_root=output_root,
            manifest=manifest,
            plan=plan,
            chunk=chunk,
            client=client,
        )
        state_record = {
            "chunk_id": chunk_id,
            "job_name": job_name,
            "state": state,
        }
        if state not in TERMINAL_JOB_STATES:
            pending.append(state_record)
            continue
        terminal.append(state_record)
        if state not in RESULT_JOB_STATES:
            error = _job_error_text(job)
            for entry in chunk["entries"]:
                if _record_job_failure(
                    output_root=output_root,
                    manifest=manifest,
                    plan=plan,
                    chunk=chunk,
                    entry=entry,
                    job_name=job_name,
                    state=state,
                    error=error,
                ):
                    new_attempts += 1
                    failed_attempts += 1
            continue

        response_by_key = _responses_by_request_key(
            job=job,
            job_name=job_name,
            entries=chunk["entries"],
        )
        for entry in chunk["entries"]:
            outcome = _collect_inline_entry(
                output_root=output_root,
                manifest=manifest,
                plan=plan,
                chunk=chunk,
                entry=entry,
                inline_response=response_by_key[
                    str(entry["request_key"])
                ],
                lookup=lookup,
                job_name=job_name,
            )
            if outcome is not None:
                new_attempts += 1
                success_attempts += int(outcome == "success")
                failed_attempts += int(outcome == "failure")

    complete = not pending
    result = {
        "format": COLLECTION_FORMAT,
        "run_contract_sha256": manifest["contract_sha256"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "wave": domain.batch_wave(manifest),
        "complete": complete,
        "logical_request_count": int(plan["logical_request_count"]),
        "new_attempts_collected": new_attempts,
        "new_successes": success_attempts,
        "new_failures": failed_attempts,
        "pending_jobs": pending,
        "terminal_jobs": terminal,
        "checked_at_utc": utc_now(),
    }
    if complete:
        collection_path = _collection_path(
            output_root,
            str(plan["plan_id"]),
        )
        if collection_path.exists():
            existing = load_json_object(collection_path)
            immutable_fields = (
                "format",
                "run_contract_sha256",
                "plan_id",
                "plan_sha256",
                "wave",
                "complete",
                "logical_request_count",
            )
            if any(
                existing.get(field) != result.get(field)
                for field in immutable_fields
            ):
                raise ValueError(
                    f"Centroid collection marker drifted: {collection_path}"
                )
        else:
            write_json_exclusive(collection_path, result)
    return result


def _read_attempt_records(
    output_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    responses_root = Path(output_root).resolve() / "responses"
    if not responses_root.exists():
        return []
    records = []
    for path in sorted(responses_root.rglob("attempt-*.json")):
        records.append((path, load_json_object(path)))
    return records


def audit_transport(output_root: Path) -> dict[str, Any]:
    """Replay every plan, receipt, collection, and response linkage."""

    output_root = Path(output_root).resolve()
    manifest = domain.freeze_or_validate_manifest(output_root)
    run_contract_sha256 = str(manifest["contract_sha256"])
    contexts = domain.load_contexts(manifest)
    lookup = _context_lookup(contexts)
    plan_paths = _all_plan_paths(output_root)
    if not plan_paths:
        raise ValueError("Transport audit found no centroid Batch plan")
    loaded = [
        _load_plan(output_root, plan_path)[0]
        for plan_path in plan_paths
    ]

    expected_receipts: set[Path] = set()
    entry_links: dict[
        tuple[str, int],
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    for plan in loaded:
        for chunk in plan["chunks"]:
            _regenerate_chunk_requests(
                output_root=output_root,
                manifest=manifest,
                contexts=contexts,
                plan=plan,
                chunk=chunk,
            )
            receipt_path = _receipt_path(
                output_root,
                str(plan["plan_id"]),
                str(chunk["chunk_id"]),
            ).resolve()
            expected_receipts.add(receipt_path)
            for entry in chunk["entries"]:
                entry_key = (
                    str(entry["logical_id"]),
                    int(entry["attempt"]),
                )
                if entry_key in entry_links:
                    raise ValueError(
                        "Centroid request attempt appears in multiple plans: "
                        f"{entry_key}"
                    )
                entry_links[entry_key] = (plan, chunk, entry)

    jobs_root = output_root / "jobs"
    actual_receipts = (
        {path.resolve() for path in jobs_root.rglob("*.json")}
        if jobs_root.exists()
        else set()
    )
    if actual_receipts != expected_receipts:
        raise ValueError(
            "Centroid Batch receipts do not exactly match planned chunks"
        )

    receipt_jobs: dict[tuple[str, str], dict[str, Any]] = {}
    seen_job_names: set[str] = set()
    seen_display_names: set[str] = set()
    for plan in loaded:
        for chunk in plan["chunks"]:
            receipt_path = _receipt_path(
                output_root,
                str(plan["plan_id"]),
                str(chunk["chunk_id"]),
            )
            receipt = load_json_object(receipt_path)
            _validate_receipt(
                receipt,
                run_contract_sha256=run_contract_sha256,
                plan=plan,
                chunk=chunk,
                receipt_path=receipt_path,
            )
            if receipt.get("state") != "terminal":
                raise ValueError(
                    f"Centroid Batch receipt is not terminal: {receipt_path}"
                )
            job = receipt.get("job")
            if not isinstance(job, dict):
                raise ValueError(
                    f"Centroid Batch receipt has no job: {receipt_path}"
                )
            job_name = job.get("name")
            display_name = job.get("display_name")
            state = str(job.get("state", ""))
            if (
                not isinstance(job_name, str)
                or not job_name.startswith("batches/")
                or display_name != receipt["display_name"]
                or _normalized_model(job.get("model")) != domain.MODEL
                or state not in TERMINAL_JOB_STATES
            ):
                raise ValueError(
                    f"Centroid Batch job contract drifted: {receipt_path}"
                )
            if job_name in seen_job_names:
                raise ValueError(
                    f"Duplicate centroid provider job name: {job_name}"
                )
            if str(display_name) in seen_display_names:
                raise ValueError(
                    "Duplicate centroid provider display name: "
                    f"{display_name}"
                )
            seen_job_names.add(job_name)
            seen_display_names.add(str(display_name))
            receipt_jobs[
                (str(plan["plan_id"]), str(chunk["chunk_id"]))
            ] = job

    expected_collections = {
        _collection_path(output_root, str(plan["plan_id"])).resolve()
        for plan in loaded
    }
    collections_root = output_root / "collections"
    actual_collections = (
        {path.resolve() for path in collections_root.glob("*.json")}
        if collections_root.exists()
        else set()
    )
    if actual_collections != expected_collections:
        raise ValueError(
            "Centroid collection markers do not exactly match plans"
        )
    for plan in loaded:
        collection_path = _collection_path(
            output_root,
            str(plan["plan_id"]),
        )
        collection = load_json_object(collection_path)
        expected_fields = {
            "format": COLLECTION_FORMAT,
            "run_contract_sha256": run_contract_sha256,
            "plan_id": plan["plan_id"],
            "plan_sha256": plan["plan_sha256"],
            "wave": domain.batch_wave(manifest),
            "complete": True,
            "logical_request_count": int(plan["logical_request_count"]),
            "pending_jobs": [],
        }
        mismatched = [
            field
            for field, value in expected_fields.items()
            if collection.get(field) != value
        ]
        if mismatched:
            raise ValueError(
                f"Centroid collection drifted at {collection_path}: "
                f"{mismatched}"
            )
        terminal_jobs = collection.get("terminal_jobs")
        expected_jobs = {
            str(chunk["chunk_id"]): receipt_jobs[
                (str(plan["plan_id"]), str(chunk["chunk_id"]))
            ]
            for chunk in plan["chunks"]
        }
        if (
            not isinstance(terminal_jobs, list)
            or len(terminal_jobs) != len(expected_jobs)
        ):
            raise ValueError(
                f"Centroid terminal jobs drifted: {collection_path}"
            )
        seen_chunks: set[str] = set()
        for terminal_job in terminal_jobs:
            if not isinstance(terminal_job, Mapping):
                raise ValueError(
                    f"Invalid centroid terminal job: {collection_path}"
                )
            chunk_id = str(terminal_job.get("chunk_id", ""))
            if chunk_id in seen_chunks or chunk_id not in expected_jobs:
                raise ValueError(
                    f"Centroid terminal chunk drifted: {collection_path}"
                )
            seen_chunks.add(chunk_id)
            receipt_job = expected_jobs[chunk_id]
            if (
                terminal_job.get("job_name") != receipt_job["name"]
                or terminal_job.get("state") != receipt_job["state"]
            ):
                raise ValueError(
                    f"Centroid terminal job linkage drifted: {chunk_id}"
                )

    attempt_records = _read_attempt_records(output_root)
    actual_attempts: dict[tuple[str, int], dict[str, Any]] = {}
    response_ids: list[str] = []
    logical_attempts: dict[str, list[dict[str, Any]]] = {}
    for path, record in attempt_records:
        key = (
            str(record.get("logical_id", "")),
            int(record.get("attempt", -1)),
        )
        if key in actual_attempts:
            raise ValueError(f"Duplicate centroid attempt record: {key}")
        actual_attempts[key] = record
        if key not in entry_links:
            raise ValueError(f"Unplanned centroid attempt record: {path}")
        plan, chunk, entry = entry_links[key]
        expected_fields = {
            "format": domain.ATTEMPT_FORMAT,
            "origin": "fresh_provider_batch",
            "run_contract_sha256": run_contract_sha256,
            "logical_id": entry["logical_id"],
            "condition_id": entry["condition_id"],
            "cluster_id": entry["cluster_id"],
            "representative_index": int(
                entry["representative_index"]
            ),
            "attempt": int(entry["attempt"]),
            "request_key": entry["request_key"],
            "semantic_request_sha256": entry[
                "semantic_request_sha256"
            ],
            "wire_request_sha256": entry["wire_request_sha256"],
            "plan_id": plan["plan_id"],
            "chunk_id": chunk["chunk_id"],
            "job_name": receipt_jobs[
                (str(plan["plan_id"]), str(chunk["chunk_id"]))
            ]["name"],
        }
        mismatched = [
            field
            for field, value in expected_fields.items()
            if record.get(field) != value
        ]
        if mismatched:
            raise ValueError(
                f"Centroid response linkage drifted at {path}: {mismatched}"
            )
        status = str(record.get("status", ""))
        job = receipt_jobs[
            (str(plan["plan_id"]), str(chunk["chunk_id"]))
        ]
        if status == "job_error":
            if (
                job["state"] in RESULT_JOB_STATES
                or not isinstance(record.get("error"), str)
                or not str(record["error"]).startswith(
                    f"{job['state']}:"
                )
                or any(
                    record.get(field) is not None
                    for field in (
                        "raw_response",
                        "provider_raw_response",
                        "response_normalization",
                        "batch_response_sha256",
                        "provider_inline_response",
                        "representative_annotation",
                    )
                )
            ):
                raise ValueError(
                    f"Centroid job-error attempt drifted: {path}"
                )
            logical_attempts.setdefault(key[0], []).append(record)
            continue
        if job["state"] not in RESULT_JOB_STATES:
            raise ValueError(
                f"Response-bearing attempt is linked to a job without "
                f"results: {path}"
            )
        inline_dump = record.get("provider_inline_response")
        if (
            not isinstance(inline_dump, Mapping)
            or canonical_sha256(inline_dump)
            != record.get("batch_response_sha256")
        ):
            raise ValueError(
                f"Centroid provider envelope drifted: {path}"
            )
        inline_response = types.InlinedResponse.model_validate(
            dict(inline_dump)
        )
        context, cluster = lookup[
            (
                str(entry["condition_id"]),
                str(entry["cluster_id"]),
            )
        ]
        expected_record, expected_success = (
            domain.build_inline_attempt_record(
                manifest=manifest,
                plan=plan,
                chunk=chunk,
                entry=entry,
                context=context,
                cluster=cluster,
                inline_response=inline_response,
                job_name=str(job["name"]),
            )
        )
        actual_comparable = dict(record)
        expected_comparable = dict(expected_record)
        actual_comparable.pop("collected_at_utc", None)
        expected_comparable.pop("collected_at_utc", None)
        if actual_comparable != expected_comparable:
            raise ValueError(
                f"Centroid response attempt does not replay: {path}"
            )
        if expected_success != (status == "success"):
            raise ValueError(
                f"Centroid response success state drifted: {path}"
            )
        if status == "success":
            annotation = record["representative_annotation"]
            response_id = annotation["response_id"]
            response_ids.append(str(response_id))
        logical_attempts.setdefault(key[0], []).append(record)

    if set(actual_attempts) != set(entry_links):
        raise ValueError(
            "Centroid response attempts do not exactly match plan entries"
        )
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Duplicate centroid provider response IDs detected")
    for logical_id, records in logical_attempts.items():
        ordered = sorted(records, key=lambda item: int(item["attempt"]))
        numbers = [int(record["attempt"]) for record in ordered]
        if numbers != list(range(1, len(ordered) + 1)):
            raise ValueError(
                f"Centroid attempt sequence drifted: {logical_id}"
            )
        if len(
            {
                str(record["semantic_request_sha256"])
                for record in ordered
            }
        ) != 1:
            raise ValueError(
                f"Centroid semantic request drifted across retries: "
                f"{logical_id}"
            )
        if ordered[-1].get("status") != "success":
            raise ValueError(
                f"Centroid logical request is incomplete: {logical_id}"
            )
        if any(
            record.get("status") == "success"
            for record in ordered[:-1]
        ):
            raise ValueError(
                f"Centroid attempts continued after success: {logical_id}"
            )
        if domain.successful_attempt(output_root, logical_id) is None:
            raise ValueError(
                f"Domain cannot replay centroid success: {logical_id}"
            )

    expected_logical_ids = set(
        domain.expected_logical_request_ids(
            output_root,
            manifest,
            contexts,
        )
    )
    if set(logical_attempts) != expected_logical_ids:
        raise ValueError(
            "Planned centroid requests do not match the domain contract"
        )

    return {
        "format": TRANSPORT_AUDIT_FORMAT,
        "audited_at_utc": utc_now(),
        "run_contract_sha256": run_contract_sha256,
        "plan_count": len(loaded),
        "chunk_count": len(expected_receipts),
        "provider_job_count": len(seen_job_names),
        "response_attempt_count": len(entry_links),
        "successful_logical_requests": len(logical_attempts),
        "unique_response_ids": len(set(response_ids)),
        "complete": True,
    }


def make_client(
    *,
    api_key_path: Path | None = None,
) -> genai.Client:
    """Build a no-SDK-retry client for explicit Batch attempt accounting."""

    return genai.Client(
        api_key=load_api_key(api_key_path),
        http_options=types.HttpOptions(
            timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


__all__ = [
    "audit_transport",
    "collect_plan",
    "make_client",
    "prepare_plan",
    "submit_plan",
]
