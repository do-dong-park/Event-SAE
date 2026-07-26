from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from google.genai import types
import pytest

from event_sae.groot import cluster_annotation as domain
from event_sae.groot import gemini_batch as transport


def _request(key: str, text: str = "prompt") -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            )
        ],
        metadata={"key": key},
    )


def _plan() -> dict:
    entry = {
        "logical_id": "condition|cluster|representative-1",
        "condition_id": "condition",
        "cluster_id": "cluster",
        "representative_index": 1,
        "attempt": 1,
        "request_key": "request-1",
    }
    chunks = transport._chunk_requests(
        [(entry, _request("request-1"))],
        max_serialized_bytes=10_000,
    )
    return {
        "plan_id": "centroid-test",
        "plan_sha256": "plan-sha",
        "logical_request_count": 1,
        "chunks": chunks,
    }


def test_chunking_is_deterministic_and_respects_cap() -> None:
    requests = [
        (
            {
                "logical_id": f"logical-{index}",
                "request_key": f"request-{index}",
            },
            _request(f"request-{index}", "x" * 80),
        )
        for index in range(4)
    ]
    one = transport._chunk_requests(
        requests,
        max_serialized_bytes=500,
    )
    two = transport._chunk_requests(
        requests,
        max_serialized_bytes=500,
    )
    assert one == two
    assert [chunk["chunk_id"] for chunk in one] == [
        f"chunk-{index:03d}" for index in range(1, len(one) + 1)
    ]
    assert all(
        chunk["serialized_payload_size_bytes"] <= 500
        for chunk in one
    )


def test_plan_round_trip_uses_domain_batch_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {"contract_sha256": "run"}
    entry = {
        "logical_id": "condition|cluster|representative-6",
        "representative_index": 6,
        "attempt": 1,
        "request_key": "request-1",
    }
    request = _request("request-1")
    monkeypatch.setattr(
        domain,
        "freeze_or_validate_manifest",
        lambda _: manifest,
    )
    monkeypatch.setattr(domain, "load_contexts", lambda _: [])
    monkeypatch.setattr(
        domain,
        "batch_wave",
        lambda _: "adaptive-seven",
        raising=False,
    )
    monkeypatch.setattr(
        domain,
        "requestable_representative_indices",
        lambda _: (6, 7, 8, 9),
        raising=False,
    )
    monkeypatch.setattr(
        transport,
        "_compiled_requirements",
        lambda **_: [(entry, request)],
    )
    monkeypatch.setattr(
        domain,
        "validate_plan_entry",
        lambda *_: (None, None, request),
    )

    plan_path = transport.prepare_plan(tmp_path)
    plan, loaded_manifest, contexts = transport._load_plan(
        tmp_path,
        plan_path,
    )

    assert plan["wave"] == "adaptive-seven"
    assert loaded_manifest is manifest
    assert contexts == []


def test_collect_and_audit_use_domain_transport_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {"contract_sha256": "run"}
    plan = {
        "plan_id": "adaptive-test",
        "plan_sha256": "plan-sha",
        "logical_request_count": 0,
        "chunks": [],
    }
    monkeypatch.setattr(
        transport,
        "_load_plan",
        lambda *_: (plan, manifest, []),
    )
    monkeypatch.setattr(
        domain,
        "freeze_or_validate_manifest",
        lambda _: manifest,
    )
    monkeypatch.setattr(domain, "load_contexts", lambda _: [])
    monkeypatch.setattr(
        domain,
        "batch_wave",
        lambda _: "adaptive-seven",
        raising=False,
    )

    collected = transport.collect_plan(
        tmp_path,
        plan_path=tmp_path / "plans/adaptive-test.json",
        client=SimpleNamespace(),
    )
    assert collected["wave"] == "adaptive-seven"

    monkeypatch.setattr(
        transport,
        "_all_plan_paths",
        lambda _: [tmp_path / "plans/adaptive-test.json"],
    )
    expected_id_calls: list[tuple[Path, object, object]] = []

    def expected_logical_request_ids(
        output_root: Path,
        current_manifest: object,
        contexts: object,
    ) -> set[str]:
        expected_id_calls.append(
            (output_root, current_manifest, contexts)
        )
        return set()

    monkeypatch.setattr(
        domain,
        "expected_logical_request_ids",
        expected_logical_request_ids,
        raising=False,
    )

    audited = transport.audit_transport(tmp_path)

    assert audited["complete"] is True
    assert expected_id_calls == [(tmp_path.resolve(), manifest, [])]


def test_receipt_contract_rejects_drift(tmp_path: Path) -> None:
    plan = _plan()
    chunk = plan["chunks"][0]
    receipt = transport._receipt_contract(
        run_contract_sha256="run",
        plan=plan,
        chunk=chunk,
    )
    transport._validate_receipt(
        receipt,
        run_contract_sha256="run",
        plan=plan,
        chunk=chunk,
        receipt_path=tmp_path / "receipt.json",
    )
    receipt["plan_sha256"] = "drifted"
    with pytest.raises(ValueError, match="receipt drifted"):
        transport._validate_receipt(
            receipt,
            run_contract_sha256="run",
            plan=plan,
            chunk=chunk,
            receipt_path=tmp_path / "receipt.json",
        )


def test_submit_is_idempotent_after_receipt_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    manifest = {"contract_sha256": "run"}
    monkeypatch.setattr(
        transport,
        "_load_plan",
        lambda *_: (plan, manifest, []),
    )
    monkeypatch.setattr(
        transport,
        "_regenerate_chunk_requests",
        lambda **_: [_request("request-1")],
    )
    monkeypatch.setattr(
        transport,
        "_matching_provider_jobs",
        lambda *_: [],
    )

    created: list[object] = []

    def create(**_: object) -> object:
        job = SimpleNamespace(
            name="batches/job-1",
            display_name=transport._display_name(
                plan["plan_id"],
                plan["chunks"][0]["chunk_id"],
            ),
            model=domain.MODEL,
            state="JOB_STATE_PENDING",
        )
        created.append(job)
        return job

    client = SimpleNamespace(
        batches=SimpleNamespace(create=create),
    )
    first = transport.submit_plan(
        tmp_path,
        plan_path=tmp_path / "plans/centroid-test.json",
        client=client,
    )
    second = transport.submit_plan(
        tmp_path,
        plan_path=tmp_path / "plans/centroid-test.json",
        client=client,
    )
    assert first["new_jobs_submitted"] == 1
    assert second["new_jobs_submitted"] == 0
    assert second["existing_jobs_reused"] == 1
    assert len(created) == 1


def test_response_key_mapping_rejects_missing_or_duplicate_keys() -> None:
    entries = [{"request_key": "a"}, {"request_key": "b"}]
    valid = SimpleNamespace(
        dest=SimpleNamespace(
            inlined_responses=[
                SimpleNamespace(metadata={"key": "b"}),
                SimpleNamespace(metadata={"key": "a"}),
            ]
        )
    )
    assert set(
        transport._responses_by_request_key(
            job=valid,
            job_name="batches/test",
            entries=entries,
        )
    ) == {"a", "b"}
    invalid = SimpleNamespace(
        dest=SimpleNamespace(
            inlined_responses=[
                SimpleNamespace(metadata={"key": "a"}),
                SimpleNamespace(metadata={"key": "a"}),
            ]
        )
    )
    with pytest.raises(ValueError, match="invalid request key"):
        transport._responses_by_request_key(
            job=invalid,
            job_name="batches/test",
            entries=entries,
        )


def test_job_failure_provenance_is_redacted_and_idempotent(
    tmp_path: Path,
) -> None:
    entry = {
        "logical_id": "condition|cluster|representative-1",
        "condition_id": "condition",
        "cluster_id": "cluster",
        "representative_index": 1,
        "attempt": 1,
        "request_key": "request-1",
        "semantic_request_sha256": "semantic",
        "wire_request_sha256": "wire",
    }
    keyword = "AIzaSecretShouldNeverPersist"
    kwargs = {
        "output_root": tmp_path,
        "manifest": {"contract_sha256": "run"},
        "plan": {"plan_id": "plan"},
        "chunk": {"chunk_id": "chunk"},
        "entry": entry,
        "job_name": "batches/job",
        "state": "JOB_STATE_FAILED",
        "error": keyword,
    }
    assert domain.record_job_failure(**kwargs) is True
    assert domain.record_job_failure(**kwargs) is False
    record = domain.load_attempts(tmp_path, entry["logical_id"])[0]
    assert keyword not in record["error"]
    assert "[REDACTED_GOOGLE_API_KEY]" in record["error"]
    assert record["representative_annotation"] is None
