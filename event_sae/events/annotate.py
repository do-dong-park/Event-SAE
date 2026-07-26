"""Gemini VLM annotation of task-local event clusters.

For each cluster, sends the prompt (`prompts.build_cluster_annotation_prompt`)
plus the cluster's representative frame sequences (image bytes inline) to
Gemini, parses the JSON response into `{phrase, phase}`, and writes one
JSONL row per cluster.

API key: read from `GEMINI_API_KEY` env var, or pass `--api-key-path` to
read from a file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import google.genai as genai
from google.genai import types

from event_sae.events.io import load_jsonl
from event_sae.events.prompts import (
    AnnotationProtocol,
    PAPER_ANNOTATION_PROTOCOL,
    PHASE_LABELS,
    build_cluster_annotation_prompt,
)

RESPONSE_MIME_TYPE = "application/json"
_ANNOTATION_RESPONSE_SCHEMA_ID = "event_sae_cluster_phrase_phase_v1"
_ANNOTATION_PROMPT_INPUT_POLICY = {
    "task_instruction": True,
    "representative_images": True,
    "media_layout_description": True,
    "cluster_id": True,
    "episode_coverage": True,
    "relative_progress": False,
    "source_frame_step": False,
}


def _build_annotation_response_schema(
    allowed_phase_labels: tuple[str, ...],
) -> dict:
    """Return the exact structured-output schema sent to Gemini."""

    return {
        "type": "object",
        "properties": {
            "phrase": {"type": "string", "minLength": 1},
            "phase": {"type": "string", "enum": list(allowed_phase_labels)},
        },
        "required": ["phrase", "phase"],
        "additionalProperties": False,
    }


def load_api_key(api_key_path: Path | None = None) -> str:
    """Resolve a Gemini API key from env var or file."""
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    if api_key_path is None:
        raise ValueError(
            "GEMINI_API_KEY not set and no --api-key-path provided. "
            "Set GEMINI_API_KEY or pass a path to a text file containing the key."
        )
    api_key_path = Path(api_key_path).resolve()
    if not api_key_path.is_file():
        raise FileNotFoundError(f"Gemini API key file not found: {api_key_path}")
    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"Gemini API key file is empty: {api_key_path}")
    return api_key


def call_gemini(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
    frame_path_groups: list[list[str]],
    allowed_phase_labels: tuple[str, ...] = PHASE_LABELS,
    temperature: float = 0.2,
) -> str:
    contents: list[types.Part | str] = [prompt]
    for sequence_idx, frame_paths in enumerate(frame_path_groups, start=1):
        contents.append(
            f"BEGIN CLIP {sequence_idx}: the next {len(frame_paths)} images are consecutive "
            "frames from one short clip in chronological order."
        )
        for frame_idx, frame_path in enumerate(frame_paths, start=1):
            contents.append(
                f"Clip {sequence_idx}, frame {frame_idx}/{len(frame_paths)}:"
            )
            frame_path = Path(frame_path)
            suffix = frame_path.suffix.lower()
            mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
                suffix
            )
            if mime_type is None:
                raise ValueError(f"Unsupported representative frame type: {frame_path}")
            contents.append(types.Part.from_bytes(data=frame_path.read_bytes(), mime_type=mime_type))
        contents.append(f"END CLIP {sequence_idx}")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type=RESPONSE_MIME_TYPE,
            response_json_schema=_build_annotation_response_schema(
                allowed_phase_labels
            ),
        ),
    )
    return response.text


def parse_annotation_response(
    response_text: str,
    allowed_phase_labels: tuple[str, ...] = PHASE_LABELS,
) -> tuple[str | None, str | None, str | None]:
    """Parse `{phrase, phase}` JSON; returns (phrase, phase, error)."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return None, None, f"json_decode_error: {exc}"
    if not isinstance(parsed, dict):
        return None, None, f"top_level_json_not_object:{type(parsed).__name__}"

    phrase = parsed.get("phrase")
    phase = parsed.get("phase")
    parse_errors: list[str] = []
    unexpected_keys = sorted(set(parsed).difference({"phrase", "phase"}))
    if unexpected_keys:
        parse_errors.append(f"unexpected_keys:{unexpected_keys}")
    allowed_phase_set = set(allowed_phase_labels)

    if not isinstance(phrase, str) or not phrase.strip():
        phrase = None
        parse_errors.append("missing_or_invalid_phrase")
    else:
        phrase = phrase.strip()

    if not isinstance(phase, str):
        phase = None
        parse_errors.append("missing_or_invalid_phase")
    else:
        phase = phase.strip()
        if phase not in allowed_phase_set:
            parse_errors.append(f"invalid_phase:{phase}")

    if parse_errors:
        return phrase, phase, "; ".join(parse_errors)
    return phrase, phase, None


def annotate_clusters(
    clusters_path: Path,
    output_path: Path,
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_clusters: int | None = None,
    cluster_ids: tuple[str, ...] | None = None,
    min_episode_coverage: float | None = None,
    protocol: AnnotationProtocol = PAPER_ANNOTATION_PROTOCOL,
    media_layout_override: str | None = None,
    request_timeout_seconds: float = 120.0,
) -> None:
    """Annotate each cluster with Gemini and write one JSONL row per cluster."""
    model = model or protocol.default_model
    clusters_path = Path(clusters_path).resolve()
    if not clusters_path.is_file():
        raise FileNotFoundError(f"clusters.jsonl not found: {clusters_path}")
    output_path = Path(output_path).resolve()
    if output_path.exists() and output_path.stat().st_size > 0:
        raise FileExistsError(f"Refusing to overwrite annotation output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if min_episode_coverage is not None and not 0.0 <= min_episode_coverage <= 1.0:
        raise ValueError("min_episode_coverage must be between 0 and 1, or None")

    clusters = load_jsonl(clusters_path)
    if cluster_ids:
        requested_cluster_ids = list(dict.fromkeys(cluster_ids))
        if len(requested_cluster_ids) != len(cluster_ids):
            raise ValueError("cluster_ids must not contain duplicates")
        available_cluster_ids = {str(cluster["cluster_id"]) for cluster in clusters}
        missing_cluster_ids = sorted(set(requested_cluster_ids) - available_cluster_ids)
        if missing_cluster_ids:
            raise ValueError(f"Unknown cluster_ids: {missing_cluster_ids}")
        requested_cluster_id_set = set(requested_cluster_ids)
        clusters = [
            cluster for cluster in clusters if cluster["cluster_id"] in requested_cluster_id_set
        ]
    if min_episode_coverage is not None:
        clusters = [
            cluster
            for cluster in clusters
            if float(cluster["episode_coverage"]) >= min_episode_coverage
        ]
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
    if not clusters:
        raise ValueError("No clusters matched the annotation criteria")

    if api_key is None:
        api_key = load_api_key()
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(request_timeout_seconds * 1000),
        ),
    )

    with output_path.open("w", encoding="utf-8") as out_file:
        for idx, cluster in enumerate(clusters, start=1):
            frame_path_groups = cluster["representative_frame_paths"]
            cluster_media_layout = cluster.get("annotation_media_layout")
            if (
                media_layout_override is not None
                and cluster_media_layout is not None
                and media_layout_override != cluster_media_layout
            ):
                raise ValueError(
                    f"{cluster['cluster_id']}: media layout override "
                    f"{media_layout_override!r} conflicts with cluster metadata "
                    f"{cluster_media_layout!r}"
                )
            media_layout = media_layout_override or cluster_media_layout
            progress_percents = [
                float(p) for p in cluster.get("representative_progress_percents", [])
            ]
            phase_labels, _ = protocol.resolve_vocabulary(
                cluster["task_description"]
            )
            prompt = build_cluster_annotation_prompt(
                task_description=cluster["task_description"],
                cluster_id=cluster["cluster_id"],
                num_sequences=len(frame_path_groups),
                num_frames_per_sequence=len(frame_path_groups[0]) if frame_path_groups else 0,
                episode_coverage=float(cluster["episode_coverage"]),
                media_layout=media_layout,
                protocol=protocol,
            )
            record = {
                "cluster_id": cluster["cluster_id"],
                "task_description": cluster["task_description"],
                "model": model,
                "prompt_version": protocol.prompt_version,
                "prompt_text": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_input_policy": dict(
                    _ANNOTATION_PROMPT_INPUT_POLICY
                ),
                "generation_config": {
                    "temperature": float(temperature),
                    "response_mime_type": RESPONSE_MIME_TYPE,
                    "response_schema_version": _ANNOTATION_RESPONSE_SCHEMA_ID,
                    "response_json_schema": _build_annotation_response_schema(
                        phase_labels
                    ),
                },
                "transport_config": {
                    "request_timeout_seconds": float(
                        request_timeout_seconds
                    ),
                },
                "phase_labeler_provenance": dict(
                    protocol.phase_labeler_provenance
                ),
                "phase_scheme": protocol.scheme,
                "allowed_phase_labels": list(phase_labels),
                "representative_sample_ids": cluster["representative_sample_ids"],
                "representative_clip_paths": cluster["representative_clip_paths"],
                "representative_frame_paths": frame_path_groups,
                "annotation_media_layout": media_layout,
                "representative_progress_percents": progress_percents,
                "episode_coverage": cluster["episode_coverage"],
                "annotation_min_episode_coverage": min_episode_coverage,
            }
            try:
                response_text = call_gemini(
                    client=client,
                    model=model,
                    prompt=prompt,
                    frame_path_groups=frame_path_groups,
                    allowed_phase_labels=phase_labels,
                    temperature=temperature,
                )
                record["raw_response"] = response_text
                phrase, phase, parse_error = parse_annotation_response(
                    response_text, phase_labels
                )
                record["phrase"] = phrase
                record["phase"] = phase
                record["parse_error"] = parse_error
                record["api_error"] = None
            except Exception as exc:
                record["raw_response"] = None
                record["phrase"] = None
                record["phase"] = None
                record["parse_error"] = None
                record["api_error"] = f"{type(exc).__name__}: {exc}"

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            print(
                f"[{idx}/{len(clusters)}] cluster_id={cluster['cluster_id']} "
                f"api_error={record['api_error'] is not None} "
                f"parse_error={record['parse_error'] is not None}"
            )
