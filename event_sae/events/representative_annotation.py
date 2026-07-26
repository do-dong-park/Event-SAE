"""Provider-independent contracts for representative-clip annotation.

Each representative clip is classified independently from five timestamps.
Representative sampling and cross-clip consensus are injected by the caller;
this module owns only request, response, media, and record validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from event_sae.events.prompts import (
    SEPARATE_MULTIVIEW_LAYOUT,
    SINGLE_VIEW_LEFT_LAYOUT,
    resolve_representative_phase_vocabulary,
)


RESPONSE_MIME_TYPE = "application/json"
RESPONSE_SCHEMA_ID = "event_sae_representative_clip_clean_visual_v12r3"
VIEW_ORDER = ("left", "right", "wrist")
SUPPORTED_MEDIA_LAYOUTS = (
    SINGLE_VIEW_LEFT_LAYOUT,
    SEPARATE_MULTIVIEW_LAYOUT,
)
TIMESTAMPS_PER_REPRESENTATIVE = 5
NOMINAL_TIMESTAMP_RELATIVE_SECONDS = (-0.2, -0.1, 0.0, 0.1, 0.2)
TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS = 0.05
VISIBILITY_LABELS = ("clear", "partial", "insufficient")
UNRESOLVED_PHASE = "unresolved"
REQUEST_EXPOSURE_POLICY = {
    "task_instruction": True,
    "representative_image_bytes": True,
    "view_and_time_layout": True,
    "phase_vocabulary_and_visual_definitions": True,
    "cluster_id": False,
    "episode_coverage": False,
    "sample_or_episode_id": False,
    "representative_selection_metadata": False,
    "source_path_or_filename": False,
    "source_step_or_progress": False,
    "awe_anchor_metadata": False,
    "oracle_phase_or_timeline": False,
    "simulator_state_or_predicates": False,
    "success_or_failure": False,
    "phase_taxonomy_provenance": False,
}


def resolve_annotation_media_layout(
    media_layout: str,
) -> tuple[str, tuple[str, ...], str]:
    """Return annotation view, request view order, and cluster media field."""

    if media_layout == SINGLE_VIEW_LEFT_LAYOUT:
        return (
            "left",
            ("left",),
            "representative_single_view_frame_paths",
        )
    if media_layout == SEPARATE_MULTIVIEW_LAYOUT:
        return (
            "multiview",
            VIEW_ORDER,
            "representative_source_view_frame_paths",
        )
    raise ValueError(
        f"Unsupported representative media layout: {media_layout!r}; "
        f"expected one of {SUPPORTED_MEDIA_LAYOUTS}"
    )


def redact_sensitive_error(error_text: str) -> str:
    """Remove credential-shaped values before persisting an exception."""

    redacted = re.sub(
        r"AIza[0-9A-Za-z_-]+",
        "[REDACTED_GOOGLE_API_KEY]",
        str(error_text),
    )
    redacted = re.sub(
        r"(?i)(x-goog-api-key\s*[:=]\s*)([^\s,;&]+)",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(api[_-]?key\s*=\s*)([^\s,;&]+)",
        r"\1[REDACTED]",
        redacted,
    )
    return redacted


def build_representative_response_schema(task_description: str) -> dict:
    """Return the minimal clean-visual schema sent for one clip."""

    phase_labels, _ = resolve_representative_phase_vocabulary(task_description)
    return {
        "type": "object",
        "properties": {
            "phrase": {"type": "string", "minLength": 1},
            "phase": {
                "type": "string",
                "enum": [*phase_labels, UNRESOLVED_PHASE],
            },
            "visibility": {
                "type": "string",
                "enum": list(VISIBILITY_LABELS),
            },
        },
        "required": ["phrase", "phase", "visibility"],
        "additionalProperties": False,
    }


def parse_representative_response(
    response_text: str,
    *,
    task_description: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and defensively validate one representative response."""

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(parsed, dict):
        return None, f"top_level_json_not_object:{type(parsed).__name__}"

    errors: list[str] = []
    required = {"phrase", "phase", "visibility"}
    unexpected = sorted(set(parsed).difference(required))
    missing = sorted(required.difference(parsed))
    if unexpected:
        errors.append(f"unexpected_keys:{unexpected}")
    if missing:
        errors.append(f"missing_keys:{missing}")

    phrase = parsed.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        errors.append("invalid_phrase")
    else:
        parsed["phrase"] = phrase.strip()

    phase_labels, _ = resolve_representative_phase_vocabulary(task_description)
    phase = parsed.get("phase")
    if phase not in {*phase_labels, UNRESOLVED_PHASE}:
        errors.append(f"invalid_phase:{phase!r}")

    visibility = parsed.get("visibility")
    if visibility not in VISIBILITY_LABELS:
        errors.append(f"invalid_visibility:{visibility!r}")
    if phase == UNRESOLVED_PHASE and visibility != "insufficient":
        errors.append("unresolved_phase_requires_insufficient_visibility")
    if phase in phase_labels and visibility == "insufficient":
        errors.append("insufficient_visibility_requires_unresolved_phase")

    if errors:
        return parsed, "; ".join(errors)
    parsed["raw_phase"] = phase
    parsed["phase"] = None if phase == UNRESOLVED_PHASE else phase
    return parsed, None


def _mime_type(path: Path) -> str:
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported representative frame type: {path}")
    return mime_type


def read_clean_image_bytes(path: Path) -> bytes:
    """Read one frame while refusing embedded text or provenance metadata."""

    allowed_info_keys = (
        {"jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi"}
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else set()
    )
    with Image.open(path) as image:
        unexpected_info = sorted(set(image.info).difference(allowed_info_keys))
        has_exif = bool(image.getexif())
    if unexpected_info or has_exif:
        raise ValueError(
            f"Dirty inline image metadata is forbidden: {path}; "
            f"info_keys={unexpected_info}, has_exif={has_exif}"
        )
    return path.read_bytes()


def hash_annotation_media(
    source_view_frames: Sequence[Mapping[str, str]],
    *,
    view_order: Sequence[str] = VIEW_ORDER,
) -> str:
    """Hash timestamp/view labels and exact image bytes in request order."""

    digest = hashlib.sha256()
    for frame_index, frame in enumerate(source_view_frames, start=1):
        for view in view_order:
            path = Path(frame[view]).resolve()
            digest.update(f"{frame_index}:{view}\0".encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _timestamp_label(timestamp_index: int) -> str:
    """Return an ordinal marker without overstating render-time alignment."""

    if not 1 <= timestamp_index <= len(NOMINAL_TIMESTAMP_RELATIVE_SECONDS):
        raise ValueError(f"Invalid timestamp index: {timestamp_index}")
    return (
        f"T{timestamp_index} (CENTER)"
        if timestamp_index == 3
        else f"T{timestamp_index}"
    )


def serialize_usage_metadata(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    return dict(usage)


@dataclass(frozen=True)
class AnnotationImagePart:
    """Provider-neutral image payload in a representative request."""

    data: bytes
    mime_type: str


def build_representative_request_parts(
    *,
    prompt: str,
    source_view_frames: Sequence[Mapping[str, str]],
    view_order: Sequence[str] = VIEW_ORDER,
) -> list[str | AnnotationImagePart]:
    """Build ordered text/image request parts without a provider SDK."""

    if not source_view_frames:
        raise ValueError("source_view_frames must not be empty")
    view_order = tuple(view_order)
    if not view_order:
        raise ValueError("view_order must not be empty")
    if len(source_view_frames) != len(NOMINAL_TIMESTAMP_RELATIVE_SECONDS):
        raise ValueError(
            "Representative annotation requires exactly "
            f"{len(NOMINAL_TIMESTAMP_RELATIVE_SECONDS)} timestamps"
        )

    parts: list[str | AnnotationImagePart] = [prompt]
    for frame_index, source_paths in enumerate(source_view_frames, start=1):
        if set(source_paths) != set(view_order):
            raise ValueError(
                f"frame {frame_index}: views={sorted(source_paths)}, "
                f"expected={view_order}"
            )
        timestamp_label = _timestamp_label(frame_index)
        parts.append(
            f"BEGIN {timestamp_label}; timestamp "
            f"{frame_index}/{len(source_view_frames)}"
        )
        for view in view_order:
            path = Path(source_paths[view]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {view} frame: {path}")
            parts.append(f"{timestamp_label}, VIEW {view.upper()}:")
            parts.append(
                AnnotationImagePart(
                    data=read_clean_image_bytes(path),
                    mime_type=_mime_type(path),
                )
            )
        parts.append(
            f"END {timestamp_label}; timestamp "
            f"{frame_index}/{len(source_view_frames)}"
        )
    return parts


def validate_representative_media(
    cluster: Mapping[str, Any],
    *,
    media_layout: str = SEPARATE_MULTIVIEW_LAYOUT,
) -> list[list[Mapping[str, str]]]:
    _, view_order, source_field = resolve_annotation_media_layout(media_layout)
    source_groups = cluster.get(source_field)
    if not isinstance(source_groups, list) or not source_groups:
        raise ValueError(
            f"{cluster.get('cluster_id')}: missing {source_field}"
        )
    sample_ids = cluster.get("representative_sample_ids")
    clip_paths = cluster.get("representative_clip_paths")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(source_groups):
        raise ValueError(
            f"{cluster.get('cluster_id')}: representative sample/media mismatch"
        )
    if not isinstance(clip_paths, list) or len(clip_paths) != len(source_groups):
        raise ValueError(
            f"{cluster.get('cluster_id')}: representative clip/media mismatch"
        )
    validated: list[list[Mapping[str, str]]] = []
    for group_index, group in enumerate(source_groups, start=1):
        if (
            not isinstance(group, list)
            or len(group) != TIMESTAMPS_PER_REPRESENTATIVE
        ):
            raise ValueError(
                f"{cluster.get('cluster_id')}: representative {group_index} "
                f"must have exactly {TIMESTAMPS_PER_REPRESENTATIVE} timestamps"
            )
        normalized_group: list[Mapping[str, str]] = []
        for frame_index, frame in enumerate(group, start=1):
            if media_layout == SINGLE_VIEW_LEFT_LAYOUT:
                if not isinstance(frame, str) or not frame:
                    raise ValueError(
                        f"{cluster.get('cluster_id')}: representative "
                        f"{group_index} frame {frame_index} must be one LEFT path"
                    )
                normalized_frame: Mapping[str, str] = {"left": frame}
            elif (
                isinstance(frame, dict)
                and set(frame) == set(view_order)
                and all(isinstance(frame[view], str) for view in view_order)
            ):
                normalized_frame = {
                    view: str(frame[view]) for view in view_order
                }
            else:
                raise ValueError(
                    f"{cluster.get('cluster_id')}: representative {group_index} "
                    f"frame {frame_index} has invalid views"
                )
            normalized_group.append(normalized_frame)
        validated.append(normalized_group)
    return validated


def hash_canonical_record(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_representative_set(cluster: Mapping[str, Any]) -> str:
    """Hash ordered representative identity independently of annotation view."""

    identity = {
        "cluster_id": str(cluster["cluster_id"]),
        "task_description": str(cluster["task_description"]),
        "representative_sample_ids": [
            str(value) for value in cluster["representative_sample_ids"]
        ],
        "representative_episode_nums": [
            int(value) for value in cluster["representative_episode_nums"]
        ],
        "representative_selection_strategy": str(
            cluster["representative_selection_strategy"]
        ),
        "representative_selection_roles": [
            str(value) for value in cluster["representative_selection_roles"]
        ],
    }
    return hash_canonical_record(identity)


def _annotation_succeeded(annotation: Mapping[str, Any]) -> bool:
    return (
        annotation.get("api_error") is None
        and annotation.get("parse_error") is None
    )


def validate_representative_annotation(
    annotation: Mapping[str, Any],
    *,
    task_description: str,
) -> None:
    """Validate persisted response fields against the original model JSON."""

    if not isinstance(annotation, Mapping):
        raise ValueError("Representative annotation must be an object")
    attempts = annotation.get("request_attempts")
    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
        or attempts < 1
    ):
        raise ValueError("Representative request_attempts must be positive")

    api_error = annotation.get("api_error")
    parse_error = annotation.get("parse_error")
    for field, error in (
        ("api_error", api_error),
        ("parse_error", parse_error),
    ):
        if error is not None and (
            not isinstance(error, str) or not error.strip()
        ):
            raise ValueError(f"Representative {field} must be null or text")
        if isinstance(error, str) and redact_sensitive_error(error) != error:
            raise ValueError(f"Representative {field} contains a credential")
    if api_error is not None and parse_error is not None:
        raise ValueError(
            "Representative cannot contain both API and parse errors"
        )
    if not _annotation_succeeded(annotation):
        return

    raw_response = annotation.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError(
            "Successful representative must preserve its raw response"
        )
    parsed, reparsed_error = parse_representative_response(
        raw_response,
        task_description=task_description,
    )
    if parsed is None or reparsed_error is not None:
        raise ValueError(
            "Successful representative raw response no longer passes schema: "
            f"{reparsed_error}"
        )
    expected_fields = {
        "phase": parsed["phase"],
        "raw_phase": parsed["raw_phase"],
        "phrase": parsed["phrase"],
        "visibility": parsed["visibility"],
    }
    mismatched = [
        field
        for field, expected in expected_fields.items()
        if annotation.get(field) != expected
    ]
    if mismatched:
        raise ValueError(
            "Successful representative fields disagree with raw response: "
            f"{mismatched}"
        )
