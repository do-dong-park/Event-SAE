"""Build per-event vision embedding + state vector for each sample.

Reads `samples.jsonl` from `event_sae.events.extract_media`, encodes the
selected frames through a frozen vision encoder (default SigLIP base), and
joins per-waypoint end-effector position + (optional) gripper action.

Output: `event_features.jsonl` (one record per sample) ready for
`event_sae.events.cluster`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl, write_jsonl


@dataclass
class EpisodeStateSummary:
    num_steps: int
    center_records: dict[int, dict]
    state_feature_names: list[str]


def build_qpos_episode_state_index(
    samples: list[dict],
    *,
    state_position_frame: str,
) -> dict[tuple[str, int], EpisodeStateSummary]:
    """Build exact center states using full-task qpos normalization."""
    from event_sae.keyframes.extract import load_episode_trajectories

    requested: dict[str, dict[int, dict]] = defaultdict(dict)
    for sample in samples:
        trajectory_path = str(
            Path(sample["source_trajectory_records_path"]).resolve()
        )
        episode_num = int(sample["episode_num"])
        task_description = str(sample["task_description"])
        request = requested[trajectory_path].setdefault(
            episode_num,
            {"task_description": task_description, "steps": set()},
        )
        if request["task_description"] != task_description:
            raise ValueError(
                f"episode {episode_num} has conflicting task descriptions"
            )
        request["steps"].add(int(sample["waypoint_step"]))

    output: dict[tuple[str, int], EpisodeStateSummary] = {}
    for trajectory_path, episode_requests in requested.items():
        requested_tasks = {
            str(request["task_description"])
            for request in episode_requests.values()
        }
        try:
            episodes = load_episode_trajectories(
                Path(trajectory_path),
                eef_position_frame=state_position_frame,
            )
        except ValueError as error:
            # Preserve the event-feature builder's established missing-field
            # contract while delegating trajectory parsing to keyframes.
            if "requires eef_pos_abs" in str(error):
                raise KeyError(
                    "qpos_aperture_delta state requires eef_pos_abs"
                ) from error
            raise

        episodes_by_num = {}
        for episode in episodes:
            if episode.task_description not in requested_tasks:
                continue
            episode_num = int(episode.episode_num)
            if episode_num in episodes_by_num:
                raise ValueError(
                    f"episode {episode_num} spans multiple task identities"
                )
            if episode.step_indices != list(range(len(episode.step_indices))):
                raise ValueError(
                    f"episode {episode_num} has non-contiguous step indices"
                )
            if episode.gripper_state is None:
                raise KeyError(
                    "qpos_aperture_delta state requires gripper_qpos in "
                    f"every source-task record; missing episode={episode_num}"
                )
            episodes_by_num[episode_num] = episode

        missing_episodes = sorted(set(episode_requests).difference(episodes_by_num))
        if missing_episodes:
            raise ValueError(
                f"Missing selected episodes in {trajectory_path}: {missing_episodes[:10]}"
            )

        position_names = [
            f"eef_pos_{state_position_frame}_{axis}" for axis in ("x", "y", "z")
        ]
        state_feature_names = [
            *position_names,
            "gripper_aperture_normalized",
            "gripper_aperture_delta",
        ]
        for episode_num, request in episode_requests.items():
            episode = episodes_by_num[episode_num]
            expected_task = str(request["task_description"])
            if episode.task_description != expected_task:
                raise ValueError(
                    f"episode {episode_num} task mismatch: "
                    f"{episode.task_description!r} != "
                    f"{expected_task!r}"
                )

            gripper_state = episode.gripper_state
            if gripper_state is None:  # guarded while indexing; narrows the type
                raise RuntimeError(
                    f"episode {episode_num} gripper state was not initialized"
                )
            center_records = {}
            for step in sorted(request["steps"]):
                if not 0 <= step < len(episode.positions):
                    raise ValueError(
                        f"Missing center state for episode {episode_num}, step {step}"
                    )
                position = np.asarray(episode.positions[step], dtype=np.float64)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(
                        f"episode {episode_num}, step {step}: invalid "
                        f"eef_pos_{state_position_frame}"
                    )
                center_records[step] = {
                    "eef_pos": position.astype(np.float32).tolist(),
                    "gripper_aperture": float(gripper_state.aperture[step]),
                    "gripper_aperture_normalized": float(
                        gripper_state.normalized_aperture[step]
                    ),
                    "gripper_aperture_delta": float(
                        gripper_state.aperture_delta[step]
                    ),
                    "gripper_normalization_min": float(
                        gripper_state.normalization_min
                    ),
                    "gripper_normalization_max": float(
                        gripper_state.normalization_max
                    ),
                }
            output[(trajectory_path, episode_num)] = EpisodeStateSummary(
                num_steps=len(episode.step_indices),
                center_records=center_records,
                state_feature_names=state_feature_names,
            )
    return output


@dataclass(frozen=True)
class EventFeatureArtifactSpec:
    """Serialization contract for an event-feature artifact family."""

    manifest_format: str = "event_sae_event_features_v3"
    record_source_format: str | None = None
    include_trajectory_record_sources: bool = True
    sort_manifest_keys: bool = False
    manifest_trailing_newline: bool = False


@dataclass(frozen=True)
class VisionEmbeddingResult:
    """One embedding plus provider-specific, per-record provenance."""

    embedding: np.ndarray
    provenance: dict[str, object] = field(default_factory=dict)


class VisionEmbeddingProvider(Protocol):
    """Extension point for reusing or otherwise supplying vision embeddings."""

    def encode(
        self,
        *,
        sample: dict,
        selected_frame_paths: list[str],
    ) -> VisionEmbeddingResult:
        ...

    def finalize(self) -> dict[str, object]:
        """Validate provider-wide invariants and return manifest provenance."""
        ...


def _l2_normalize(
    vec: np.ndarray,
    *,
    context: str = "event feature",
) -> np.ndarray:
    if vec.ndim != 1 or vec.size == 0 or not np.isfinite(vec).all():
        raise ValueError(f"{context} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"{context} has zero norm")
    return vec / norm


class VisionEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        revision: str | None = None,
    ) -> None:
        model_kwargs = {"revision": revision} if revision is not None else {}
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **model_kwargs)
        self.model = (
            AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            .eval()
            .to(device)
        )
        self.device = torch.device(device)

    @torch.no_grad()
    def encode(self, frame_paths: list[str]) -> np.ndarray:
        images = []
        for path in frame_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        if hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(**inputs)
        else:
            outputs = self.model(**inputs)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                feats = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feats = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError(
                    "Could not derive image features from the selected vision model outputs."
                )
        mean_feat = feats.float().mean(dim=0).cpu().numpy()
        return _l2_normalize(mean_feat)


class ExactReuseVisionEmbeddingProvider:
    """Reuse exactly matching feature rows and lazily compute all other rows.

    Reuse is accepted only when the model identity, revision, selected frame
    positions, and resolved frame paths match. Every row in the reusable artifact
    must be consumed exactly once before the output can be written.
    """

    def __init__(
        self,
        *,
        reusable_features_path: Path,
        vision_model_name_or_path: str,
        vision_model_revision: str | None,
        frame_positions: Sequence[int],
        device: str,
        embedder_factory: Callable[..., VisionEmbedder] = VisionEmbedder,
        key_fields: tuple[str, ...] = ("episode_num", "waypoint_step"),
    ) -> None:
        self.reusable_features_path = Path(reusable_features_path).resolve()
        self.vision_model_name_or_path = vision_model_name_or_path
        self.vision_model_revision = vision_model_revision
        self.frame_positions = list(frame_positions)
        self.device = device
        self._embedder_factory = embedder_factory
        self._key_fields = key_fields
        self._embedder: VisionEmbedder | None = None
        self._reusable = self._index_reusable_rows()
        self._used_reusable_keys: set[tuple[object, ...]] = set()
        self._num_computed = 0

    def _row_key(self, record: dict) -> tuple[object, ...]:
        return tuple(record[field] for field in self._key_fields)

    def _index_reusable_rows(self) -> dict[tuple[object, ...], dict]:
        indexed: dict[tuple[object, ...], dict] = {}
        for record in load_jsonl(self.reusable_features_path):
            key = self._row_key(record)
            if key in indexed:
                raise ValueError(f"Duplicate reusable vision key: {key}")
            indexed[key] = record
        return indexed

    def _validated_reusable_embedding(
        self,
        *,
        record: dict,
        sample: dict,
        selected_frame_paths: list[str],
    ) -> np.ndarray:
        sample_id = sample["sample_id"]
        if str(record["vision_model_name_or_path"]) != self.vision_model_name_or_path:
            raise ValueError(f"{sample_id}: reusable vision model mismatch")
        if record.get("vision_model_revision") != self.vision_model_revision:
            raise ValueError(f"{sample_id}: reusable vision revision mismatch")
        positions = [int(value) for value in record["vision_frame_positions"]]
        if positions != self.frame_positions:
            raise ValueError(f"{sample_id}: reusable vision frame positions mismatch")
        reusable_paths = [
            str(Path(value).resolve()) for value in record["selected_frame_paths"]
        ]
        if reusable_paths != selected_frame_paths:
            raise ValueError(
                f"{sample_id}: reusable vision frames do not exactly match"
            )
        embedding = np.asarray(record["vision_embedding"], dtype=np.float32)
        if (
            embedding.ndim != 1
            or embedding.size == 0
            or not np.isfinite(embedding).all()
        ):
            raise ValueError(f"{sample_id}: invalid reusable vision embedding")
        return embedding

    def encode(
        self,
        *,
        sample: dict,
        selected_frame_paths: list[str],
    ) -> VisionEmbeddingResult:
        key = self._row_key(sample)
        reusable_record = self._reusable.get(key)
        if reusable_record is None:
            if self._embedder is None:
                self._embedder = self._embedder_factory(
                    self.vision_model_name_or_path,
                    self.device,
                    self.vision_model_revision,
                )
            self._num_computed += 1
            return VisionEmbeddingResult(
                embedding=self._embedder.encode(selected_frame_paths),
                provenance={
                    "vision_embedding_source": "computed",
                    "reused_vision_sample_id": None,
                },
            )

        if key in self._used_reusable_keys:
            raise ValueError(f"Reusable vision key matched multiple samples: {key}")
        embedding = self._validated_reusable_embedding(
            record=reusable_record,
            sample=sample,
            selected_frame_paths=selected_frame_paths,
        )
        self._used_reusable_keys.add(key)
        return VisionEmbeddingResult(
            embedding=embedding,
            provenance={
                "vision_embedding_source": "reused",
                "reused_vision_sample_id": str(reusable_record["sample_id"]),
            },
        )

    def finalize(self) -> dict[str, object]:
        if len(self._used_reusable_keys) != len(self._reusable):
            raise ValueError(
                "Reusable vision features do not exactly match current samples: "
                f"used={len(self._used_reusable_keys)}, "
                f"available={len(self._reusable)}"
            )
        return {
            "reusable_vision_features_path": str(self.reusable_features_path),
            "reusable_vision_features_sha256": _sha256(
                self.reusable_features_path
            ),
            "num_reused_vision_embeddings": len(self._used_reusable_keys),
            "num_computed_vision_embeddings": self._num_computed,
        }


def normalize_media_samples(
    samples_path: Path,
    *,
    trajectory_records_path: Path | None = None,
) -> list[dict]:
    """Normalize legacy and temporal_vla v4 media records for feature building."""
    samples_path = Path(samples_path).resolve()
    bundle_dir = samples_path.parent
    normalized: list[dict] = []
    for sample in load_jsonl(samples_path):
        record = dict(sample)
        if sample.get("format") == "event_sae_stage3_media_v4":
            frames = sorted(sample["frames"], key=lambda frame: int(frame["position"]))
            positions = [int(frame["position"]) for frame in frames]
            if positions != list(range(len(frames))):
                raise ValueError(
                    f"Non-contiguous frame positions for sample_id={sample['sample_id']}: "
                    f"{positions}"
                )
            record["frame_paths"] = [
                str((bundle_dir / frame["path"]).resolve()) for frame in frames
            ]
            record["waypoint_step"] = int(sample["waypoint_index"])
            record["clip_path"] = str(sample.get("source_video_relative_path", ""))

        if "frame_paths" not in record or "waypoint_step" not in record:
            raise ValueError(
                f"Unsupported media sample schema for sample_id={sample.get('sample_id')!r}"
            )
        record["frame_paths"] = [
            str((Path(path) if Path(path).is_absolute() else bundle_dir / path).resolve())
            for path in record["frame_paths"]
        ]
        missing_frames = [path for path in record["frame_paths"] if not Path(path).is_file()]
        if missing_frames:
            raise FileNotFoundError(
                f"Missing frame files for sample_id={sample['sample_id']}: {missing_frames[:3]}"
            )

        source_records = trajectory_records_path or record.get("source_trajectory_records_path")
        if source_records is None:
            raise ValueError(
                "trajectory_records_path is required for media records that do not embed "
                "source_trajectory_records_path"
            )
        source_records = Path(source_records).resolve()
        if not source_records.is_file():
            raise FileNotFoundError(f"trajectory records not found: {source_records}")
        record["source_trajectory_records_path"] = str(source_records)
        normalized.append(record)
    return normalized


def audit_event_feature_records(
    records: list[dict],
    *,
    expected_samples: int | None = None,
) -> dict:
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} event features, found {len(records)}")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id in event features")
    if not records:
        raise ValueError("No event features were produced")

    vision_dims: set[int] = set()
    state_dims: set[int] = set()
    frame_counts: set[int] = set()
    for record in records:
        vision = np.asarray(record["vision_embedding"], dtype=np.float32)
        state = np.asarray(record["state_vector"], dtype=np.float32)
        progress = float(record["progress_percent"])
        if (
            not np.isfinite(vision).all()
            or not np.isfinite(state).all()
            or not np.isfinite(progress)
        ):
            raise ValueError(f"Non-finite feature for sample_id={record['sample_id']}")
        if not 0.0 <= progress <= 1.0:
            raise ValueError(
                f"Out-of-range progress for sample_id={record['sample_id']}: {progress}"
            )
        vision_dims.add(int(vision.size))
        state_dims.add(int(state.size))
        frame_counts.add(len(record["selected_frame_paths"]))
    if len(vision_dims) != 1 or len(state_dims) != 1 or len(frame_counts) != 1:
        raise ValueError(
            f"Inconsistent feature schema: vision={vision_dims}, state={state_dims}, "
            f"frames={frame_counts}"
        )
    return {
        "num_samples": len(records),
        "num_tasks": len({str(record["task_description"]) for record in records}),
        "vision_dimension": next(iter(vision_dims)),
        "state_dimension": next(iter(state_dims)),
        "frames_per_sample": next(iter(frame_counts)),
        "passed": True,
    }


def build_episode_state_index(
    samples: list[dict],
    *,
    state_position_frame: str = "rel",
    gripper_state_mode: str = "legacy",
) -> dict[tuple[str, int], EpisodeStateSummary]:
    """Collect per-waypoint center states from every referenced trajectory."""
    if gripper_state_mode == "qpos_aperture_delta":
        return build_qpos_episode_state_index(
            samples,
            state_position_frame=state_position_frame,
        )
    if gripper_state_mode != "legacy":
        raise ValueError(f"Unsupported gripper_state_mode={gripper_state_mode!r}")
    if state_position_frame != "rel":
        raise ValueError("legacy gripper state mode only supports relative EEF position")
    needed_steps: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for sample in samples:
        trajectory_path = str(Path(sample["source_trajectory_records_path"]).resolve())
        needed_steps[trajectory_path][int(sample["episode_num"])].add(int(sample["waypoint_step"]))

    index: dict[tuple[str, int], EpisodeStateSummary] = {}
    for trajectory_path, episode_steps in needed_steps.items():
        center_records: dict[int, dict[int, dict]] = {
            episode_num: {} for episode_num in episode_steps
        }
        step_counts: dict[int, int] = defaultdict(int)
        gripper_action_presence: set[bool] = set()
        with Path(trajectory_path).open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                episode_num = int(record["episode_num"])
                if episode_num not in episode_steps:
                    continue
                step = int(record["step_in_episode"])
                step_counts[episode_num] += 1
                if step in episode_steps[episode_num]:
                    if "eef_pos" not in record:
                        raise KeyError(
                            f"Missing eef_pos for episode {episode_num}, "
                            f"step {step} in {trajectory_path}"
                        )
                    has_gripper_action = "gripper_action" in record
                    gripper_action_presence.add(has_gripper_action)
                    center_record = {"eef_pos": [float(x) for x in record["eef_pos"]]}
                    if has_gripper_action:
                        center_record["gripper_action"] = float(record["gripper_action"])
                    center_records[episode_num][step] = center_record

        if len(gripper_action_presence) > 1:
            raise ValueError(
                "Inconsistent gripper_action availability across selected keyframe steps "
                f"in {trajectory_path}"
            )
        state_feature_names = ["eef_pos_x", "eef_pos_y", "eef_pos_z"]
        if gripper_action_presence == {True}:
            state_feature_names.append("gripper_action")

        for episode_num, step_map in episode_steps.items():
            missing_steps = sorted(step_map.difference(center_records[episode_num]))
            if missing_steps:
                raise ValueError(
                    f"Missing center states for episode {episode_num} steps "
                    f"{missing_steps[:10]} in {trajectory_path}"
                )
            index[(trajectory_path, episode_num)] = EpisodeStateSummary(
                num_steps=int(step_counts[episode_num]),
                center_records=center_records[episode_num],
                state_feature_names=state_feature_names,
            )
    return index


def state_vector_from_record(center_state: dict, state_feature_names: list[str]) -> np.ndarray:
    values = [
        center_state["eef_pos"][0],
        center_state["eef_pos"][1],
        center_state["eef_pos"][2],
    ]
    if "gripper_action" in state_feature_names:
        values.append(center_state["gripper_action"])
    if "gripper_aperture_normalized" in state_feature_names:
        values.append(center_state["gripper_aperture_normalized"])
    if "gripper_aperture_delta" in state_feature_names:
        values.append(center_state["gripper_aperture_delta"])
    return np.asarray(values, dtype=np.float32)


def build_event_features(
    samples_path: Path,
    output_path: Path,
    vision_model_name_or_path: str = "google/siglip-base-patch16-224",
    vision_model_revision: str | None = None,
    device: str | None = None,
    frame_positions: Sequence[int] = (0, 1, 2, 3, 4),
    trajectory_records_path: Path | None = None,
    expected_samples: int | None = None,
    state_position_frame: str = "rel",
    gripper_state_mode: str = "legacy",
    vision_embedding_provider: VisionEmbeddingProvider | None = None,
    artifact_spec: EventFeatureArtifactSpec | None = None,
) -> dict:
    """Build event features with a computed or caller-supplied vision provider.

    The default path retains the paper implementation's frozen-encoder behavior.
    Extensions can supply a provider that reuses existing embeddings while this
    function continues to own media normalization, physical-state joins, record
    construction, auditing, and artifact writes.
    """
    samples_path = Path(samples_path).resolve()
    if not samples_path.is_file():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    artifact_spec = artifact_spec or EventFeatureArtifactSpec()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite event features: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite event feature manifest: {manifest_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = normalize_media_samples(
        samples_path,
        trajectory_records_path=trajectory_records_path,
    )
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f"Expected {expected_samples} media samples, found {len(samples)}")
    if len(frame_positions) != len(set(frame_positions)):
        raise ValueError(f"Duplicate frame positions requested: {frame_positions}")
    for sample in samples:
        invalid = [
            position
            for position in frame_positions
            if not 0 <= position < len(sample["frame_paths"])
        ]
        if invalid:
            raise ValueError(
                f"Invalid frame positions for sample_id={sample['sample_id']}: {invalid}"
            )

    state_index = build_episode_state_index(
        samples,
        state_position_frame=state_position_frame,
        gripper_state_mode=gripper_state_mode,
    )
    embedder = (
        VisionEmbedder(
            vision_model_name_or_path,
            device,
            vision_model_revision,
        )
        if vision_embedding_provider is None
        else None
    )

    records: list[dict] = []
    for idx, sample in enumerate(samples, start=1):
        frame_paths = sample["frame_paths"]
        selected_frame_paths = [frame_paths[pos] for pos in frame_positions]
        trajectory_path = str(Path(sample["source_trajectory_records_path"]).resolve())
        episode_num = int(sample["episode_num"])
        waypoint_step = int(sample["waypoint_step"])
        state_summary = state_index[(trajectory_path, episode_num)]
        center_state = state_summary.center_records[waypoint_step]
        progress_percent = waypoint_step / max(state_summary.num_steps - 1, 1)
        if vision_embedding_provider is None:
            if embedder is None:
                raise RuntimeError("Vision embedder was not initialized")
            embedding_result = VisionEmbeddingResult(
                embedding=embedder.encode(selected_frame_paths)
            )
        else:
            embedding_result = vision_embedding_provider.encode(
                sample=sample,
                selected_frame_paths=selected_frame_paths,
            )
        vision_embedding = np.asarray(
            embedding_result.embedding,
            dtype=np.float32,
        )
        state_vector = state_vector_from_record(
            center_state=center_state,
            state_feature_names=state_summary.state_feature_names,
        )
        record = {
            "sample_id": sample["sample_id"],
            "source_format": (
                artifact_spec.record_source_format
                if artifact_spec.record_source_format is not None
                else sample.get("format")
            ),
            "task_id": int(sample["task_id"]),
            "task_description": sample["task_description"],
            "prompt_task_description": sample["prompt_task_description"],
            "episode_num": episode_num,
            "task_episode_idx": int(sample["task_episode_idx"]),
            "cell_id": sample.get("cell_id"),
            "success": sample.get("success"),
            "waypoint_rank": int(sample["waypoint_rank"]),
            "waypoint_step": waypoint_step,
            "clip_path": sample.get("clip_path", ""),
            "frame_paths": frame_paths,
            "selected_frame_paths": selected_frame_paths,
            "source_trajectory_records_path": trajectory_path,
            "vision_model_name_or_path": vision_model_name_or_path,
            "vision_model_revision": vision_model_revision,
            "vision_frame_positions": list(frame_positions),
            "vision_embedding": vision_embedding.astype(np.float32).tolist(),
        }
        record_tail = {
            "state_vector": state_vector.tolist(),
            "state_feature_names": state_summary.state_feature_names,
            "state_position_frame": state_position_frame,
            "gripper_state_mode": gripper_state_mode,
            "gripper_state": (
                None
                if "gripper_aperture_normalized" not in center_state
                else {
                    "aperture": center_state["gripper_aperture"],
                    "normalized_aperture": center_state[
                        "gripper_aperture_normalized"
                    ],
                    "aperture_delta": center_state["gripper_aperture_delta"],
                    "normalization_min": center_state["gripper_normalization_min"],
                    "normalization_max": center_state["gripper_normalization_max"],
                }
            ),
            "anchor_source": sample.get("anchor_source"),
            "progress_percent": float(progress_percent),
            "num_steps": int(state_summary.num_steps),
            "boundary_shift_category": sample.get("boundary_shift_category"),
            "anchor_env_step_error": sample.get("anchor_env_step_error"),
        }
        provenance_collisions = set(embedding_result.provenance).intersection(
            record.keys() | record_tail.keys()
        )
        if provenance_collisions:
            raise ValueError(
                f"{sample['sample_id']}: vision provider provenance collides with "
                f"core fields: {sorted(provenance_collisions)}"
            )
        record.update(embedding_result.provenance)
        record.update(record_tail)
        records.append(record)
        print(
            f"[{idx}/{len(samples)}] sample_id={sample['sample_id']} "
            f"task={sample['task_description']} progress={progress_percent:.3f}"
        )

    audit = audit_event_feature_records(records, expected_samples=expected_samples)
    provider_manifest = (
        vision_embedding_provider.finalize()
        if vision_embedding_provider is not None
        else {}
    )
    reserved_manifest_fields = {
        "format",
        "samples_path",
        "samples_sha256",
        "trajectory_record_sources",
        "trajectory_records_path",
        "trajectory_records_sha256",
        "output_path",
        "output_sha256",
        "vision_model_name_or_path",
        "vision_model_revision",
        "vision_frame_positions",
        "state_position_frame",
        "gripper_state_mode",
        "device",
        *audit,
    }
    manifest_collisions = set(provider_manifest).intersection(
        reserved_manifest_fields
    )
    if manifest_collisions:
        raise ValueError(
            "Vision provider manifest provenance collides with core fields: "
            f"{sorted(manifest_collisions)}"
        )
    write_jsonl(output_path, records)
    source_records_paths = sorted(
        {
            Path(record["source_trajectory_records_path"]).resolve()
            for record in records
        }
    )
    trajectory_record_sources = [
        {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        }
        for source_path in source_records_paths
    ]
    single_source = (
        trajectory_record_sources[0]
        if len(trajectory_record_sources) == 1
        else None
    )
    manifest = {
        "format": artifact_spec.manifest_format,
        "samples_path": str(samples_path),
        "samples_sha256": _sha256(samples_path),
    }
    if artifact_spec.include_trajectory_record_sources:
        # Multi-source builds retain full provenance while existing single-source
        # consumers continue to use the singular keys below.
        manifest["trajectory_record_sources"] = trajectory_record_sources
    manifest_tail = {
        "trajectory_records_path": (
            single_source["path"] if single_source is not None else None
        ),
        "trajectory_records_sha256": (
            single_source["sha256"] if single_source is not None else None
        ),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "vision_model_name_or_path": vision_model_name_or_path,
        "vision_model_revision": vision_model_revision,
        "vision_frame_positions": list(frame_positions),
        "state_position_frame": state_position_frame,
        "gripper_state_mode": gripper_state_mode,
        "device": str(device),
        **audit,
    }
    manifest.update(provider_manifest)
    manifest.update(manifest_tail)
    manifest_text = json.dumps(
        manifest,
        indent=2,
        sort_keys=artifact_spec.sort_manifest_keys,
    )
    if artifact_spec.manifest_trailing_newline:
        manifest_text += "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    print(f"Samples path: {samples_path}")
    print(f"Saved event features to: {output_path}")
    print(f"Saved event feature manifest to: {manifest_path}")
    return manifest


EQUAL_VIEW_CONCAT_FUSION_ID = "equal_view_l2_concat_v1"
MULTIVIEW_FEATURE_VIEW_ORDER = ("left", "right", "wrist")
MULTIVIEW_FEATURE_ALIGNMENT_FIELDS = (
    "sample_id",
    "task_id",
    "task_description",
    "prompt_task_description",
    "episode_num",
    "task_episode_idx",
    "cell_id",
    "success",
    "anchor_source",
    "waypoint_rank",
    "waypoint_step",
    "clip_path",
    "source_trajectory_records_path",
    "state_vector",
    "state_feature_names",
    "progress_percent",
    "num_steps",
    "boundary_shift_category",
    "anchor_env_step_error",
    "vision_model_name_or_path",
    "vision_model_revision",
    "vision_frame_positions",
)


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_multiview_feature_index(
    path: Path,
    *,
    view: str,
) -> tuple[list[str], dict[str, dict]]:
    records = load_jsonl(path)
    sample_ids = [str(record["sample_id"]) for record in records]
    if not records or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            f"{view} event features must be non-empty with unique sample_id values"
        )
    return sample_ids, {
        str(record["sample_id"]): record for record in records
    }


def _verify_multiview_feature_alignment(
    *,
    sample_id: str,
    records: dict[str, dict],
) -> None:
    for field in MULTIVIEW_FEATURE_ALIGNMENT_FIELDS:
        values = {view: record.get(field) for view, record in records.items()}
        if len({_stable_value(value) for value in values.values()}) != 1:
            raise ValueError(
                f"{sample_id}: aligned field {field!r} differs across views: {values}"
            )

    selected_counts = {
        view: len(record.get("selected_frame_paths", []))
        for view, record in records.items()
    }
    if (
        len(set(selected_counts.values())) != 1
        or next(iter(selected_counts.values())) == 0
    ):
        raise ValueError(
            f"{sample_id}: selected frame counts differ across views: {selected_counts}"
        )


def combine_multiview_event_features(
    *,
    left_features_path: Path,
    right_features_path: Path,
    wrist_features_path: Path,
    output_path: Path,
    expected_samples: int | None = None,
) -> dict:
    """Create one equal-weight multiview embedding per aligned event sample."""
    input_paths = {
        "left": Path(left_features_path).resolve(),
        "right": Path(right_features_path).resolve(),
        "wrist": Path(wrist_features_path).resolve(),
    }
    for view, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{view} event features not found: {path}")

    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite multiview features: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite multiview feature manifest: {manifest_path}"
        )

    order_by_view: dict[str, list[str]] = {}
    index_by_view: dict[str, dict[str, dict]] = {}
    for view in MULTIVIEW_FEATURE_VIEW_ORDER:
        order, index = _load_multiview_feature_index(
            input_paths[view],
            view=view,
        )
        order_by_view[view] = order
        index_by_view[view] = index

    reference_ids = order_by_view["left"]
    reference_set = set(reference_ids)
    for view in MULTIVIEW_FEATURE_VIEW_ORDER[1:]:
        candidate_set = set(order_by_view[view])
        if candidate_set != reference_set:
            missing = sorted(reference_set - candidate_set)
            extra = sorted(candidate_set - reference_set)
            raise ValueError(
                f"{view} sample coverage differs from left: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    if expected_samples is not None and len(reference_ids) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} aligned samples, found {len(reference_ids)}"
        )

    output_records: list[dict] = []
    view_dimensions: dict[str, int] | None = None
    for sample_id in reference_ids:
        view_records = {
            view: index_by_view[view][sample_id]
            for view in MULTIVIEW_FEATURE_VIEW_ORDER
        }
        _verify_multiview_feature_alignment(
            sample_id=sample_id,
            records=view_records,
        )

        normalized_embeddings = {
            view: _l2_normalize(
                np.asarray(record["vision_embedding"], dtype=np.float32),
                context=f"{sample_id}: {view} vision embedding",
            )
            for view, record in view_records.items()
        }
        sample_dimensions = {
            view: int(embedding.size)
            for view, embedding in normalized_embeddings.items()
        }
        if len(set(sample_dimensions.values())) != 1:
            raise ValueError(
                f"{sample_id}: view embedding dimensions differ: {sample_dimensions}"
            )
        if view_dimensions is None:
            view_dimensions = sample_dimensions
        elif sample_dimensions != view_dimensions:
            raise ValueError(
                f"{sample_id}: embedding dimensions changed: "
                f"{sample_dimensions} != {view_dimensions}"
            )

        fused = np.concatenate(
            [
                normalized_embeddings[view]
                for view in MULTIVIEW_FEATURE_VIEW_ORDER
            ]
        )
        fused = _l2_normalize(
            fused,
            context=f"{sample_id}: fused vision embedding",
        )

        left_record = dict(view_records["left"])
        left_record["vision_embedding"] = fused.astype(np.float32).tolist()
        left_record["vision_fusion"] = EQUAL_VIEW_CONCAT_FUSION_ID
        left_record["vision_view_order"] = list(MULTIVIEW_FEATURE_VIEW_ORDER)
        left_record["vision_view_dimensions"] = sample_dimensions
        left_record["view_frame_paths"] = {
            view: list(view_records[view]["frame_paths"])
            for view in MULTIVIEW_FEATURE_VIEW_ORDER
        }
        left_record["selected_view_frame_paths"] = {
            view: list(view_records[view]["selected_frame_paths"])
            for view in MULTIVIEW_FEATURE_VIEW_ORDER
        }
        output_records.append(left_record)

    audit = audit_event_feature_records(
        output_records,
        expected_samples=expected_samples,
    )
    write_jsonl(output_path, output_records)
    manifest = {
        "format": "event_sae_multiview_event_features_v1",
        "fusion": EQUAL_VIEW_CONCAT_FUSION_ID,
        "view_order": list(MULTIVIEW_FEATURE_VIEW_ORDER),
        "view_dimensions": view_dimensions,
        "inputs": {
            view: {
                "path": str(input_paths[view]),
                "sha256": _sha256(input_paths[view]),
            }
            for view in MULTIVIEW_FEATURE_VIEW_ORDER
        },
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        **audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
