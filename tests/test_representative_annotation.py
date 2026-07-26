import json

from PIL import Image
import pytest

from event_sae.events.representative_annotation import (
    AnnotationImagePart,
    REQUEST_EXPOSURE_POLICY,
    SEPARATE_MULTIVIEW_LAYOUT,
    build_representative_request_parts,
    build_representative_response_schema,
    hash_annotation_media,
    hash_representative_set,
    parse_representative_response,
    read_clean_image_bytes,
    validate_representative_media,
)
from event_sae.events.prompts import (
    ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID,
    SINGLE_VIEW_LEFT_LAYOUT,
    build_representative_clip_annotation_prompt,
    validate_clean_visual_annotation_prompt,
)


PICK_TASK = "Pick the beer from the counter and place it in the cabinet."
DRAWER_TASK = "Open the left drawer."


def _pick_response(phase="grasp", visibility="clear"):
    return {
        "phrase": "initiating a grasp around the target",
        "phase": phase,
        "visibility": visibility,
    }


def test_representative_prompt_defines_grasp_as_pre_hold_attempt():
    prompt = build_representative_clip_annotation_prompt(
        task_description=PICK_TASK,
        cluster_id="cluster_00",
        representative_index=2,
        num_frames=5,
    )

    assert ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID.endswith(
        "_v12r3_clean_visual"
    )
    assert "first contact, and an acquisition attempt" in prompt
    assert "secure closure is not required" in prompt
    assert "object is not visibly held and the gripper is outside" in prompt
    assert "object is not yet visibly established as held" in prompt
    assert "If the target is not inside but is visibly held" in prompt
    assert "target is not held but a distractor is visibly held" in prompt
    assert "T3 is the temporal center" in prompt
    assert "ONE representative" in prompt
    assert "shared phase" not in prompt
    assert "cluster_00" not in prompt
    assert "Representative index" not in prompt
    assert '"phase": "exactly one label from the closed vocabulary, or unresolved"' in prompt
    for forbidden in (
        "oracle",
        "predicate",
        "threshold",
        "waypoint",
        "success",
        "latched",
        "target_hold_established_before_center",
    ):
        assert forbidden not in prompt.lower()


def test_representative_prompt_uses_only_visible_drawer_disengage_history():
    prompt = build_representative_clip_annotation_prompt(
        task_description=DRAWER_TASK,
        cluster_id="cluster_00",
        representative_index=1,
        num_frames=5,
    )

    assert "supplied frames visibly show earlier handle proximity" in prompt
    assert "gripper is not near the instructed handle at the temporal center" in prompt
    assert "Centered motion means displacement spanning T3" in prompt
    assert "open-done overrides all other evidence" in prompt
    assert "For labels other than wrong-grasp" in prompt
    assert "return unresolved with insufficient visibility" in prompt
    assert "latched" not in prompt.lower()
    assert "current-state priority" not in prompt.lower()
    assert "prior_handle_engagement_visible" not in prompt
    assert "grasping the target" not in prompt
    assert "Track the instruction-named drawer" in prompt


def test_representative_clean_request_policy_denies_oracle_and_source_metadata():
    assert REQUEST_EXPOSURE_POLICY["task_instruction"] is True
    assert REQUEST_EXPOSURE_POLICY["representative_image_bytes"] is True
    for field in (
        "cluster_id",
        "episode_coverage",
        "sample_or_episode_id",
        "representative_selection_metadata",
        "source_path_or_filename",
        "source_step_or_progress",
        "awe_anchor_metadata",
        "oracle_phase_or_timeline",
        "simulator_state_or_predicates",
        "success_or_failure",
        "phase_taxonomy_provenance",
    ):
        assert REQUEST_EXPOSURE_POLICY[field] is False

    prompt = build_representative_clip_annotation_prompt(
        task_description=PICK_TASK,
        cluster_id="cluster_00",
        representative_index=2,
        num_frames=5,
    )
    validate_clean_visual_annotation_prompt(
        prompt,
        forbidden_identifiers=("cluster_00", "ep0001_wp000_r0001"),
    )
    with pytest.raises(ValueError, match="forbidden request input"):
        validate_clean_visual_annotation_prompt(prompt + "\nwaypoint")


def test_representative_schema_and_parser_use_minimal_clean_visual_contract():
    schema = build_representative_response_schema(PICK_TASK)
    assert schema["required"] == ["phrase", "phase", "visibility"]
    assert schema["additionalProperties"] is False
    assert "grasp" in schema["properties"]["phase"]["enum"]
    assert "unresolved" in schema["properties"]["phase"]["enum"]
    assert "evidence" not in schema["properties"]

    parsed, error = parse_representative_response(
        json.dumps(_pick_response()),
        task_description=PICK_TASK,
    )
    assert error is None
    assert parsed["phase"] == "grasp"

    broken = _pick_response()
    broken["evidence"] = {"target_held_at_center": "yes"}
    _, error = parse_representative_response(
        json.dumps(broken),
        task_description=PICK_TASK,
    )
    assert "unexpected_keys:['evidence']" in error

    unresolved, error = parse_representative_response(
        json.dumps(_pick_response("unresolved", "insufficient")),
        task_description=PICK_TASK,
    )
    assert error is None
    assert unresolved["raw_phase"] == "unresolved"
    assert unresolved["phase"] is None

    _, error = parse_representative_response(
        json.dumps(_pick_response("grasp", "insufficient")),
        task_description=PICK_TASK,
    )
    assert "insufficient_visibility_requires_unresolved_phase" in error


def test_representative_request_has_fifteen_ordered_multiview_images(
    tmp_path,
):
    source_frames = []
    for timestamp in range(5):
        frame = {}
        for view in ("left", "right", "wrist"):
            path = tmp_path / f"{timestamp}_{view}.jpg"
            Image.new(
                "RGB",
                (8, 8),
                color=(timestamp * 20, len(view) * 20, 0),
            ).save(path, format="JPEG")
            frame[view] = str(path)
        source_frames.append(frame)

    parts = build_representative_request_parts(
        prompt="prompt",
        source_view_frames=source_frames,
    )

    image_parts = [
        part for part in parts if isinstance(part, AnnotationImagePart)
    ]
    text_parts = [
        part for part in parts if isinstance(part, str)
    ]
    assert len(image_parts) == 15
    assert len(text_parts) == 26
    assert all(
        part.mime_type == "image/jpeg"
        for part in image_parts
    )
    assert all(part.data for part in image_parts)
    assert str(tmp_path) not in "\n".join(text_parts)
    assert any("BEGIN T1;" in text for text in text_parts)
    assert any("T3 (CENTER)" in text for text in text_parts)


def test_representative_single_view_sends_only_five_left_images(tmp_path):
    source_frames = []
    for timestamp in range(5):
        path = tmp_path / f"{timestamp}_left.jpg"
        Image.new(
            "RGB",
            (8, 8),
            color=(timestamp * 20, 0, 0),
        ).save(path, format="JPEG")
        source_frames.append({"left": str(path)})

    parts = build_representative_request_parts(
        prompt="prompt",
        source_view_frames=source_frames,
        view_order=("left",),
    )

    image_parts = [
        part for part in parts if isinstance(part, AnnotationImagePart)
    ]
    text_parts = [
        part for part in parts if isinstance(part, str)
    ]
    assert len(image_parts) == 5
    assert len(text_parts) == 16
    assert all("RIGHT" not in text and "WRIST" not in text for text in text_parts)
    assert sum("VIEW LEFT" in text for text in text_parts) == 5


def test_representative_rejects_embedded_image_metadata(tmp_path):
    clean_path = tmp_path / "clean.jpg"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(
        clean_path,
        format="JPEG",
    )
    assert read_clean_image_bytes(clean_path) == clean_path.read_bytes()

    dirty_path = tmp_path / "dirty.jpg"
    exif = Image.Exif()
    exif[0x010E] = "phase=pull"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(
        dirty_path,
        format="JPEG",
        exif=exif,
    )
    with pytest.raises(ValueError, match="Dirty inline image metadata"):
        read_clean_image_bytes(dirty_path)


def test_representative_requires_exactly_five_timestamps_per_representative():
    frames = [
        {"left": "left.jpg", "right": "right.jpg", "wrist": "wrist.jpg"}
        for _ in range(4)
    ]
    cluster = {
        "cluster_id": "cluster",
        "representative_sample_ids": ["sample"],
        "representative_clip_paths": ["clip.mp4"],
        "representative_source_view_frame_paths": [frames],
    }
    with pytest.raises(ValueError, match="exactly 5 timestamps"):
        validate_representative_media(cluster)


def test_representative_single_view_uses_only_paired_left_frame_paths():
    left_groups = [
        [f"rep-{representative}-t{timestamp}-left.jpg" for timestamp in range(5)]
        for representative in range(5)
    ]
    source_groups = [
        [
            {
                "left": left_path,
                "right": left_path.replace("left", "right"),
                "wrist": left_path.replace("left", "wrist"),
            }
            for left_path in group
        ]
        for group in left_groups
    ]
    cluster = {
        "cluster_id": "cluster",
        "representative_sample_ids": [
            f"sample-{index}" for index in range(5)
        ],
        "representative_clip_paths": [
            f"clip-{index}.mp4" for index in range(5)
        ],
        "representative_single_view_frame_paths": left_groups,
        "representative_source_view_frame_paths": source_groups,
    }

    single = validate_representative_media(
        cluster,
        media_layout=SINGLE_VIEW_LEFT_LAYOUT,
    )
    multiview = validate_representative_media(
        cluster,
        media_layout=SEPARATE_MULTIVIEW_LAYOUT,
    )

    assert single == [
        [{"left": path} for path in group] for group in left_groups
    ]
    assert [
        [frame["left"] for frame in group] for group in multiview
    ] == left_groups
    assert all(
        set(frame) == {"left"}
        for group in single
        for frame in group
    )


def test_representative_representative_hash_is_annotation_view_independent():
    cluster = {
        "cluster_id": "cluster",
        "task_description": PICK_TASK,
        "representative_sample_ids": [
            f"sample-{index}" for index in range(5)
        ],
        "representative_episode_nums": list(range(5)),
        "representative_selection_strategy": (
            "centroid_nearest_five_unique_episode"
        ),
        "representative_selection_roles": ["centroid"] * 5,
        "representative_single_view_frame_paths": [
            [f"left-{rep}-{frame}.jpg" for frame in range(5)]
            for rep in range(5)
        ],
        "representative_source_view_frame_paths": [
            [
                {
                    "left": f"left-{rep}-{frame}.jpg",
                    "right": f"right-{rep}-{frame}.jpg",
                    "wrist": f"wrist-{rep}-{frame}.jpg",
                }
                for frame in range(5)
            ]
            for rep in range(5)
        ],
    }

    before = hash_representative_set(cluster)
    cluster["representative_source_view_frame_paths"][0][0]["right"] = (
        "different-right.jpg"
    )
    assert hash_representative_set(cluster) == before
    cluster["representative_sample_ids"][0] = "different-sample"
    assert hash_representative_set(cluster) != before
