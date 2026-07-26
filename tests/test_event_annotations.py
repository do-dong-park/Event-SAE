import json
from types import SimpleNamespace

import pytest

import event_sae.events.annotate as annotation_module
from event_sae.events.annotate import (
    annotate_clusters,
    call_gemini,
    parse_annotation_response,
)
from event_sae.events.prompts import (
    MULTIVIEW_TRIPTYCH_LAYOUT,
    PAPER_ANNOTATION_PROTOCOL,
    ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    SINGLE_VIEW_LEFT_LAYOUT,
    build_cluster_annotation_prompt,
    resolve_robocasa_phase_vocabulary,
)


class _FakeModels:
    def __init__(self):
        self.contents = None
        self.config = None

    def generate_content(self, *, model, contents, config):
        self.contents = contents
        self.config = config
        return SimpleNamespace(text='{"phrase":"approaching drawer","phase":"pre_grasp"}')


def test_call_gemini_uses_jpeg_mime(tmp_path):
    frame_path = tmp_path / "frame_step0042.jpg"
    frame_path.write_bytes(b"jpeg-bytes")
    models = _FakeModels()
    client = SimpleNamespace(models=models)

    response = call_gemini(
        client=client,
        model="fake-model",
        prompt="prompt",
        frame_path_groups=[[str(frame_path)]],
    )

    assert "approaching drawer" in response
    parts = [item for item in models.contents if not isinstance(item, str)]
    assert len(parts) == 1
    assert parts[0].inline_data.mime_type == "image/jpeg"
    assert models.config.response_json_schema["required"] == ["phrase", "phase"]
    assert models.config.response_json_schema["additionalProperties"] is False
    assert "pre_grasp" in models.config.response_json_schema["properties"]["phase"]["enum"]
    text_parts = [item for item in models.contents if isinstance(item, str)]
    assert not any("step0042" in item for item in text_parts)
    assert not any("50.0%" in item for item in text_parts)


def test_annotate_clusters_uses_protocol_defaults_and_refuses_overwrite(
    tmp_path,
    monkeypatch,
):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg-bytes")
    clusters = []
    for index, coverage in enumerate((0.8, 0.3, 0.29)):
        clusters.append(
            {
                "cluster_id": f"cluster_{index}",
                "task_description": "Open the drawer.",
                # The sweep baked this flag at 0.5. Annotation must evaluate
                # its requested threshold from the raw coverage instead.
                "meets_min_coverage": coverage >= 0.5,
                "representative_sample_ids": [f"sample_{index}"],
                "representative_clip_paths": ["clip.mp4"],
                "representative_frame_paths": [[str(frame_path)]],
                "representative_progress_percents": [0.5],
                "episode_coverage": coverage,
            }
        )
    clusters_path = tmp_path / "clusters.jsonl"
    clusters_path.write_text(
        "".join(json.dumps(cluster) + "\n" for cluster in clusters), encoding="utf-8"
    )
    output_path = tmp_path / "annotations.jsonl"
    monkeypatch.setattr(annotation_module.genai, "Client", lambda api_key, **kwargs: object())
    monkeypatch.setattr(
        annotation_module,
        "call_gemini",
        lambda **kwargs: json.dumps(
            {
                "phrase": "approaching drawer",
                "phase": kwargs["allowed_phase_labels"][0],
            }
        ),
    )

    annotate_clusters(
        clusters_path,
        output_path,
        api_key="fake-key",
        min_episode_coverage=0.3,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["cluster_id"] for row in rows] == ["cluster_0", "cluster_1"]
    assert rows[0]["phase"] == "reach-to-handle"
    assert rows[0]["phase_scheme"] == "robocasa_action"
    assert rows[0]["annotation_min_episode_coverage"] == 0.3
    assert rows[0]["prompt_text"].startswith(
        "You are labeling a recurring event"
    )
    assert "Cluster id: cluster_0" in rows[0]["prompt_text"]
    assert "Episode coverage: 0.800" in rows[0]["prompt_text"]
    assert "Representative relative progress by clip" not in rows[0]["prompt_text"]
    assert rows[0]["prompt_input_policy"] == {
        "task_instruction": True,
        "representative_images": True,
        "media_layout_description": True,
        "cluster_id": True,
        "episode_coverage": True,
        "relative_progress": False,
        "source_frame_step": False,
    }
    assert len(rows[0]["prompt_sha256"]) == 64
    assert rows[0]["generation_config"] == {
        "temperature": 0.2,
        "response_mime_type": "application/json",
        "response_schema_version": "event_sae_cluster_phrase_phase_v1",
        "response_json_schema": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "minLength": 1},
                "phase": {
                    "type": "string",
                    "enum": [
                        "reach-to-handle",
                        "grasp-handle",
                        "pull",
                        "push-back",
                        "disengage",
                        "wrong-grasp",
                        "open-done",
                    ],
                },
            },
            "required": ["phrase", "phase"],
            "additionalProperties": False,
        },
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        annotate_clusters(
            clusters_path,
            output_path,
            api_key="fake-key",
            protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
        )
    assert rows[0]["phase_labeler_provenance"]["commit"] == (
        "ea61d24ac312b555ca3bcc6f668463ae8f540f7b"
    )

    all_output_path = tmp_path / "all_annotations.jsonl"
    annotate_clusters(
        clusters_path,
        all_output_path,
        api_key="fake-key",
        min_episode_coverage=None,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    all_rows = [json.loads(line) for line in all_output_path.read_text().splitlines()]
    assert [row["cluster_id"] for row in all_rows] == ["cluster_0", "cluster_1", "cluster_2"]
    selected_output_path = tmp_path / "selected_annotations.jsonl"
    annotate_clusters(
        clusters_path,
        selected_output_path,
        api_key="fake-key",
        cluster_ids=("cluster_1",),
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    selected_rows = [
        json.loads(line) for line in selected_output_path.read_text().splitlines()
    ]
    assert [row["cluster_id"] for row in selected_rows] == ["cluster_1"]

    with pytest.raises(ValueError, match="Unknown cluster_ids"):
        annotate_clusters(
            clusters_path,
            tmp_path / "unknown_cluster_annotations.jsonl",
            api_key="fake-key",
            cluster_ids=("missing_cluster",),
            protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        annotate_clusters(
            clusters_path,
            tmp_path / "invalid_annotations.jsonl",
            api_key="fake-key",
            min_episode_coverage=1.1,
            protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
        )

    paper_output_path = tmp_path / "paper_annotations.jsonl"
    annotate_clusters(
        clusters_path,
        paper_output_path,
        api_key="fake-key",
    )
    paper_rows = [
        json.loads(line) for line in paper_output_path.read_text().splitlines()
    ]
    assert len(paper_rows) == 3
    assert paper_rows[0]["model"] == "gemini-2.5-flash"
    assert paper_rows[0]["phase_scheme"] == "paper"
    assert paper_rows[0]["annotation_min_episode_coverage"] is None


def test_task_specific_action_vocabulary_keeps_paper_prompt_protocol():
    pick_task = "Pick the beer from the counter and place it in the cabinet."
    drawer_task = "Open the left drawer."
    pick_labels, _ = resolve_robocasa_phase_vocabulary(pick_task)
    drawer_labels, _ = resolve_robocasa_phase_vocabulary(drawer_task)

    assert pick_labels == (
        "reach-to-object",
        "grasp",
        "transport",
        "place",
        "insert-settle",
        "terminal",
        "wrong-grasp",
    )
    assert drawer_labels == (
        "reach-to-handle",
        "grasp-handle",
        "pull",
        "push-back",
        "disengage",
        "wrong-grasp",
        "open-done",
    )

    prompt = build_cluster_annotation_prompt(
        task_description=pick_task,
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    assert '"transport"' in prompt
    assert '"pull"' not in prompt
    assert "Use the full before-to-after window" in prompt
    assert "active target motion is not required at the temporal center" in prompt
    assert "do not require secure closure" in prompt
    assert "not hidden simulator distances" in prompt
    assert "0.25 m" not in prompt
    assert "intended to depict the same recurring event type" in prompt
    assert "Cluster id: cluster_00" in prompt
    assert "Episode coverage: 0.800" in prompt
    assert "Representative relative progress by clip" not in prompt
    assert "trajectory progress" not in prompt
    assert "last-resort tie-breaker" not in prompt
    assert "temporal center" in prompt
    assert '"phrase": "short canonical event phrase"' in prompt
    assert '"phase": "one label from the closed set above"' in prompt
    assert '"confidence"' not in prompt
    assert '"status"' not in prompt

    drawer_prompt = build_cluster_annotation_prompt(
        task_description=drawer_task,
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    assert "requested opening direction" in drawer_prompt
    assert "current handle contact is visually unclear" in drawer_prompt
    assert "prior near-handle engagement or an opening attempt" in drawer_prompt
    assert "Fully-open completion" in drawer_prompt
    assert "0.10 m" not in drawer_prompt
    assert "0.95" not in drawer_prompt

    multiview_prompt = build_cluster_annotation_prompt(
        task_description=pick_task,
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
        media_layout=MULTIVIEW_TRIPTYCH_LAYOUT,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    assert "simultaneous views, not consecutive time steps" in multiview_prompt
    assert "robot0_eye_in_hand wrist camera" in multiview_prompt
    assert "Proximity alone supports only" in multiview_prompt

    single_view_prompt = build_cluster_annotation_prompt(
        task_description=pick_task,
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
        media_layout=SINGLE_VIEW_LEFT_LAYOUT,
        protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
    )
    assert "robot0_agentview_left observation" in single_view_prompt
    assert "must not be invented from expected task order" in single_view_prompt

    with pytest.raises(ValueError, match="Unknown annotation media layout"):
        build_cluster_annotation_prompt(
            task_description=pick_task,
            cluster_id="cluster_00",
            num_sequences=5,
            num_frames_per_sequence=5,
            episode_coverage=0.8,
            media_layout="unknown-layout",
            protocol=ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
        )


def test_paper_protocol_does_not_inherit_robocasa_state_rules():
    prompt = build_cluster_annotation_prompt(
        task_description="Pick up the red block.",
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
        protocol=PAPER_ANNOTATION_PROTOCOL,
    )

    assert "Cluster id: cluster_00" in prompt
    assert "Episode coverage: 0.800" in prompt
    assert "relative progress" not in prompt.lower()
    assert "prioritize the later frames when choosing the phrase and phase" in prompt
    assert "temporal center" not in prompt
    assert "state-based and non-monotone" not in prompt
    assert "target/distractor identity" not in prompt


def test_parser_uses_selected_vocabulary_and_rejects_extra_output_keys():
    action_labels, _ = resolve_robocasa_phase_vocabulary("Open the drawer.")
    phrase, phase, error = parse_annotation_response(
        '{"phrase":"pulling the drawer open","phase":"pull"}',
        action_labels,
    )
    assert (phrase, phase, error) == ("pulling the drawer open", "pull", None)

    _, _, error = parse_annotation_response(
        '{"phrase":"pulling","phase":"pull","confidence":0.9}',
        action_labels,
    )
    assert "unexpected_keys" in error

    _, _, error = parse_annotation_response(
        '{"phrase":"contacting","phase":"contact"}',
        action_labels,
    )
    assert "invalid_phase:contact" in error


def test_paper_scheme_remains_available_as_a_separate_pass():
    prompt = build_cluster_annotation_prompt(
        task_description="Open the drawer.",
        cluster_id="cluster_00",
        num_sequences=5,
        num_frames_per_sequence=5,
        episode_coverage=0.8,
    )
    assert '"pre_grasp"' in prompt
    assert '"pull"' not in prompt


def _media_layout_cluster(frame_path):
    return {
        "cluster_id": "cluster_0",
        "task_description": "Open the drawer.",
        "representative_sample_ids": ["sample_0"],
        "representative_clip_paths": ["clip.mp4"],
        "representative_frame_paths": [[str(frame_path)]],
        "representative_progress_percents": [0.5],
        "episode_coverage": 0.8,
    }


def test_annotation_media_layout_override_is_recorded_and_conflicts_fail(
    tmp_path,
    monkeypatch,
):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg-bytes")
    cluster = _media_layout_cluster(frame_path)
    clusters_path = tmp_path / "clusters.jsonl"
    clusters_path.write_text(json.dumps(cluster) + "\n", encoding="utf-8")
    monkeypatch.setattr(annotation_module.genai, "Client", lambda api_key, **kwargs: object())
    monkeypatch.setattr(
        annotation_module,
        "call_gemini",
        lambda **kwargs: (
            '{"phrase":"approaching drawer","phase":"reach-to-handle"}'
        ),
    )

    output_path = tmp_path / "annotations.jsonl"
    annotate_clusters(
        clusters_path,
        output_path,
        api_key="fake-key",
        media_layout_override=SINGLE_VIEW_LEFT_LAYOUT,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["annotation_media_layout"] == SINGLE_VIEW_LEFT_LAYOUT

    cluster["annotation_media_layout"] = MULTIVIEW_TRIPTYCH_LAYOUT
    clusters_path.write_text(json.dumps(cluster) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with cluster metadata"):
        annotate_clusters(
            clusters_path,
            tmp_path / "conflict.jsonl",
            api_key="fake-key",
            media_layout_override=SINGLE_VIEW_LEFT_LAYOUT,
        )
