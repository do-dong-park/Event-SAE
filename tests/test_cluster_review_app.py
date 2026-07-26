import json
from pathlib import Path

import pytest

import event_sae.events.review as review_module
from event_sae.events.review import (
    ClusterReviewStore,
    build_blind_cluster_review_dataset,
    build_cluster_review_service,
    finalize_reviewed_annotations,
)
from scripts.review_clusters import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_ASSIGNMENTS_PATH,
    DEFAULT_CLUSTERS_PATH,
    DEFAULT_EVENT_FEATURES_PATH,
    DEFAULT_MEDIA_CLUSTERS_PATH,
    DEFAULT_REVIEW_UI_PATH,
    DEFAULT_REVIEWS_PATH,
    build_parser,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    task = "Open the left drawer."
    features = []
    assignments = []
    for index in range(4):
        frames = []
        for frame_index in range(5):
            frame_path = tmp_path / f"sample_{index}_frame_{frame_index}.jpg"
            frame_path.write_bytes(b"jpeg")
            frames.append(str(frame_path))
        sample_id = f"sample_{index}"
        cluster_id = f"drawer_cluster_0{index // 2}"
        features.append(
            {
                "sample_id": sample_id,
                "task_description": task,
                "task_episode_idx": index,
                "episode_num": index,
                "waypoint_rank": 0,
                "waypoint_step": index + 1,
                "num_steps": 10,
                "progress_percent": (index + 1) / 10,
                "vision_embedding": [
                    float(index == 0),
                    float(index == 1),
                    float(index == 2),
                    float(index == 3),
                ],
                "state_vector": [float(index), 0.0, 1.0],
                "frame_paths": frames,
                "success": bool(index % 2),
            }
        )
        assignments.append(
            {
                "sample_id": sample_id,
                "cluster_id": cluster_id,
            }
        )

    clusters = []
    for cluster_index in range(2):
        member_ids = [f"sample_{2 * cluster_index}", f"sample_{2 * cluster_index + 1}"]
        clusters.append(
            {
                "cluster_id": f"drawer_cluster_0{cluster_index}",
                "cluster_label": cluster_index,
                "task_description": task,
                "num_members": 2,
                "episode_coverage": 0.5,
                "total_task_episodes": 4,
                "cluster_mean_progress_percent": 0.25 + 0.5 * cluster_index,
                "member_sample_ids": member_ids,
                "representative_sample_ids": [member_ids[0]],
            }
        )

    annotation = {
        "cluster_id": "drawer_cluster_00",
        "task_description": task,
        "phrase": "reaching toward the handle",
        "phase": "reach-to-handle",
        "allowed_phase_labels": ["reach-to-handle", "pull", "open-done"],
        "model": "fake-gemini",
        "prompt_version": "test-v1",
        "annotation_min_episode_coverage": 0.3,
        "api_error": None,
        "parse_error": None,
    }

    features_path = tmp_path / "features.jsonl"
    clusters_path = tmp_path / "clusters.jsonl"
    assignments_path = tmp_path / "assignments.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    _write_jsonl(features_path, features)
    _write_jsonl(clusters_path, clusters)
    _write_jsonl(assignments_path, assignments)
    _write_jsonl(annotations_path, [annotation])
    return features_path, clusters_path, assignments_path, annotations_path


def _fixture_multiview_clusters(tmp_path: Path, *, clusters_path: Path) -> Path:
    clusters = [
        json.loads(line)
        for line in clusters_path.read_text(encoding="utf-8").splitlines()
    ]
    target = clusters[0]
    frame_groups = []
    for sample_id in target["representative_sample_ids"]:
        frames = []
        for frame_index in range(5):
            frame_path = tmp_path / f"{sample_id}_triptych_{frame_index}.jpg"
            frame_path.write_bytes(b"triptych")
            frames.append(str(frame_path))
        frame_groups.append(frames)

    path = tmp_path / "clusters_multiview.jsonl"
    _write_jsonl(
        path,
        [
            {
                "cluster_id": target["cluster_id"],
                "representative_sample_ids": target["representative_sample_ids"],
                "representative_frame_paths": frame_groups,
            }
        ],
    )
    return path


def _save_blind_review(
    store: ClusterReviewStore,
    *,
    phrase: str = "approaching the drawer handle",
    phase: str = "reach-to-handle",
    reviewer: str = "DK",
    notes: str = "",
) -> dict:
    return store.save(
        {
            "stage": "blind",
            "cluster_id": "drawer_cluster_00",
            "reviewer": reviewer,
            "human_phrase": phrase,
            "human_phase": phase,
            "mixed_cluster": False,
            "visually_insufficient": False,
            "notes": notes,
        }
    )


def _adjudicate_review(
    store: ClusterReviewStore,
    *,
    phrase_phase_consistent: bool,
    notes: str = "",
) -> dict:
    return store.save(
        {
            "stage": "adjudication",
            "cluster_id": "drawer_cluster_00",
            "phrase_phase_consistent": phrase_phase_consistent,
            "adjudication_notes": notes,
        }
    )


def test_review_dataset_joins_clusters_and_keeps_outcome_blind(tmp_path: Path) -> None:
    features_path, clusters_path, assignments_path, annotations_path = _fixture_artifacts(
        tmp_path
    )

    payload, media_paths, annotations = build_blind_cluster_review_dataset(
        event_features_path=features_path,
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        annotations_path=annotations_path,
        projection_method="pca",
    )

    assert payload["meta"]["num_samples"] == 4
    assert payload["meta"]["num_clusters"] == 2
    assert payload["meta"]["num_annotated_clusters"] == 1
    assert payload["meta"]["annotation_min_episode_coverage"] == 0.3
    assert payload["meta"]["success_omitted"] is True
    assert "task-local balanced C0 descriptor" in payload["meta"][
        "projection_scope"
    ]
    assert set(media_paths) == {f"clip_{index:06d}" for index in range(4)}
    assert set(annotations) == {"drawer_cluster_00"}
    assert all("success" not in sample for sample in payload["samples"].values())
    assert all(len(sample["projection"]) == 2 for sample in payload["samples"].values())
    assert payload["clusters"][0]["annotation_target"] is True
    assert "annotation" not in payload["clusters"][0]
    assert payload["clusters"][0]["review_protocol"]["allowed_phase_labels"] == [
        "reach-to-handle",
        "pull",
        "open-done",
    ]
    assert payload["clusters"][1]["annotation_target"] is False
    for sample in payload["samples"].values():
        assert sample["num_frames"] == 5
        assert not {
            "task_episode_idx",
            "episode_num",
            "waypoint_rank",
            "waypoint_step",
            "num_steps",
            "progress_percent",
            "frame_names",
        }.intersection(sample)
    assert all(cluster["total_task_episodes"] == 4 for cluster in payload["clusters"])


@pytest.mark.parametrize(
    ("artifact_index", "message"),
    [
        (2, "Cluster assignments contain duplicate sample IDs"),
        (3, "Annotations contain duplicate cluster IDs"),
    ],
)
def test_review_dataset_rejects_duplicate_join_keys(
    tmp_path: Path,
    artifact_index: int,
    message: str,
) -> None:
    artifacts = _fixture_artifacts(tmp_path)
    target = artifacts[artifact_index]
    first_row = target.read_text(encoding="utf-8").splitlines()[0]
    with target.open("a", encoding="utf-8") as handle:
        handle.write(first_row + "\n")

    with pytest.raises(ValueError, match=message):
        build_blind_cluster_review_dataset(
            event_features_path=artifacts[0],
            clusters_path=artifacts[1],
            assignments_path=artifacts[2],
            annotations_path=artifacts[3],
            projection_method="pca",
        )



def test_review_dataset_uses_only_versioned_representative_media(
    tmp_path: Path,
) -> None:
    features_path, clusters_path, assignments_path, annotations_path = _fixture_artifacts(
        tmp_path
    )
    multiview_path = _fixture_multiview_clusters(
        tmp_path,
        clusters_path=clusters_path,
    )

    payload, media_paths, _ = build_blind_cluster_review_dataset(
        event_features_path=features_path,
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        annotations_path=annotations_path,
        representative_media_clusters_path=multiview_path,
        condition_id="v7",
        condition_label="v7 · LEFT/RIGHT/WRIST",
        media_layout="synchronized LEFT | RIGHT | WRIST triptych",
        projection_method="pca",
    )

    assert payload["format"] == "event_sae_stage3_human_review_payload_v3"
    assert payload["meta"]["version_id"] == "v7"
    assert payload["meta"]["media_scope"] == "representative-only"
    assert payload["meta"]["num_playable_samples"] == 1
    assert set(media_paths) == {"clip_000000"}
    assert payload["samples"]["clip_000000"]["media_available"] is True
    assert payload["samples"]["clip_000001"]["media_available"] is False
    assert payload["samples"]["clip_000000"]["num_frames"] == 5
    clusters_by_id = {
        cluster["cluster_id"]: cluster for cluster in payload["clusters"]
    }
    assert clusters_by_id["drawer_cluster_00"]["playable_sample_ids"] == [
        "clip_000000"
    ]
    assert clusters_by_id["drawer_cluster_01"]["playable_sample_ids"] == []


def test_review_application_exposes_condition_data_and_store(
    tmp_path: Path,
) -> None:
    features_path, clusters_path, assignments_path, annotations_path = _fixture_artifacts(
        tmp_path
    )
    multiview_path = _fixture_multiview_clusters(
        tmp_path,
        clusters_path=clusters_path,
    )
    ui_path = tmp_path / "review.html"
    ui_path.write_text("<html></html>", encoding="utf-8")
    application = build_cluster_review_service(
        event_features_path=features_path,
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        annotations_path=annotations_path,
        reviews_path=tmp_path / "reviews_v9.json",
        media_clusters_path=multiview_path,
        ui_path=ui_path,
        projection_method="pca",
        condition_id="v9",
        condition_label="3-view clusters · ABS + gripper (v9)",
        media_layout="synchronized LEFT | RIGHT | WRIST triptych",
        condition_metadata={
            "experimental_role": "historical",
            "awe_anchor": "abs position + gripper",
            "clustering_view": "3-view",
            "annotation_view": "3-view",
        },
    )

    assert application.data()["meta"]["version_id"] == "v9"
    assert set(application.media_paths) == {"clip_000000"}

    _save_blind_review(application.review_store)
    _adjudicate_review(
        application.review_store,
        phrase_phase_consistent=True,
    )
    assert len(application.reviews()["reviews"]) == 1
    assert len(application.data()["review_document"]["reviews"]) == 1


def test_review_store_validates_corrections_and_persists_atomically(tmp_path: Path) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )

    with pytest.raises(ValueError, match="outside the allowed vocabulary"):
        _save_blind_review(
            store,
            phrase="contacting handle",
            phase="not-a-phase",
        )

    blind = _save_blind_review(
        store,
        phrase="pulling the drawer",
        phase="pull",
        notes="Matches all representative clips.",
    )
    assert blind["human_phase"] == "pull"
    assert reviews_path.is_file()
    assert not reviews_path.with_suffix(".json.tmp").exists()

    corrected = _adjudicate_review(
        store,
        phrase_phase_consistent=False,
        notes="The model phase does not match the blind assessment.",
    )
    assert corrected["verdict"] == "corrected"
    assert corrected["human_phrase"] == "pulling the drawer"
    saved_timestamp = store.document()["updated_at"]
    assert saved_timestamp == corrected["reviewed_at"]
    assert store.document()["updated_at"] == saved_timestamp
    reloaded = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    assert reloaded.document()["reviews"][0]["human_phase"] == "pull"
    loaded_timestamp = reloaded.document()["updated_at"]
    assert loaded_timestamp == corrected["reviewed_at"]
    assert reloaded.document()["updated_at"] == loaded_timestamp


def test_review_store_enforces_blind_lock_before_model_reveal(
    tmp_path: Path,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    store = ClusterReviewStore(
        tmp_path / "reviews.json",
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )

    with pytest.raises(PermissionError, match="blind assessment"):
        store.reveal_annotation("drawer_cluster_00")

    with pytest.raises(ValueError, match="Invalid review stage"):
        store.save(
            {
                "cluster_id": "drawer_cluster_00",
                "verdict": "approved",
            }
        )

    blind = store.save(
        {
            "stage": "blind",
            "cluster_id": "drawer_cluster_00",
            "reviewer": "DK",
            "human_phrase": "approaching the drawer handle",
            "human_phase": "reach-to-handle",
            "mixed_cluster": False,
            "visually_insufficient": False,
            "notes": "All clips share the same far approach state.",
        }
    )
    assert blind["review_stage"] == "blind_recorded"
    assert "model_phrase" not in blind
    assert "model_phase" not in blind
    assert "legacy_review" not in blind
    assert store.document()["format"] == (
        "event_sae_stage3_human_cluster_reviews_v2"
    )
    assert store.reveal_annotation("drawer_cluster_00")["phase"] == (
        "reach-to-handle"
    )

    with pytest.raises(ValueError, match="already locked"):
        store.save(
            {
                "stage": "blind",
                "cluster_id": "drawer_cluster_00",
                "human_phrase": "changed after reveal",
                "human_phase": "pull",
                "mixed_cluster": False,
                "visually_insufficient": False,
            }
        )

    adjudicated = store.save(
        {
            "stage": "adjudication",
            "cluster_id": "drawer_cluster_00",
            "phrase_phase_consistent": True,
            "adjudication_notes": "Phrase and phase agree.",
        }
    )
    assert adjudicated["review_stage"] == "adjudicated"
    assert adjudicated["verdict"] == "approved"
    assert adjudicated["phase_corrected"] is False
    assert adjudicated["phrase_phase_consistent"] is True


def test_review_store_rejects_duplicate_rows_in_existing_document(
    tmp_path: Path,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    _save_blind_review(store)
    document = json.loads(reviews_path.read_text(encoding="utf-8"))
    document["reviews"].append(dict(document["reviews"][0]))
    reviews_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate cluster IDs"):
        ClusterReviewStore(
            reviews_path,
            annotations_by_id={annotation["cluster_id"]: annotation},
            annotations_path=annotations_path,
        )


def test_review_store_rejects_legacy_document_format(tmp_path: Path) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    _save_blind_review(store)
    document = json.loads(reviews_path.read_text(encoding="utf-8"))
    document["format"] = "event_sae_stage3_human_cluster_reviews_v1"
    reviews_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported review document format"):
        ClusterReviewStore(
            reviews_path,
            annotations_by_id={annotation["cluster_id"]: annotation},
            annotations_path=annotations_path,
        )


def test_review_store_replace_failure_rolls_back_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    _save_blind_review(store, notes="persisted")
    before_document = store.document()
    before_bytes = reviews_path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(review_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        _adjudicate_review(
            store,
            phrase_phase_consistent=False,
        )

    assert store.document() == before_document
    assert reviews_path.read_bytes() == before_bytes
    assert list(tmp_path.glob(f".{reviews_path.name}.*.tmp")) == []


def test_review_store_write_failure_rolls_back_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    before_document = store.document()

    real_named_temporary_file = review_module.tempfile.NamedTemporaryFile

    class FailingWrite:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.handle = real_named_temporary_file(*args, **kwargs)

        @property
        def name(self) -> str:
            return self.handle.name

        def __enter__(self) -> "FailingWrite":
            self.handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.handle.__exit__(*args)

        def write(self, value: str) -> None:
            raise OSError("injected write failure")

    monkeypatch.setattr(
        review_module.tempfile,
        "NamedTemporaryFile",
        FailingWrite,
    )
    with pytest.raises(OSError, match="injected write failure"):
        _save_blind_review(store)

    assert store.document() == before_document
    assert not reviews_path.exists()
    assert list(tmp_path.glob(f".{reviews_path.name}.*.tmp")) == []


def test_finalize_reviewed_annotations_derives_count_from_validated_artifact(
    tmp_path: Path,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    output_path = tmp_path / "finalized.jsonl"

    rows = finalize_reviewed_annotations(
        annotations_path=annotations_path,
        output_path=output_path,
        reviews_path=None,
        assume_approved=True,
    )

    assert len(rows) == 1
    assert output_path.is_file()


def test_finalize_reviewed_annotations_rejects_blind_only_reviews(
    tmp_path: Path,
) -> None:
    _, _, _, annotations_path = _fixture_artifacts(tmp_path)
    annotation = json.loads(annotations_path.read_text().splitlines()[0])
    reviews_path = tmp_path / "reviews.json"
    store = ClusterReviewStore(
        reviews_path,
        annotations_by_id={annotation["cluster_id"]: annotation},
        annotations_path=annotations_path,
    )
    store.save(
        {
            "stage": "blind",
            "cluster_id": "drawer_cluster_00",
            "human_phrase": "approaching the handle",
            "human_phase": "reach-to-handle",
            "mixed_cluster": False,
            "visually_insufficient": False,
        }
    )

    with pytest.raises(ValueError, match="not completed model adjudication"):
        finalize_reviewed_annotations(
            annotations_path=annotations_path,
            output_path=tmp_path / "finalized.jsonl",
            reviews_path=reviews_path,
            assume_approved=False,
        )


def test_review_ui_includes_human_audit_safety_controls() -> None:
    html = DEFAULT_REVIEW_UI_PATH.read_text(encoding="utf-8")

    for element_id in (
        "coverageValue",
        "heroWrap",
        "previousFrame",
        "nextFrame",
        "shortcutDialog",
        "saveReview",
        "reviewNext",
        "groupMode",
    ):
        assert f'id="{element_id}"' in html or f'id=\"{element_id}\"' in html

    assert "const PHASE_GUIDES" not in html
    assert "protocol.phase_descriptions[phase]" in html
    assert "confirmDiscardReview" in html
    assert 'addEventListener("beforeunload"' in html
    assert 'data-review-filter="unreviewed"' in html
    assert 'preserveAspectRatio="xMidYMid meet"' in html
    assert 'preserveAspectRatio="none"' not in html
    assert "const scatterExtent = 540" in html
    assert "@media (max-width: 920px)" in html
    assert "<span>LEFT</span><span>RIGHT</span><span>WRIST</span>" in html
    assert 'id="coverageSlider"' not in html
    assert 'type="range"' not in html
    assert "Fixed by the annotation artifact" in html
    assert "annotation_min_episode_coverage" in html
    assert '<option value="phase">Human phase groups</option>' in html
    assert '<option value="cluster">Raw event clusters</option>' in html
    assert 'groupMode: "cluster"' in html
    assert "function resolvedAnnotation(cluster)" in html
    assert "cluster.annotation" not in html
    assert "Blind review pending" in html
    assert "Lock assessment + reveal Gemini" in html
    assert 'stage: "blind"' in html
    assert 'stage: "adjudication"' in html
    assert "phrase_phase_consistent" in html
    assert "mixed_cluster" in html
    assert "visually_insufficient" in html
    assert "appendPhaseGroupHeading" in html
    assert "C${count.clusters} · S${count.samples}" in html
    assert "resolvedAnnotation(cluster).phase" in html
    assert "sample.waypoint_step" not in html
    assert "sample.progress_percent" not in html
    assert "sample.episode_num" not in html
    assert "sample.frame_names" not in html
    assert 'id="versionSelect"' not in html
    assert 'data-mode="alignment"' not in html
    assert "/api/alignment" not in html
    assert 'id="reviewWorkspace"' not in html
    assert 'id="queueFilter"' not in html
    assert 'id="scatterWrap"' not in html
    assert "review-only" not in html
    assert "sample.num_frames" in html
    assert "(state.frame + 1) % 5" not in html
    assert "Math.min(4" not in html
    assert "Five-frame" not in html


def test_review_cli_exposes_focused_subcommands_and_serve_defaults() -> None:
    parser = build_parser()
    subparser_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    command_parsers = subparser_action.choices
    assert set(command_parsers) == {
        "serve",
        "results",
        "triptychs",
        "audit",
        "finalize",
        "phase-groups",
        "contact-sheets",
        "filter-annotation-rows",
        "merge-annotation-attempts",
        "annotation-consistency",
    }
    expected_help_options = {
        "serve": "--projection",
        "triptychs": "--left-samples-path",
        "audit": "--output-annotations-path",
        "finalize": "--assume-approved",
        "phase-groups": "--finalized-annotations-path",
        "contact-sheets": "--tile-size",
        "filter-annotation-rows": "--rows-path",
        "merge-annotation-attempts": "--attempt-path",
        "annotation-consistency": "--run-manifest",
    }
    for command, option in expected_help_options.items():
        assert option in command_parsers[command].format_help()

    args = parser.parse_args(["serve"])
    assert args.event_features_path == DEFAULT_EVENT_FEATURES_PATH
    assert args.clusters_path == DEFAULT_CLUSTERS_PATH
    assert args.assignments_path == DEFAULT_ASSIGNMENTS_PATH
    assert args.media_clusters_path == DEFAULT_MEDIA_CLUSTERS_PATH
    assert args.annotations_path == DEFAULT_ANNOTATIONS_PATH
    assert args.reviews_path == DEFAULT_REVIEWS_PATH
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.projection == "tsne"
    assert args.condition_id == "v9"
    assert args.annotation_view == "3-view"
    assert args.clustering_view == "3-view"
    assert args.awe_anchor == "abs position + gripper"
    assert args.use_feature_media is False
    assert command_parsers["finalize"].get_default("expected_clusters") is None
