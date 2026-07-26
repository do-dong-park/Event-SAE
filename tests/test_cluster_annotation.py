from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from event_sae import sha256_file
from event_sae.groot import cluster_annotation as annotation


def _vote(index: int, phase: str | None, *, visibility: str = "clear") -> dict:
    return {
        "representative_index": index,
        "phase": phase,
        "phrase": f"phrase-{index}",
        "visibility": visibility,
        "api_error": None,
        "parse_error": None,
    }


def test_centroid_selection_is_stable_and_episode_unique() -> None:
    records = [
        {
            "sample_id": f"s{index}",
            "episode_num": episode,
            "waypoint_step": index,
            "waypoint_rank": index,
        }
        for index, episode in enumerate((0, 0, 1, 2, 3, 4, 5))
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.95, 0.05],
            [-1.0, 0.0],
        ]
    )
    selected, distances, ranks = annotation.select_centroid_representatives(
        records,
        vectors,
    )
    assert len(selected) == len(distances) == len(ranks) == 5
    assert len({row["episode_num"] for row in selected}) == 5
    reversed_selected, _, _ = annotation.select_centroid_representatives(
        list(reversed(records)),
        vectors[::-1],
    )
    assert [row["sample_id"] for row in reversed_selected] == [
        row["sample_id"] for row in selected
    ]


def test_centroid_selection_supports_nine_unique_episodes() -> None:
    records = [
        {
            "sample_id": f"s{index}",
            "episode_num": episode,
            "waypoint_step": index,
            "waypoint_rank": index,
        }
        for index, episode in enumerate((0, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    ]
    vectors = np.eye(len(records))
    first_five, _, _ = annotation.select_centroid_representatives(
        records,
        vectors,
        count=5,
    )
    selected, distances, ranks = annotation.select_centroid_representatives(
        records,
        vectors,
        count=9,
    )
    assert len(selected) == len(distances) == len(ranks) == 9
    assert len({row["episode_num"] for row in selected}) == 9
    assert [row["sample_id"] for row in selected[:5]] == [
        row["sample_id"] for row in first_five
    ]


def test_centroid_selection_fails_without_five_episodes() -> None:
    records = [
        {
            "sample_id": f"s{index}",
            "episode_num": index % 4,
            "waypoint_step": index,
            "waypoint_rank": index,
        }
        for index in range(8)
    ]
    with pytest.raises(ValueError, match="at least 5 unique episodes"):
        annotation.select_centroid_representatives(
            records,
            np.eye(8),
        )


def test_response_normalization_preserves_object_and_unwraps_once() -> None:
    task = "Pick the beer from the counter and place it in the cabinet."
    object_text = json.dumps(
        {
            "phrase": "grasp object",
            "phase": "grasp",
            "visibility": "clear",
        },
        indent=2,
    )
    identity = annotation.normalize_response_text(
        object_text,
        task_description=task,
    )
    assert identity.normalized_text == object_text
    assert identity.source_shape == "object"
    assert identity.changed is False

    singleton = annotation.normalize_response_text(
        f"[{object_text}]",
        task_description=task,
    )
    assert json.loads(singleton.normalized_text) == json.loads(object_text)
    assert singleton.source_shape == "singleton_object_list"
    assert singleton.changed is True
    assert annotation.normalization_provenance(singleton)[
        "normalized_text_sha256"
    ] == singleton.normalized_text_sha256


@pytest.mark.parametrize(
    "payload",
    ("[]", "[{}, {}]", "[[{}]]", '"text"', "null"),
)
def test_response_normalization_rejects_other_shapes(payload: str) -> None:
    with pytest.raises(ValueError):
        annotation.normalize_response_text(
            payload,
            task_description=(
                "Pick the beer from the counter and place it in the cabinet."
            ),
        )


def test_strict_majority_accepts_three_votes() -> None:
    result = annotation.resolve_representative_consensus(
        [
            _vote(1, "grasp"),
            _vote(2, "grasp"),
            _vote(3, "grasp"),
            _vote(4, "transport"),
            _vote(5, "release"),
        ],
        annotation.STRICT_MAJORITY,
    )
    assert result["status"] == "consensus-3-of-5"
    assert result["phase"] == "grasp"
    assert result["policy_id"] == annotation.STRICT_MAJORITY.policy_id


def test_unique_plurality_accepts_unique_two_vote_lead() -> None:
    result = annotation.resolve_representative_consensus(
        [
            _vote(1, "grasp"),
            _vote(2, "grasp"),
            _vote(3, "approach"),
            _vote(4, "transport"),
            _vote(5, "release"),
        ],
        annotation.UNIQUE_PLURALITY,
    )
    assert result["status"] == "plurality-2-of-5"
    assert result["phase"] == "grasp"
    assert result["unique_winner"] is True


def test_unique_plurality_rejects_a_two_vote_tie() -> None:
    result = annotation.resolve_representative_consensus(
        [
            _vote(1, "grasp"),
            _vote(2, "grasp"),
            _vote(3, "transport"),
            _vote(4, "transport"),
            _vote(5, "release"),
        ],
        annotation.UNIQUE_PLURALITY,
    )
    assert result["status"] == "mixed"
    assert result["phase"] is None
    assert result["unique_winner"] is False


def test_adaptive_plurality_resolves_after_seven_contiguous_votes() -> None:
    votes = [
        _vote(1, "grasp"),
        _vote(2, "grasp"),
        _vote(3, "transport"),
        _vote(4, "transport"),
        _vote(5, "release"),
    ]
    initial = annotation.resolve_adaptive_plurality(votes)
    assert initial["status"] == "mixed"
    assert initial["phase"] is None
    assert initial["unique_winner"] is False
    assert initial["num_representatives_evaluated"] == 5

    after_six = annotation.resolve_adaptive_plurality(
        [*votes, _vote(6, "release")]
    )
    assert after_six["status"] == "mixed"
    assert after_six["phase"] is None
    assert after_six["phase_counts"] == {
        "grasp": 2,
        "release": 2,
        "transport": 2,
    }

    after_seven = annotation.resolve_adaptive_plurality(
        [
            *votes,
            _vote(6, "release"),
            _vote(7, "grasp"),
        ]
    )
    assert after_seven["status"] == "plurality-3-of-7"
    assert after_seven["phase"] == "grasp"
    assert after_seven["unique_winner"] is True
    assert after_seven["dominant_votes"] == 3
    assert after_seven["runner_up_votes"] == 2
    assert after_seven["num_representatives_evaluated"] == 7


def test_adaptive_plurality_preserves_a_top_vote_tie_at_seven() -> None:
    result = annotation.resolve_adaptive_plurality(
        [
            _vote(1, "grasp"),
            _vote(2, "grasp"),
            _vote(3, "transport"),
            _vote(4, "transport"),
            _vote(5, "release"),
            _vote(6, "grasp"),
            _vote(7, "transport"),
        ]
    )
    assert result["status"] == "mixed"
    assert result["phase"] is None
    assert result["unique_winner"] is False
    assert result["dominant_votes"] == 3
    assert result["runner_up_votes"] == 3


def test_adaptive_plurality_marks_an_exhausted_tie_at_nine() -> None:
    result = annotation.resolve_adaptive_plurality(
        [
            _vote(1, "grasp"),
            _vote(2, "grasp"),
            _vote(3, "transport"),
            _vote(4, "transport"),
            _vote(5, "release"),
            _vote(6, "grasp"),
            _vote(7, "transport"),
            _vote(8, "release"),
            _vote(9, "release"),
        ],
        exhausted=True,
    )
    assert result["status"] == "mixed-after-9"
    assert result["phase"] is None
    assert result["unique_winner"] is False
    assert result["phase_counts"] == {
        "grasp": 3,
        "release": 3,
        "transport": 3,
    }
    assert result["num_representatives_evaluated"] == 9


def test_adaptive_plurality_continuation_resolves_at_eleven() -> None:
    result = annotation.resolve_adaptive_plurality(
        [
            *[_vote(index, "grasp") for index in range(1, 5)],
            *[_vote(index, "transport") for index in range(5, 9)],
            _vote(9, "place"),
            _vote(10, "grasp"),
            _vote(11, "place"),
        ],
        maximum_rank=17,
    )
    assert result["status"] == "plurality-5-of-11"
    assert result["phase"] == "grasp"
    assert result["phase_counts"] == {
        "grasp": 5,
        "place": 2,
        "transport": 4,
    }


def test_adaptive_plurality_marks_an_exhausted_tie_at_seventeen() -> None:
    result = annotation.resolve_adaptive_plurality(
        [
            *[_vote(index, "grasp") for index in range(1, 9)],
            *[_vote(index, "transport") for index in range(9, 17)],
            _vote(17, "place"),
        ],
        exhausted=True,
        maximum_rank=17,
    )
    assert result["status"] == "mixed-after-17"
    assert result["phase"] is None
    assert result["phase_counts"] == {
        "grasp": 8,
        "place": 1,
        "transport": 8,
    }


def test_adaptive_plurality_rejects_premature_rank_seventeen_exhaustion() -> None:
    with pytest.raises(ValueError, match="maximum-rank prefix"):
        annotation.resolve_adaptive_plurality(
            [_vote(index, "grasp") for index in range(1, 12)],
            exhausted=True,
            maximum_rank=17,
        )


def test_user_phase_override_preserves_tied_evidence_and_provenance() -> None:
    votes = [
        _vote(1, "grasp"),
        _vote(2, "transport"),
        _vote(3, "grasp"),
        _vote(4, "transport"),
        _vote(5, "place"),
    ]
    for vote in votes:
        vote["phrase"] = (
            "initiating a grasp on the target"
            if vote["phase"] == "grasp"
            else f"{vote['phase']} phrase"
        )
    consensus = annotation.resolve_adaptive_plurality(votes)
    source = {
        "cluster_id": "cluster-10",
        "task_description": "Pick the beer.",
        "allowed_phase_labels": ["grasp", "place", "transport"],
        "representative_annotations": votes,
        "consensus": consensus,
        "status": "mixed",
        "phase": None,
        "phrase": None,
        "human_review_completed": False,
    }
    decision = {
        "phase": "grasp",
        "phrase": "initiating a grasp on the target",
        "authorized_by": "workspace_owner",
        "reason": "explicit analysis override",
    }

    result = annotation._apply_user_phase_override(
        source,
        source_run_contract_sha256="source-contract",
        decision=decision,
    )

    assert result["phase"] == "grasp"
    assert result["status"] == "user-directed-phase-override"
    assert result["consensus"] == consensus
    assert result["representative_annotations"] == votes
    assert result["actual_human_review_completed"] is False
    assert result["phase_override"][
        "phrase_source_representative_indices"
    ] == [1, 3]
    assert result["phase_override"]["provider_response_generated"] is False
    assert (
        result["phase_override"]["formal_blind_review_completed"] is False
    )


@pytest.mark.parametrize(
    "indices",
    (
        (1, 2, 3, 4, 6),
        (1, 2, 3, 4, 4),
        (1, 2, 3, 5, 4),
    ),
)
def test_adaptive_plurality_rejects_noncontiguous_vote_indices(
    indices: tuple[int, ...],
) -> None:
    votes = [
        _vote(index, "grasp" if position < 2 else "transport")
        for position, index in enumerate(indices)
    ]
    with pytest.raises(ValueError):
        annotation.resolve_adaptive_plurality(votes)


@pytest.mark.parametrize("count", (4, 10))
def test_adaptive_plurality_rejects_vote_counts_outside_five_to_nine(
    count: int,
) -> None:
    with pytest.raises(ValueError):
        annotation.resolve_adaptive_plurality(
            [_vote(index, "grasp") for index in range(1, count + 1)]
        )


def test_historical_centroid_manifest_and_outputs_remain_read_only() -> None:
    root = (
        annotation.EXPERIMENT_ROOT
        / "annotations_batch_centroid_nearest_five"
    )
    if not root.is_dir():
        pytest.skip("historical centroid artifact is not available")
    manifest_before = sha256_file(root / "run_manifest.json")
    contract_sha, condition_paths = annotation._source_condition_paths(root)
    assert contract_sha == (
        "2a203be108f3568c98b654de9245957582ffa530e0f605b00a2b2279d13c8e2b"
    )
    assert len(condition_paths) == 5
    assert sum(len(annotation.load_jsonl(path)) for _, path in condition_paths) == 90
    assert sha256_file(root / "run_manifest.json") == manifest_before
    with pytest.raises(ValueError, match="disjoint"):
        annotation._validate_derivation_output_root(root, root)


def test_archived_runtime_is_content_addressed() -> None:
    archive = (
        annotation.EXPERIMENT_ROOT
        / "archives/annotation_lineage_source_20260725.tar"
    )
    if not archive.is_file():
        pytest.skip("annotation source archive is not available")
    assert sha256_file(archive) == (
        "71d6ddf9395c5bb41ec3e1a0d455dc55fcdf925ade75e0ec"
        "b15ebb441411543d"
    )
