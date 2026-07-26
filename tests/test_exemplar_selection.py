import math

import numpy as np
import pytest

from event_sae.events.cluster import (
    EXEMPLAR_SELECTION_CENTROID_DIVERSITY,
    select_centroid_diversity_exemplars,
    select_exemplars,
)
from event_sae.groot.cluster_annotation import (
    select_centroid_representatives,
)


def _record(
    sample_id: str,
    episode_num: int,
    *,
    waypoint_step: int = 0,
    waypoint_rank: int = 0,
) -> dict:
    return {
        "sample_id": sample_id,
        "episode_num": episode_num,
        "waypoint_step": waypoint_step,
        "waypoint_rank": waypoint_rank,
    }


def _unit_vector(angle_degrees: float) -> list[float]:
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


def test_centroid3_maximin2_selects_central_three_then_diverse_two() -> None:
    records = [
        _record("center_minus", 0),
        _record("center", 1),
        _record("center_plus", 2),
        _record("outer_plus_55", 3),
        _record("outer_plus_60", 4),
        _record("outer_minus_60", 5),
    ]
    vectors = np.asarray(
        [
            _unit_vector(-5),
            _unit_vector(0),
            _unit_vector(5),
            _unit_vector(55),
            _unit_vector(60),
            _unit_vector(-60),
        ],
        dtype=np.float64,
    )

    selected = select_centroid_diversity_exemplars(records, vectors)

    assert [row["sample_id"] for row in selected[:3]] == [
        "center_plus",
        "center",
        "center_minus",
    ]
    assert [row["sample_id"] for row in selected[3:]] == [
        "outer_minus_60",
        "outer_plus_60",
    ]
    assert len({row["episode_num"] for row in selected}) == 5


def test_centroid3_maximin2_ties_are_independent_of_input_order() -> None:
    records = [
        _record(f"sample_{episode}", episode)
        for episode in range(6)
    ]
    vectors = np.ones((6, 2), dtype=np.float64)
    permutation = [4, 2, 5, 0, 3, 1]

    forward = select_centroid_diversity_exemplars(records, vectors)
    shuffled = select_centroid_diversity_exemplars(
        [records[index] for index in permutation],
        vectors[permutation],
    )

    expected = [f"sample_{episode}" for episode in range(5)]
    assert [row["sample_id"] for row in forward] == expected
    assert [row["sample_id"] for row in shuffled] == expected


def test_centroid3_maximin2_unique_episode_policy_fails_or_falls_back() -> None:
    records = [
        _record("ep0", 0),
        _record("ep1", 1),
        _record("ep2_a", 2, waypoint_step=0),
        _record("ep2_b", 2, waypoint_step=1),
        _record("ep2_c", 2, waypoint_step=2),
    ]
    vectors = np.ones((5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 5 unique episodes"):
        select_centroid_diversity_exemplars(records, vectors)

    selected = select_centroid_diversity_exemplars(
        records,
        vectors,
        insufficient_unique_episodes="fallback",
    )

    assert [row["sample_id"] for row in selected] == [
        "ep0",
        "ep1",
        "ep2_a",
        "ep2_b",
        "ep2_c",
    ]
    assert len({row["episode_num"] for row in selected[:3]}) == 3


def test_select_exemplars_legacy_default_and_opt_in_contract() -> None:
    records = [
        _record(f"sample_{episode}", episode)
        for episode in range(5)
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.95, 0.2],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    default_ids = [
        row["sample_id"]
        for row in select_exemplars(
            records,
            vectors,
            num_exemplars=3,
        )
    ]
    explicit_legacy_ids = [
        row["sample_id"]
        for row in select_exemplars(
            records,
            vectors,
            num_exemplars=3,
            strategy="centroid",
        )
    ]
    assert default_ids == explicit_legacy_ids

    selected = select_exemplars(
        records,
        vectors,
        num_exemplars=5,
        strategy=EXEMPLAR_SELECTION_CENTROID_DIVERSITY,
    )
    assert len(selected) == 5

    with pytest.raises(ValueError, match="requires num_exemplars=5"):
        select_exemplars(
            records,
            vectors,
            num_exemplars=4,
            strategy=EXEMPLAR_SELECTION_CENTROID_DIVERSITY,
        )

    with pytest.raises(ValueError, match="Unknown exemplar"):
        select_exemplars(
            records,
            vectors,
            num_exemplars=5,
            strategy="unknown",
        )


def test_centroid_five_is_stable_and_skips_duplicate_episodes() -> None:
    records = [
        _record("ep0_nearest", 0, waypoint_step=0),
        _record("ep0_second", 0, waypoint_step=1),
        _record("ep1", 1),
        _record("ep2", 2),
        _record("ep3", 3),
        _record("ep4", 4),
    ]
    vectors = np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.95, 0.05],
        ],
        dtype=np.float64,
    )
    permutation = [5, 2, 1, 4, 0, 3]

    selected, distances, raw_ranks = select_centroid_representatives(
        records,
        vectors,
    )
    shuffled, shuffled_distances, shuffled_raw_ranks = (
        select_centroid_representatives(
            [records[index] for index in permutation],
            vectors[permutation],
        )
    )

    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in shuffled
    ]
    assert len({row["episode_num"] for row in selected}) == 5
    assert sum(row["episode_num"] == 0 for row in selected) == 1
    assert distances == pytest.approx(shuffled_distances)
    assert raw_ranks == shuffled_raw_ranks
    assert distances == sorted(distances)


def test_centroid_five_fails_closed_without_five_episodes() -> None:
    records = [
        _record(f"sample_{index}", index // 2)
        for index in range(8)
    ]
    vectors = np.ones((8, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 5 unique episodes"):
        select_centroid_representatives(records, vectors)
