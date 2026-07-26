"""CLI: build the four feature-ranking lists used as intervention candidates.

Outputs:
- ``event_aligned.jsonl``  — per-cluster top-N features (informational).
- ``window_mean.jsonl``    — per-cluster top-N features (informational).
- ``task_mean.jsonl``      — per-task top-N features (informational).
- ``random_alive.jsonl``   — K randomly sampled alive features (control).
- ``candidates.jsonl``     — flat list of 4*K suite-level features ready
                             for the intervention CLI (paper Section 5.3
                             top-K aggregation: K=5 for OpenVLA, K=3 for
                             pi_0.5).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.io import write_jsonl
from event_sae.scoring.rankings import (
    event_aligned_suite_top_k,
    event_aligned_top_features_per_row,
    random_alive_features,
    task_mean_suite_top_k,
    task_mean_top_features_per_task,
    window_mean_suite_top_k,
    window_mean_top_features_per_row,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 4 feature-ranking candidate lists.")
    ap.add_argument("--scores-pt", required=True, help="Path to event_feature_scores.pt from step (i).")
    ap.add_argument("--topk-run-dir", required=True, help="Directory with token_topk_sparse_v1 manifest + shards.")
    ap.add_argument("--output-dir", required=True, help="Where to write the JSONL outputs.")
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Suite-level top-K per ranking (paper: 5 for OpenVLA, 3 for pi_0.5). Default: 5.",
    )
    ap.add_argument(
        "--top-n-per-row",
        type=int,
        default=20,
        help="Per-row/per-task top-N for the informational JSONL outputs. Default: 20.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--step-mapping",
        choices=("auto", "action_executed", "chunk_executed", "inference_step"),
        default="auto",
        help="Mapping used to define the random-alive feature population.",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help=(
            "Canonical-row filter: include only score-matrix rows with "
            "episode_coverage >= this threshold in suite-level aggregation "
            "for event_aligned and window_mean. Matches mechanistic-steering-vlas "
            "filter_and_visualize_cluster_matrix.py default. Default: 0.5."
        ),
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] event_aligned …", flush=True)
    ea_rows = event_aligned_top_features_per_row(Path(args.scores_pt), args.top_n_per_row)
    ea_suite = event_aligned_suite_top_k(
        Path(args.scores_pt), args.top_k, min_coverage=args.min_coverage
    )
    write_jsonl(output_dir / "event_aligned.jsonl", ea_rows)

    print("[2/4] window_mean …", flush=True)
    # window_mean / task_mean read their pre-computed score-artifact matrices;
    # only random_alive scans topk_run_dir below.
    wm_rows = window_mean_top_features_per_row(
        scores_pt_path=Path(args.scores_pt),
        top_n=args.top_n_per_row,
    )
    wm_suite = window_mean_suite_top_k(
        scores_pt_path=Path(args.scores_pt),
        top_k=args.top_k,
        min_coverage=args.min_coverage,
    )
    write_jsonl(output_dir / "window_mean.jsonl", wm_rows)

    print("[3/4] task_mean …", flush=True)
    tm_rows = task_mean_top_features_per_task(
        scores_pt_path=Path(args.scores_pt),
        top_n=args.top_n_per_row,
    )
    tm_suite = task_mean_suite_top_k(
        scores_pt_path=Path(args.scores_pt),
        top_k=args.top_k,
        min_coverage=args.min_coverage,
    )
    write_jsonl(output_dir / "task_mean.jsonl", tm_rows)

    informed_ids: set[int] = set()
    for pairs in (ea_suite, wm_suite, tm_suite):
        informed_ids.update(int(p["feature_id"]) for p in pairs)

    print(f"[4/4] random_alive (excluding {len(informed_ids)} informed top-K features) …", flush=True)
    random_ids = random_alive_features(
        topk_run_dir=Path(args.topk_run_dir),
        num_features=args.top_k,
        exclude_feature_ids=informed_ids,
        seed=args.seed,
        step_mapping=args.step_mapping,
    )
    random_rows = [{"ranking": "random_alive", "feature_id": int(fid)} for fid in random_ids]
    write_jsonl(output_dir / "random_alive.jsonl", random_rows)

    candidates: list[dict] = []
    for rank, pair in enumerate(ea_suite):
        candidates.append({"ranking": "event_aligned", "rank": rank + 1, "feature_id": int(pair["feature_id"]), "score": pair["score"]})
    for rank, pair in enumerate(wm_suite):
        candidates.append({"ranking": "window_mean", "rank": rank + 1, "feature_id": int(pair["feature_id"]), "score": pair["score"]})
    for rank, pair in enumerate(tm_suite):
        candidates.append({"ranking": "task_mean", "rank": rank + 1, "feature_id": int(pair["feature_id"]), "score": pair["score"]})
    for rank, fid in enumerate(random_ids):
        candidates.append({"ranking": "random_alive", "rank": rank + 1, "feature_id": int(fid), "score": 0.0})
    candidates_path = output_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    (output_dir / "ranking_config.json").write_text(
        json.dumps(
            {
                "scores_pt": str(Path(args.scores_pt).resolve()),
                "topk_run_dir": str(Path(args.topk_run_dir).resolve()),
                "top_k": args.top_k,
                "top_n_per_row": args.top_n_per_row,
                "min_coverage": args.min_coverage,
                "seed": args.seed,
                "step_mapping": args.step_mapping,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "top_k_per_ranking": args.top_k,
                "event_aligned_rows": len(ea_rows),
                "window_mean_rows": len(wm_rows),
                "task_mean_rows": len(tm_rows),
                "random_alive_features": len(random_ids),
                "total_candidates": len(candidates),
                "candidates_path": str(candidates_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
