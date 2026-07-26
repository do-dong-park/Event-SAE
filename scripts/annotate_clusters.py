"""CLI: Gemini VLM annotation of event clusters.

Reads `clusters.jsonl` from `scripts/cluster_events.py`, writes one
JSONL row per cluster with the assigned `phrase` + `phase`.

API key: `GEMINI_API_KEY` env var (preferred), or `--api-key-path` pointing
at a text file with the key.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.annotate import annotate_clusters, load_api_key
from event_sae.events.prompts import (
    ANNOTATION_MEDIA_LAYOUTS,
    PAPER_ANNOTATION_PROTOCOL,
    ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
)


ANNOTATION_PROTOCOLS = {
    "paper": PAPER_ANNOTATION_PROTOCOL,
    "robocasa_action": ROBOCASA_ACTION_ANNOTATION_PROTOCOL,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate event clusters with Gemini.")
    parser.add_argument("--clusters-path", required=True, help="Path to clusters.jsonl")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSONL path (default: derived from model, protocol, and selection)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name (default: selected by --phase-scheme)",
    )
    parser.add_argument(
        "--phase-scheme",
        choices=tuple(ANNOTATION_PROTOCOLS),
        default="paper",
        help="Annotation vocabulary and prompt policy",
    )
    parser.add_argument(
        "--api-key-path",
        default=None,
        help="Optional path to text file with API key (fallback if GEMINI_API_KEY env var unset)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request transport timeout; does not alter model generation",
    )
    parser.add_argument(
        "--media-layout",
        choices=ANNOTATION_MEDIA_LAYOUTS,
        default=None,
        help=(
            "Explicit annotation image layout; must match cluster metadata "
            "when both are present."
        ),
    )
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument(
        "--cluster-id",
        dest="cluster_ids",
        action="append",
        help="Annotate only this exact cluster id; repeat for multiple clusters",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--min-episode-coverage",
        type=float,
        default=None,
        help=(
            "Annotate clusters appearing in at least this fraction of task "
            "episodes (default: selected by --phase-scheme)"
        ),
    )
    selection_group.add_argument(
        "--all-clusters",
        action="store_true",
        help="Annotate all clusters; use only for supplemental rare-phase auditing",
    )
    args = parser.parse_args()
    protocol = ANNOTATION_PROTOCOLS[args.phase_scheme]
    model = args.model or protocol.default_model
    min_episode_coverage = (
        None
        if args.all_clusters
        else (
            args.min_episode_coverage
            if args.min_episode_coverage is not None
            else protocol.default_min_episode_coverage
        )
    )

    if args.cluster_ids:
        selection_tag = f"selected{len(args.cluster_ids)}"
    elif min_episode_coverage is None:
        selection_tag = "all"
    else:
        selection_tag = f"cov{min_episode_coverage:g}".replace(".", "p")
    clusters_path = Path(args.clusters_path).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else clusters_path.with_name(
            (
                f"{model.replace('.', '_')}_cluster_annotations.jsonl"
                if args.phase_scheme == "paper" and selection_tag == "all"
                else (
                    f"{model.replace('.', '_')}_{args.phase_scheme}_"
                    f"{selection_tag}_cluster_annotations.jsonl"
                )
            )
        ).resolve()
    )

    api_key = load_api_key(Path(args.api_key_path) if args.api_key_path else None)

    annotate_clusters(
        clusters_path=clusters_path,
        output_path=output_path,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_clusters=args.max_clusters,
        cluster_ids=tuple(args.cluster_ids) if args.cluster_ids else None,
        min_episode_coverage=min_episode_coverage,
        protocol=protocol,
        media_layout_override=args.media_layout,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    print(f"Clusters: {clusters_path}")
    print(f"Annotations: {output_path}")


if __name__ == "__main__":
    main()
