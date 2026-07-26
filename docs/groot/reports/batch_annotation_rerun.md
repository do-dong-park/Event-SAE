# Gemini Batch annotation — compatibility locator

This path is retained because the frozen controlled-ablation protocol links to
it. The authoritative execution ledger is consolidated in
[Annotation lineage §8–9](annotation_lineage.md#8-batch-source-and-normalized-child).

Key facts:

| Run | Result | Role |
| --- | --- | --- |
| Source Batch v1 | 233/270 initial valid, 37 exhausted, materialized 0/90 | Immutable fail-closed history |
| Singleton-normalized child v2 | Initial 270 + expanded 136, materialized 90/90 | Complete audited replacement; not integrated |
| Centroid-nearest-five | 450/450 responses | Alternative representative policy |
| Unique-plurality derivation | 72/90 accepted | E4-only inferential sensitivity |

None of these roots overwrites V11, the V12r3 interactive partial, or another
Batch root. Human review remains `0/90`, so completion means transport and
artifact completeness rather than semantic correctness.

The exact Batch runtime is preserved in
`archives/batch_annotation_runtime_20260725.tar` (SHA-256
`3a07e51c5d84846b6d5a7110d6f30f98259aaff6aef4b8b66bb8047e90d7a9ae`).
The completed interactive + Batch execution sources, manifests, and summaries
are preserved in `archives/annotation_lineage_source_20260725.tar` (SHA-256
`71d6ddf9395c5bb41ec3e1a0d455dc55fcdf925ade75e0ecb15ebb441411543d`).
Both paths are relative to
`logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/`. Active code
no longer resumes these historical roots.
