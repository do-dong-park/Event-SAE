# Event-SAE

Code for [Event-Grounded Sparse Autoencoders for Vision-Language-Action
Policies](https://arxiv.org/abs/2605.17204).

Event-SAE anchors sparse-autoencoder analysis in kinematic events from
closed-loop rollouts. It clusters recurring events, ranks SAE features around
them, optionally adds VLM labels for interpretation, and tests selected
features through residual-preserving interventions.

## Choose a workflow

| Workflow | Environment | Scope | Guide |
| --- | --- | --- | --- |
| OpenVLA | LIBERO | Paper pipeline, including intervention | [OpenVLA runbook](docs/backends/openvla.md) |
| OpenPI (π₀.₅) | LIBERO | Paper pipeline, including intervention | [OpenPI runbook](docs/backends/openpi.md) |
| GR00T N1.5 | RoboCasa PQ3 | Offline SAE, event, annotation, and phase-feature analysis | [GR00T status hub](docs/groot/README.md) |

The two paper backends share the offline pipeline but use backend-specific
activation collection and intervention hooks. GR00T currently stops at
feature analysis; no closed-loop GR00T intervention result is claimed.

## Pipeline

The paper workflow has 11 labeled steps (`a`–`k`).

| Stage | Steps | Output |
| --- | --- | --- |
| SAE training | a–b | Activation shards and trained SAE |
| Kinematic events | c | AWE waypoints per episode |
| Event interpretation | d–g | Media, descriptors, clusters, optional VLM labels |
| Feature ranking | h–j | Sparse activations, scores, candidate rankings |
| Intervention | k | Per-feature behavioral effect |

VLM labels are descriptive metadata in the paper pipeline; event timing and
SAE activations determine the numerical ranking. GR00T additionally uses
provisional phase labels for explicitly separated diagnostic analyses.

## Repository map

```text
event_sae/
├─ sae.py, train.py          shared SAE model and training
├─ keyframes/                kinematic event extraction
├─ events/                   media, descriptors, clustering, annotation
├─ scoring/                  sparse scoring, ranking, phase analysis
├─ openvla/, openpi/         backend collection and intervention
└─ groot/                    GR00T data and annotation adapters

scripts/                     command-line entry points
configs/examples/            backend example configurations
configs/groot/               GR00T semantic profiles and frozen contracts
docs/backends/               paper-backend runbooks
docs/groot/                  GR00T methods, lineage, analysis, and protocols
```

Historical artifact IDs such as `v9`, `v12r3`, and schema suffixes such as
`v1`/`v2` are provenance locators. New Python APIs use semantic names, while
stored identifiers remain unchanged.

## Extension contracts

Backbone-specific behavior stays at the data-source boundary:

- `train_sae(..., data=...)` consumes a validated activation iterable.
- `VisionEmbeddingProvider` supplies computed or exactly reused embeddings.
- `AnnotationProtocol` owns the model, vocabulary, and prompt policy.
- `TriptychFrameProvider` supplies aligned views to the shared renderer.
- `WaypointSelection` keeps composite anchors and provenance together.

Loaders validate source identities and refuse to overwrite historical
artifacts.

## Validation

Use the development environment for shared offline code. Backend rollouts use
the dedicated environments in their runbooks.

```bash
conda env create -f environment-sae-dev.yml
conda run --no-capture-output -n event-sae-dev \
  python -m pytest -q tests
```

GR00T's completed annotation runs are immutable historical artifacts. Their
exact execution sources are archived separately, while active Python APIs use
semantic names; see the [GR00T status hub](docs/groot/README.md).

## License

MIT (see `LICENSE`). Dependencies under `external/` retain their own licenses.
