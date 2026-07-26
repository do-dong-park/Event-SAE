"""Gemini VLM prompts for event-cluster annotation.

The paper-compatible cluster prompt and the clean-visual multiview prompt share
task-local phase vocabularies while keeping source and oracle provenance outside
the model request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


VocabularyResolver = Callable[
    [str],
    tuple[tuple[str, ...], dict[str, str]],
]
ContrastiveRuleResolver = Callable[[str], tuple[str, ...]]


@dataclass(frozen=True)
class AnnotationPromptSections:
    """Scheme-specific prompt fragments injected around the shared template."""

    assignment_guidance: tuple[str, ...] = ()
    phrase_guidance: tuple[str, ...] = (
        "Describe the main action or transition shared by the clips, using the "
        "earlier and later frames as temporal evidence.",
        "Prefer a brief action-focused verb phrase when an observable transition "
        "is shared across the clips.",
    )
    event_guidance: tuple[str, ...] = (
        "If the later frames make the event clearer than the earlier frames, "
        "prioritize the later frames when choosing the phrase and phase.",
    )
    vocabulary_guidance: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotationProtocol:
    """Closed vocabulary and provenance injected into the shared annotator."""

    scheme: str
    prompt_version: str
    default_model: str
    default_min_episode_coverage: float | None
    resolve_vocabulary: VocabularyResolver
    resolve_contrastive_rules: ContrastiveRuleResolver
    phase_labeler_provenance: Mapping[str, str]
    prompt_sections: AnnotationPromptSections = AnnotationPromptSections()


PAPER_ANNOTATION_PROMPT_ID = "gemini_task_local_event_cluster_png_v4"
# Existing paper-backend callers import this public name.
PROMPT_VERSION = PAPER_ANNOTATION_PROMPT_ID

SINGLE_VIEW_LEFT_LAYOUT = "single_view_left_sequence_v1"
MULTIVIEW_TRIPTYCH_LAYOUT = "synchronized_triptych_left_right_wrist_v1"
SEPARATE_MULTIVIEW_LAYOUT = "separate_synchronized_left_right_wrist_v12"
ANNOTATION_MEDIA_LAYOUTS = (
    SINGLE_VIEW_LEFT_LAYOUT,
    MULTIVIEW_TRIPTYCH_LAYOUT,
    SEPARATE_MULTIVIEW_LAYOUT,
)

PAPER_PHASE_LABELS = (
    "pre_grasp",
    "immobilization",
    "contact",
    "detach",
    "post_grasp",
    "transition",
)

PAPER_PHASE_DESCRIPTIONS = {
    "pre_grasp": "gripper is open and the robot has not yet firmly engaged the target object",
    "immobilization": "gripper is closing around the object but firm contact is not yet clearly established",
    "contact": "gripper has firm contact with the object and the object appears secured",
    "detach": "gripper is opening to release the object but is not yet fully open",
    "post_grasp": "gripper is fully open after the main grasp or release interaction has completed",
    "transition": "unclear, temporary, or ambiguous transition that does not cleanly fit the other labels",
}


PHASE_LABELS = PAPER_PHASE_LABELS
PHASE_DESCRIPTIONS = PAPER_PHASE_DESCRIPTIONS


def _resolve_paper_vocabulary(
    task_description: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    del task_description
    return PAPER_PHASE_LABELS, PAPER_PHASE_DESCRIPTIONS


def _no_contrastive_rules(task_description: str) -> tuple[str, ...]:
    del task_description
    return ()


PAPER_ANNOTATION_PROTOCOL = AnnotationProtocol(
    scheme="paper",
    prompt_version=PAPER_ANNOTATION_PROMPT_ID,
    default_model="gemini-2.5-flash",
    default_min_episode_coverage=None,
    resolve_vocabulary=_resolve_paper_vocabulary,
    resolve_contrastive_rules=_no_contrastive_rules,
    phase_labeler_provenance={},
)


ROBOCASA_CENTERED_EVENT_PROMPT_ID = (
    "gemini_task_local_visual_event_centered_v11"
)
ROBOCASA_REPRESENTATIVE_CLIP_PROMPT_ID = (
    "gemini_task_local_visual_representative_clip_v12r3_clean_visual"
)
ROBOCASA_PHASE_LABELER_PROVENANCE = {
    "repository": "pkt_ws/temporal_vla",
    "commit": "ea61d24ac312b555ca3bcc6f668463ae8f540f7b",
}

# These terms name per-rollout oracle state or implementation details.  They
# belong in provenance and offline audits, never in a Gemini request prompt.
ROBOCASA_CLEAN_VISUAL_PROMPT_FORBIDDEN_FRAGMENTS = (
    "oracle",
    "phase_labeler",
    "env_step",
    "event_steps",
    "phase_timeline",
    "feature_phases",
    "waypoint",
    "anchor_source",
    "qpos",
    "ground truth",
    "ground_truth",
    "simulator state",
    "progress",
    "episode",
    "success",
    "failure",
    "predicate",
    "threshold",
    "debounced",
    "latched",
    "_engaged",
    "_diseng_state",
)

PICK_PLACE_ACTION_PHASE_LABELS = (
    "reach-to-object",
    "grasp",
    "transport",
    "place",
    "insert-settle",
    "terminal",
    "wrong-grasp",
)

DRAWER_ACTION_PHASE_LABELS = (
    "reach-to-handle",
    "grasp-handle",
    "pull",
    "push-back",
    "disengage",
    "wrong-grasp",
    "open-done",
)

ROBOCASA_ACTION_PHASE_DESCRIPTIONS = {
    "reach-to-object": (
        "the target is not held and the gripper remains visibly far from the intended "
        "object; this is the far approach state"
    ),
    "grasp": (
        "the target is not yet held, but the gripper is visibly near the intended object; "
        "secure finger closure is not required for this proximity sub-phase"
    ),
    "transport": (
        "the intended target is visibly held away from the destination and is not yet "
        "placed or supported there; visible co-motion supports the held-state judgment "
        "but active target motion is not required at the temporal center"
    ),
    "place": (
        "the target is still held and visibly near or entering the destination, but is "
        "not yet clearly placed or supported there"
    ),
    "insert-settle": (
        "the target is already placed or supported in the destination while the gripper "
        "is still holding, touching, opening near, or only beginning to withdraw"
    ),
    "terminal": (
        "the target remains placed and the gripper is clearly far from it; active "
        "placement, settling, and near-object release are over"
    ),
    "wrong-grasp": (
        "the gripper visibly secures or moves a distractor instead of the intended target"
    ),
    "reach-to-handle": (
        "the gripper is visibly far from the intended drawer handle, typically while "
        "approaching or re-approaching it"
    ),
    "grasp-handle": (
        "the gripper is visibly near the intended handle before debounced drawer motion; "
        "secure visible contact or finger closure is not required"
    ),
    "pull": (
        "the drawer visibly moves outward in the intended opening direction; drawer "
        "motion determines this label even when current handle contact is visually unclear"
    ),
    "push-back": (
        "the drawer visibly moves inward, opposite to the intended opening direction"
    ),
    "disengage": (
        "after prior near-handle engagement or an opening attempt, the gripper visibly "
        "retreats from the handle while the drawer is not actively moving"
    ),
    "open-done": (
        "the drawer has reached the requested fully-open completion state; completion "
        "takes priority even when the gripper remains near the handle"
    ),
}

PICK_PLACE_CONTRASTIVE_RULES = (
    "Target not held and gripper visibly far from it -> reach-to-object.",
    "Target not held but gripper visibly near it -> grasp; do not require secure closure.",
    "Target visibly held away from the destination and not placed/supported -> transport; co-motion supports the held-state judgment but active motion is not required at the center.",
    "Still-held target is near or entering the destination but not yet supported -> place.",
    "Target is already placed/supported but gripper is not yet far -> insert-settle.",
    "Target is placed and the gripper is clearly far -> terminal.",
    "If the target is dropped before placement, classify the reverted current state as reach-to-object or grasp; phases are not monotone.",
    "A held target has priority over a simultaneous distractor contact; wrong-grasp requires target not held and a distractor held.",
    "Visible motion and contact evidence override apparent distance or expected task order.",
)

DRAWER_CONTRASTIVE_RULES = (
    "Gripper visibly far from the handle -> reach-to-handle.",
    "Gripper visibly near the handle before drawer motion -> grasp-handle; secure contact is not required.",
    "Drawer visibly moves in the requested opening direction -> pull.",
    "Drawer visibly moves inward -> push-back.",
    "After prior near-handle engagement, sustained retreat away from the handle -> disengage; re-approach can return to reach-to-handle.",
    "Fully-open completion -> open-done, even if the gripper is still near the handle.",
    "For nonterminal states use current-state priority: wrong-grasp, pull, push-back, near-handle grasp, disengage, then reach.",
    "Drawer phases can move backward after push-back or re-approach; do not force chronological monotonicity.",
    "Visible motion and contact evidence override apparent distance or expected task order.",
)


def resolve_robocasa_phase_vocabulary(
    task_description: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    task_lower = task_description.lower()
    if "drawer" in task_lower:
        labels = DRAWER_ACTION_PHASE_LABELS
    elif "pick" in task_lower and "place" in task_lower:
        labels = PICK_PLACE_ACTION_PHASE_LABELS
    else:
        raise ValueError(
            "Could not select a RoboCasa action-phase vocabulary for task: "
            f"{task_description!r}"
        )
    return labels, {
        label: ROBOCASA_ACTION_PHASE_DESCRIPTIONS[label]
        for label in labels
    }


def resolve_robocasa_contrastive_rules(
    task_description: str,
) -> tuple[str, ...]:
    if "drawer" in task_description.lower():
        return DRAWER_CONTRASTIVE_RULES
    resolve_robocasa_phase_vocabulary(task_description)
    return PICK_PLACE_CONTRASTIVE_RULES


REPRESENTATIVE_PHASE_DESCRIPTIONS = {
    **ROBOCASA_ACTION_PHASE_DESCRIPTIONS,
    "reach-to-object": (
        "the instruction-named object is not visibly held and the gripper is outside its "
        "immediate grasping interaction area at the temporal center; motion toward the "
        "object is not required"
    ),
    "grasp": (
        "the instruction-named object is not yet visibly established as held and the "
        "gripper is in its immediate grasping interaction area at the temporal center; "
        "final approach, alignment, closing, first contact, and an acquisition attempt "
        "before an established hold all belong here; secure closure is not required"
    ),
    "transport": (
        "the instruction-named object is visibly secured or held at the temporal center "
        "and remains away from the cabinet opening; target motion is not required"
    ),
    "place": (
        "the instruction-named object is visibly held at, near, or entering the cabinet "
        "opening at the temporal center but is not yet clearly inside the cabinet interior"
    ),
    "insert-settle": (
        "the instruction-named object is visibly inside the cabinet interior while the "
        "gripper remains nearby, touching it, opening nearby, or beginning to withdraw"
    ),
    "terminal": (
        "the instruction-named object is visibly inside the cabinet interior and the "
        "gripper is clearly far from it at the temporal center"
    ),
    "grasp-handle": (
        "the gripper is visibly in the immediate interaction area of the instructed drawer "
        "handle while the drawer is not visibly moving at the temporal center"
    ),
    "disengage": (
        "the gripper is not near the instructed handle at the temporal center, and the "
        "supplied frames visibly show earlier handle proximity followed by retreat or a "
        "withdrawn state without centered re-approach"
    ),
    "open-done": (
        "the instructed drawer is visibly at its fully-open end state at the temporal center"
    ),
}

PICK_PLACE_REPRESENTATIVE_DECISION_RULES = (
    "Judge the visible state at the temporal center; earlier and later frames are evidence, not different labeling targets.",
    "Track only the object named in the task instruction and the cabinet interior.",
    "Apply this decision order at the center: target inside the cabinet -> terminal if the gripper is clearly far, otherwise insert-settle.",
    "If the target is not inside but is visibly held -> place when at, near, or entering the cabinet opening, otherwise transport.",
    "If the target is not held but a distractor is visibly held -> wrong-grasp; a held target always takes priority over distractor contact.",
    "If neither object is held, use grasp whenever the gripper is in the target's immediate grasping area, aligning, closing, contacting, or acquiring; secure closure is not required.",
    "Use reach-to-object only when the target is not held and the gripper remains outside the immediate grasping area.",
    "Rigid object-gripper co-motion or persistent visible enclosure supports holding; a closed gripper beside a stationary object does not.",
    "Return unresolved only when an ambiguity needed to distinguish the remaining candidate phases cannot be resolved visually, such as target identity, near-versus-far interaction, held state, or cabinet entry; do not require evidence irrelevant to that boundary.",
    "Do not infer a later phase from an expected task order.",
)

DRAWER_REPRESENTATIVE_DECISION_RULES = (
    "Judge the visible state or motion at the temporal center; earlier and later frames are evidence, not different labeling targets.",
    "Track only the left or right drawer named in the task instruction and its corresponding handle.",
    "Centered motion means displacement spanning T3: compare T2->T3 and T3->T4; motion visible only in T1->T2 or T4->T5 is context.",
    "Use side-view displacement of the drawer front or handle relative to the fixed cabinet; wrist-camera motion alone is not drawer motion.",
    "Apply this decision order at T3: open-done overrides all other evidence; otherwise wrong-grasp, centered pull, centered push-back, grasp-handle, disengage, then reach-to-handle.",
    "Use grasp-handle when the gripper is in the correct handle's immediate interaction area at T3 and no higher-priority label applies; secure contact is not required.",
    "Use disengage only when the gripper is not near the handle at T3 and the supplied frames show earlier handle proximity followed by retreat or a withdrawn state without centered re-approach.",
    "Use reach-to-handle for a far gripper only when visible approach or re-approach supports it.",
    "If a far gripper could be either approach or post-interaction withdrawal and the supplied history cannot distinguish them, return unresolved with insufficient visibility.",
    "Otherwise return unresolved only when an ambiguity needed to distinguish the remaining candidate phases cannot be resolved visually, such as correct-drawer identity, centered motion direction, handle proximity, or loose-object identity; do not require evidence irrelevant to that boundary.",
    "Do not infer a later phase from an expected task order.",
)


def resolve_representative_phase_vocabulary(
    task_description: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    labels, _ = resolve_robocasa_phase_vocabulary(task_description)
    descriptions = {
        label: REPRESENTATIVE_PHASE_DESCRIPTIONS[label]
        for label in labels
    }
    if "drawer" in task_description.lower():
        descriptions["wrong-grasp"] = (
            "the gripper visibly secures or moves a loose object instead of "
            "interacting with the instructed drawer handle"
        )
    return labels, descriptions


def resolve_representative_decision_rules(
    task_description: str,
) -> tuple[str, ...]:
    if "drawer" in task_description.lower():
        return DRAWER_REPRESENTATIVE_DECISION_RULES
    resolve_robocasa_phase_vocabulary(task_description)
    return PICK_PLACE_REPRESENTATIVE_DECISION_RULES


ROBOCASA_ACTION_ANNOTATION_PROTOCOL = AnnotationProtocol(
    scheme="robocasa_action",
    prompt_version=ROBOCASA_CENTERED_EVENT_PROMPT_ID,
    default_model="gemini-3.1-pro-preview",
    default_min_episode_coverage=0.3,
    resolve_vocabulary=resolve_robocasa_phase_vocabulary,
    resolve_contrastive_rules=resolve_robocasa_contrastive_rules,
    phase_labeler_provenance=ROBOCASA_PHASE_LABELER_PROVENANCE,
    prompt_sections=AnnotationPromptSections(
        assignment_guidance=(
            "For each clip, determine the observable action phase at its temporal center.",
            "Use earlier and later frames only to disambiguate what is happening at "
            "that center; do not report the furthest phase reached elsewhere in the clip.",
            "The phrase and phase must describe that same center interpretation.",
        ),
        phrase_guidance=(
            "Describe the shared center event or state with a short action-focused phrase.",
            "Use a dynamic verb phrase only when an active transition crosses the "
            "temporal center. Otherwise, a concise current-state phrase is appropriate.",
            "Do not let a transition visible only before or after the center replace "
            "the center interpretation.",
        ),
        event_guidance=(
            "Use the full before-to-after window as evidence for whether contact, "
            "target motion, support, drawer motion, or retreat is active at the center.",
            "The operational labeler is state-based and non-monotone: drops, "
            "push-back, retreat, and re-approach can move a rollout back to an "
            "earlier label.",
            "Do not force the clips into a successful forward-only task order.",
            "Assign a cluster one phase only when that phase is the shared "
            "center-state interpretation across its representative clips.",
            "If representatives visibly disagree, choose the best shared phase "
            "conservatively; human review will mark mixed clusters ambiguous.",
        ),
        vocabulary_guidance=(
            "The phase names refer to observable visual events in these clips, not "
            "hidden simulator distances, predicates, or success thresholds.",
            "First identify target/distractor identity, held versus not-held state, "
            "target support, gripper distance, drawer motion direction, and retreat.",
            "Then apply the task-specific priority rules above and choose the "
            "matching current phase.",
        ),
    ),
)


def validate_clean_visual_annotation_prompt(
    prompt: str,
    *,
    forbidden_identifiers: tuple[str, ...] = (),
) -> None:
    """Refuse prompts containing oracle implementation text or source identifiers."""

    prompt_lower = prompt.lower()
    forbidden_fragments = [
        fragment
        for fragment in ROBOCASA_CLEAN_VISUAL_PROMPT_FORBIDDEN_FRAGMENTS
        if fragment in prompt_lower
    ]
    leaked_identifiers = [
        identifier
        for identifier in forbidden_identifiers
        if identifier and identifier in prompt
    ]
    if forbidden_fragments or leaked_identifiers:
        raise ValueError(
            "Clean visual annotation prompt contains forbidden request input: "
            f"fragments={forbidden_fragments}, "
            f"identifiers={leaked_identifiers}"
        )


def build_representative_clip_annotation_prompt(
    *,
    task_description: str,
    cluster_id: str,
    representative_index: int,
    num_frames: int,
    media_layout: str = SEPARATE_MULTIVIEW_LAYOUT,
) -> str:
    """Build one layout-specific representative annotation prompt."""

    # These identifiers are retained by the caller as audit metadata, but they
    # carry no visual semantics and must not perturb otherwise identical calls.
    del cluster_id, representative_index
    if media_layout not in (
        SINGLE_VIEW_LEFT_LAYOUT,
        SEPARATE_MULTIVIEW_LAYOUT,
    ):
        raise ValueError(f"Unknown representative clip media layout: {media_layout}")
    phase_labels, phase_descriptions = resolve_representative_phase_vocabulary(
        task_description
    )
    phase_lines = "\n".join(
        f'- "{phase}": {phase_descriptions[phase]}' for phase in phase_labels
    )
    decision_lines = "\n".join(
        f"- {rule}"
        for rule in resolve_representative_decision_rules(task_description)
    )
    center_index = num_frames // 2 + 1
    if "drawer" in task_description.lower():
        task_focus = (
            "Track the instruction-named drawer and its corresponding handle. "
            "Use the fixed cabinet as the reference for drawer motion."
        )
        temporal_evidence_guidance = (
            "drawer position and motion relative to the fixed cabinet, gripper-handle "
            "proximity, handle engagement, retreat, and re-approach"
        )
        phrase_guidance = """Task-specific phrase guidance:
- reach-to-handle: "approaching or re-approaching the drawer handle"
- grasp-handle: "near or aligned with the drawer handle"
- pull / push-back: describe the visible drawer motion, not wrist-camera motion
- disengage: "retreating or withdrawn from the drawer handle"
- open-done: "drawer at the fully-open end state"
- wrong-grasp: name the loose object held instead of the instructed handle
For labels other than wrong-grasp, use handle-specific wording."""
    else:
        task_focus = (
            "Track only the object named in the task instruction. "
            "The destination is the cabinet interior."
        )
        temporal_evidence_guidance = (
            "target identity, contact or held state, target-gripper co-motion, cabinet "
            "entry, gripper-target distance, and withdrawal"
        )
        phrase_guidance = """Task-specific phrase guidance:
- reach-to-object: "target not held; gripper outside the grasping area"
- grasp: "initiating a grasp on the target"
- transport: "holding the target away from the cabinet"
- place: "positioning or holding the target at/entering the cabinet opening"
- insert-settle: "target inside the cabinet with gripper still near"
- terminal: "target inside the cabinet; gripper withdrawn or far"
- wrong-grasp: explicitly name that a distractor, not the instructed target, is held"""
    if media_layout == SINGLE_VIEW_LEFT_LAYOUT:
        layout_guidance = """At each timestamp you will receive one LEFT image from a fixed
external camera. Time advances from T1 to T5. Do not assume finger closure, contact, or
object support when the single view does not show it."""
    else:
        layout_guidance = """At each timestamp you will receive three separate synchronized
images in this order: LEFT fixed external camera, RIGHT fixed external camera, then WRIST
camera mounted on the moving gripper. These are simultaneous views, not consecutive
times. Fuse the fixed views for scene motion with the wrist view for local interaction;
wrist-camera motion alone does not prove that an object or drawer moved."""
    prompt = f"""You are labeling ONE representative robot-manipulation clip from images.

Task instruction: "{task_description}"
{task_focus}

The clip has {num_frames} chronological timestamps, T1 through T5, sampled
approximately 0.1 seconds apart. T{center_index} is the temporal center.
{layout_guidance}

Determine the observable phase at the TEMPORAL CENTER only. Use earlier and later timestamps
only as evidence for {temporal_evidence_guidance}.
Do not label the furthest phase reached before or after the center.

Closed phase vocabulary:
{phase_lines}

Visual boundary guidance:
{decision_lines}

Visibility:
- clear: the decisive center-state evidence is visible.
- partial: some decisive evidence is partly occluded or indirect, but the chosen phase
  remains distinguishable from its adjacent alternatives.
- insufficient: the required distinction cannot be made visually from this clip.

{phrase_guidance}

Return exactly one JSON object with three keys:
{{
  "phrase": "short phrase describing the center state or transition",
  "phase": "exactly one label from the closed vocabulary, or unresolved",
  "visibility": "clear, partial, or insufficient"
}}

Requirements:
- Use only the task instruction and visible evidence in the supplied images.
- The phrase and phase must refer to the same temporal-center interpretation.
- If the required visual distinction is unavailable because of occlusion, weak motion, or
  missing history, return phrase="phase unresolved from visible evidence",
  phase="unresolved", and visibility="insufficient".
- Never use unresolved with clear or partial visibility.
- Do not guess from expected task order.
- Do not include markdown fences or any keys other than the three keys above.
"""
    validate_clean_visual_annotation_prompt(prompt)
    return prompt


def _render_prompt_lines(lines: tuple[str, ...]) -> str:
    return "\n".join(lines)


def build_cluster_annotation_prompt(
    *,
    task_description: str,
    cluster_id: str,
    num_sequences: int,
    num_frames_per_sequence: int,
    episode_coverage: float,
    media_layout: str | None = None,
    protocol: AnnotationProtocol = PAPER_ANNOTATION_PROTOCOL,
) -> str:
    phase_labels, phase_descriptions = protocol.resolve_vocabulary(task_description)
    phase_lines = "\n".join(
        f'- "{phase}": {phase_descriptions[phase]}' for phase in phase_labels
    )
    contrastive_rules = protocol.resolve_contrastive_rules(task_description)
    contrastive_lines = "\n".join(f"- {rule}" for rule in contrastive_rules)
    contrastive_section = (
        "Visual boundary rules:\n"
        f"{contrastive_lines}\n\n"
        if contrastive_lines
        else ""
    )
    layout_section = ""
    if media_layout is not None:
        if media_layout not in ANNOTATION_MEDIA_LAYOUTS:
            raise ValueError(f"Unknown annotation media layout: {media_layout}")
        if media_layout == SINGLE_VIEW_LEFT_LAYOUT:
            layout_section = (
                "Single-view image layout:\n"
                "- Every image is one robot0_agentview_left observation at one instant.\n"
                "- Time advances only from one image to the next within each clip.\n"
                "- Use only visibly supported evidence; an occluded finger closure or contact "
                "must not be invented from expected task order.\n"
                "- Object or drawer motion together with the gripper is stronger evidence than "
                "apparent proximity in a single frame.\n\n"
            )
        else:
            layout_section = (
                "Multi-view image layout:\n"
                "- Every image is one synchronized horizontal triptych of the same instant.\n"
                "- Left panel: robot0_agentview_left.\n"
                "- Middle panel: robot0_agentview_right.\n"
                "- Right panel: robot0_eye_in_hand wrist camera.\n"
                "The three panels are simultaneous views, not consecutive time steps. "
                "Time advances only from one triptych image to the next.\n"
                "Use the wrist view for visible finger closure and contact, and the side views "
                "for object or drawer motion and global context. Fuse evidence across views.\n"
                "Proximity alone supports only an approach or near-target interpretation. "
                "Do not infer that the target is held or transported unless it is visibly "
                "secured or moves together with the gripper.\n"
                "Wrist-camera motion by itself is not evidence that the target object moved.\n\n"
            )
    return f"""You are labeling a recurring event in a robot manipulation task.

You will be shown {num_sequences} image sequences from different rollout episodes of the same task.
Each sequence contains {num_frames_per_sequence} images in chronological order.
All sequences were clustered automatically and are intended to depict the same recurring event type.
Treat each sequence as one short clip. The model input is therefore a small batch of clips:
clip 1 = consecutive frames from one episode,
clip 2 = consecutive frames from another episode,
and so on.

{layout_section}

Task instruction: "{task_description}"
Cluster id: {cluster_id}
Episode coverage: {episode_coverage:.3f}
Your job is to assign one canonical phrase and one phase describing the best shared visual interpretation across the clips.
{_render_prompt_lines(protocol.prompt_sections.assignment_guidance)}

Focus on the common event across clips, not small differences between episodes.
Do not interpret the entire image list as one long timeline; instead, reason about each clip separately and then summarize the shared event.
{_render_prompt_lines(protocol.prompt_sections.phrase_guidance)}
{_render_prompt_lines(protocol.prompt_sections.event_guidance)}

Choose exactly one phase label from this closed set:
{phase_lines}

{contrastive_section}{_render_prompt_lines(protocol.prompt_sections.vocabulary_guidance)}

Return JSON with exactly two keys:
{{
  "phrase": "short canonical event phrase",
  "phase": "one label from the closed set above"
}}

Requirements:
- The top-level JSON value must be a single object, not a list or array.
- Output exactly one short human-readable phrase describing the common event across the sequences.
- The phrase should read naturally to a person.
- The phrase and phase must refer to the same shared visual interpretation.
- Do not invent contact, support, target motion, success, or task completion that is not visibly supported.
- The phase must be exactly one of the allowed labels above.
- Do not output multiple tags or a long explanation.
- Do not include any keys other than "phrase" and "phase".
- Do not include markdown fences.
"""
