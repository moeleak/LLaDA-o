"""Shared Mind2Web prompt and action protocol helpers.

The public Multimodal-Mind2Web rows contain both a high-level task/trajectory
and ``target_action_reprs``, a human-readable description of the *specific*
element acted on at that step.  GUI grounding evaluates the latter; asking a
model to infer the next step from the former additionally measures planning.

The LLaDA-V paper does not publish its exact prompt template.  The
``target_grounding`` template below follows the direct imperative shown in the
paper (for example, ``Click on Track & Field.``) and keeps the legacy
``task_history`` prompt available as an explicitly different protocol.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


TARGET_GROUNDING = "target_grounding"
TASK_HISTORY = "task_history"
MIND2WEB_PROMPT_PROTOCOLS = (TARGET_GROUNDING, TASK_HISTORY)

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PASSWORD_RE = re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;]+)")
_ROLE_RE = re.compile(r"^\s*\[([^\]]+)]\s*(.*)$", re.DOTALL)
_OPERATION_RE = re.compile(
    r"^\s*(CLICK|HOVER|TYPE|SELECT|ENTER)\s*(?::\s*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)


def compact_mind2web_text(value: Any, limit: int = 2_000) -> str:
    """Normalize user-visible text and redact common accidental PII."""

    text = " ".join(str(value or "").replace("\x00", " ").split())
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PASSWORD_RE.sub(r"\1<REDACTED>", text)
    return text[:limit]


def _target_repr_text(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = next((item for item in reversed(value) if str(item or "").strip()), "")
    return compact_mind2web_text(value)


def canonical_action(operation: str) -> str:
    """Map Mind2Web operations to the three actions used by LLaDA-V."""

    operation = str(operation or "CLICK").upper()
    if operation == "HOVER":
        return "hover"
    if operation in {"TYPE", "SELECT"}:
        # LLaDA-V exposes no separate select action.  Mapping SELECT to type_in
        # preserves the selected value in the generated action string and is
        # the mapping used by this repository's original conversion.
        return "type_in"
    return "lclick"


def mind2web_crop_plan(
    sample_ids: Sequence[str], target_count: int, seed: int
) -> list[tuple[str, int]]:
    """Allocate a balanced, deterministic set of crop variants.

    Each source target is used once before receiving a second crop, and so on.
    The seeded hash makes selection independent of Parquet row/file order.
    """

    if target_count <= 0:
        raise ValueError("Mind2Web target_count must be positive")
    normalized = [str(sample_id) for sample_id in sample_ids]
    if not normalized:
        raise ValueError("Mind2Web crop planning requires at least one source row")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Mind2Web source sample IDs must be unique")

    def selection_key(sample_id: str) -> tuple[int, str]:
        payload = f"{seed}\x1fmind2web-selection\x1f{sample_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big"), sample_id

    ordered = sorted(normalized, key=selection_key)
    return [
        (ordered[index % len(ordered)], index // len(ordered))
        for index in range(target_count)
    ]


@dataclass(frozen=True)
class Mind2WebTarget:
    action: str
    operation: str
    value: str
    description: str
    role: str
    raw_target_action_repr: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_target_action(
    target_action_repr: Any,
    operation: Mapping[str, Any],
    *,
    fallback_description: Any = "",
) -> Mind2WebTarget:
    """Parse one public Mind2Web target description into a grounding target.

    Typical source values look like ``[button] Submit -> CLICK`` or
    ``[textbox] Search -> TYPE: shoes``.  The structured ``operation`` column
    remains authoritative when the two fields disagree.
    """

    raw = _target_repr_text(target_action_repr)
    description_part = raw
    repr_operation = ""
    repr_value = ""
    if "->" in raw:
        description_part, operation_part = raw.rsplit("->", 1)
        match = _OPERATION_RE.match(operation_part)
        if match:
            repr_operation = match.group(1).upper()
            repr_value = compact_mind2web_text(match.group(2), 512)

    role = ""
    description_part = compact_mind2web_text(description_part)
    role_match = _ROLE_RE.match(description_part)
    if role_match:
        role = compact_mind2web_text(role_match.group(1), 128)
        description_part = compact_mind2web_text(role_match.group(2))

    description = description_part or compact_mind2web_text(fallback_description)
    source_operation = str(
        operation.get("original_op") or operation.get("op") or repr_operation or "CLICK"
    ).upper()
    value = compact_mind2web_text(operation.get("value") or repr_value, 512)

    if not description:
        description = f"the {role}" if role else "the target UI element"

    return Mind2WebTarget(
        action=canonical_action(source_operation),
        operation=source_operation,
        value=value,
        description=description,
        role=role,
        raw_target_action_repr=raw,
    )


def target_grounding_prompt(target: Mind2WebTarget) -> str:
    """Build a target-explicit, single-step grounding instruction."""

    description = target.description.rstrip(" .")
    value = target.value
    if target.operation == "HOVER":
        return f"Hover over {description}."
    if target.operation == "TYPE":
        return f'Type "{value}" into {description}.' if value else f"Type into {description}."
    if target.operation == "SELECT":
        return (
            f'Select "{value}" from {description}.'
            if value
            else f"Select an option from {description}."
        )
    if target.operation == "ENTER":
        return f"Press Enter on {description}."
    return f"Click on {description}."


def task_history_prompt(
    confirmed_task: Any,
    action_reprs: Sequence[Any] | None,
    target_action_index: Any,
) -> str:
    """Build the legacy agent-style next-action prompt for diagnostic A/B."""

    actions = list(action_reprs or [])
    try:
        target_index = int(target_action_index)
    except (TypeError, ValueError):
        target_index = max(0, len(actions) - 1)
    previous = [compact_mind2web_text(item, 300) for item in actions[:target_index]][-8:]
    prompt = "Complete the following web task by predicting the next GUI action.\n"
    prompt += f"Task: {compact_mind2web_text(confirmed_task, 1_000)}"
    if previous:
        prompt += "\nPrevious actions:\n" + "\n".join(f"- {item}" for item in previous)
    return prompt


def mind2web_prompt(
    protocol: str,
    target: Mind2WebTarget,
    *,
    confirmed_task: Any = "",
    action_reprs: Sequence[Any] | None = None,
    target_action_index: Any = 0,
) -> str:
    if protocol == TARGET_GROUNDING:
        return target_grounding_prompt(target)
    if protocol == TASK_HISTORY:
        return task_history_prompt(confirmed_task, action_reprs, target_action_index)
    raise ValueError(
        f"unknown Mind2Web prompt protocol {protocol!r}; "
        f"expected one of {MIND2WEB_PROMPT_PROTOCOLS}"
    )
