"""Deterministic inference helpers for GUI-grounding evaluation."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


def paired_sample_seed(sample: Mapping[str, Any], base_seed: int) -> int:
    """Return an order-independent seed shared by paired prompt protocols."""

    provenance = sample.get("provenance") or {}
    pairing_id = provenance.get("action_uid") or sample["sample_id"]
    payload = f"{base_seed}\x1f{pairing_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
