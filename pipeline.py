"""
Financial Spreading Pipeline — Conflict Resolution & Audit Trail

Implement the four functions below. Helper types are provided; you may add
more but do not change the existing signatures or field names.

Allowed: standard library + pydantic (v2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types — DO NOT change field names or remove fields. You may add fields.
# ---------------------------------------------------------------------------


@dataclass
class SourceLineItem:
    """A single line item as extracted by one source."""

    id: str
    label: str
    value: float
    confidence: float
    page: int
    bounding_box: dict[str, float]
    source: str  # "llm" or "ocr"
    section_hint: str = ""
    method: str = ""


@dataclass
class MergedLineItem:
    """A reconciled line item combining evidence from one or both sources."""

    label: str  # canonical label you chose
    value: float  # reconciled value
    confidence: float  # your assessed confidence
    sources: list[SourceLineItem]  # original items that contributed
    has_conflict: bool = False  # True when sources disagreed
    conflict_type: str | None = None  # e.g. "value_mismatch", "label_only", "single_source"
    resolution_note: str = ""  # human-readable explanation of how you resolved it
    flagged: bool = False  # True if item should be reviewed by a human


@dataclass
class AccountMapping:
    """A target account with its mapped value and contributing line items."""

    account_id: str
    account_name: str
    value: float
    mapped_items: list[MergedLineItem] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of checking one validation rule."""

    rule_id: str
    rule_name: str
    passed: bool
    expected: float  # value of the 'equals' account
    actual: float  # computed value from add/subtract
    gap: float  # actual - expected (0.0 if passed)


@dataclass
class ProvenanceRecord:
    """Full audit trail for one account."""

    account_id: str
    account_name: str
    final_value: float
    chain: list[dict[str, Any]]
    # Each entry in chain should contain at minimum:
    # {
    #   "merged_label": str,
    #   "merged_value": float,
    #   "merge_decision": str,
    #   "source_values": [
    #       {"source": str, "id": str, "value": float,
    #        "page": int, "bounding_box": dict}
    #   ]
    # }


# ---------------------------------------------------------------------------
# Data loading helpers — provided, no changes needed.
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def load_source(path: Path) -> list[SourceLineItem]:
    """Load a source JSON file and return a list of SourceLineItems."""
    raw = json.loads(path.read_text())
    source_name = raw["source"]
    items: list[SourceLineItem] = []
    for li in raw["line_items"]:
        items.append(
            SourceLineItem(
                id=li["id"],
                label=li["label"],
                value=li["value"],
                confidence=li["confidence"],
                page=li["page"],
                bounding_box=li["bounding_box"],
                source=source_name,
                section_hint=li.get("section_hint", ""),
                method=li.get("method", ""),
            )
        )
    return items


def load_taxonomy(path: Path) -> list[dict[str, Any]]:
    """Load the taxonomy JSON and return the accounts list."""
    raw = json.loads(path.read_text())
    return raw["accounts"]


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Load the validation rules JSON and return the rules list."""
    raw = json.loads(path.read_text())
    return raw["rules"]


# ---------------------------------------------------------------------------
# Implement these four functions.
# ---------------------------------------------------------------------------


def reconcile(
    source_a: list[SourceLineItem],
    source_b: list[SourceLineItem],
) -> list[MergedLineItem]:
    """
    Merge two extraction sources into a single reconciled list.

    Responsibilities:
    - Match items across sources (by label similarity, value proximity, or both).
    - When both sources found the same item but disagree on value, resolve the
      conflict and record how in ``resolution_note``.
    - Include items found in only one source with appropriate flagging.
    - Flag suspicious items (possible hallucinations, double-count risks).

    Returns:
        List of MergedLineItem, one per unique financial concept identified.
    """
    # TODO: Implement your reconciliation logic.
    raise NotImplementedError


def map_to_accounts(
    merged: list[MergedLineItem],
    taxonomy: list[dict[str, Any]],
) -> list[AccountMapping]:
    """
    Map reconciled line items to target accounts from the taxonomy.

    Responsibilities:
    - Each leaf-account line item maps to exactly one account.
    - Total/computed line items map to their corresponding total accounts.
    - Handle ambiguous labels (e.g., "Payroll - Production" → COGS_LABOR,
      not SGA, because the section hint says "Cost of Goods Sold").
    - Avoid double-counting: if both a total and its children are present,
      do not map children into a parent account that already has the total.
    - Return an AccountMapping for every account that received at least one
      line item. Accounts with no mapped items may be omitted or included
      with value=0 — your choice.

    Returns:
        List of AccountMapping, one per account that has mapped items.
    """
    # TODO: Implement your mapping logic.
    raise NotImplementedError


def validate(
    mappings: list[AccountMapping],
    rules: list[dict[str, Any]],
) -> list[ValidationResult]:
    """
    Check mathematical constraints against the mapped account values.

    Each rule has:
        - "add": list of account IDs to sum
        - "subtract": list of account IDs to subtract
        - "equals": the account ID whose value should match
        - "tolerance": allowed absolute difference

    The check is:  |sum(add) - sum(subtract) - value(equals)| <= tolerance

    Returns:
        List of ValidationResult, one per rule.
    """
    # TODO: Implement validation logic.
    raise NotImplementedError


def audit_trail(
    mappings: list[AccountMapping],
) -> list[ProvenanceRecord]:
    """
    Produce an audit trail linking every mapped account back to source evidence.

    For each account with a non-zero value, produce a ProvenanceRecord whose
    ``chain`` list contains one entry per contributing MergedLineItem. Each
    entry must trace back to the original source extractions including page
    numbers and bounding boxes.

    Returns:
        List of ProvenanceRecord, one per account with mapped items.
    """
    # TODO: Implement audit trail generation.
    raise NotImplementedError
