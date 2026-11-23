"""Use-scoped dataset approvals for reproducible speech experiments.

An open license is necessary but not sufficient: a source may be suitable for a
monolingual control while being invalid for the Tamil-English code-switch test.
These contracts keep that distinction machine-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class DatasetUse(StrEnum):
    """Experiment roles that need separate governance decisions."""

    MONOLINGUAL_CONTROL = "monolingual_control"
    CODE_SWITCH_TRAINING = "code_switch_training"
    CODE_SWITCH_EVALUATION = "code_switch_evaluation"
    LOCKED_TEST = "locked_test"


class ApprovalDecision(StrEnum):
    """Outcome of a dated, use-scoped review."""

    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DatasetApproval:
    """Auditable approval record; it never implies access to the underlying data."""

    dataset_id: str
    version: str
    reviewed_on: date
    evidence_uris: tuple[str, ...]
    decision: ApprovalDecision
    permitted_uses: frozenset[DatasetUse]
    conditions: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        for name in ("dataset_id", "version", "rationale"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and trimmed")
        if not self.evidence_uris or any(not uri.strip() for uri in self.evidence_uris):
            raise ValueError("at least one non-empty evidence URI is required")
        if len(set(self.evidence_uris)) != len(self.evidence_uris):
            raise ValueError("evidence URIs must be unique")
        if self.decision is ApprovalDecision.APPROVED and not self.permitted_uses:
            raise ValueError("an approved decision requires at least one permitted use")
        if self.decision is not ApprovalDecision.APPROVED and self.permitted_uses:
            raise ValueError("only approved decisions may grant permitted uses")
        if self.decision is ApprovalDecision.CONDITIONAL and not self.conditions:
            raise ValueError("a conditional decision must state its unmet conditions")

    def require(self, intended_use: DatasetUse) -> None:
        """Fail closed unless this exact dataset version is approved for the role."""

        if self.decision is not ApprovalDecision.APPROVED:
            raise DatasetUseNotApproved(
                f"{self.dataset_id}@{self.version} is {self.decision.value}, not approved"
            )
        if intended_use not in self.permitted_uses:
            raise DatasetUseNotApproved(
                f"{self.dataset_id}@{self.version} is not approved for {intended_use.value}"
            )


class DatasetUseNotApproved(ValueError):
    """Raised when a dataset is used outside its reviewed scope."""


def require_approved_dataset(
    approvals: tuple[DatasetApproval, ...],
    *,
    dataset_id: str,
    version: str,
    intended_use: DatasetUse,
) -> DatasetApproval:
    """Select one exact approval and reject missing or duplicate review records."""

    matches = tuple(
        approval
        for approval in approvals
        if approval.dataset_id == dataset_id and approval.version == version
    )
    if len(matches) != 1:
        raise DatasetUseNotApproved(
            f"expected exactly one approval for {dataset_id}@{version}; found {len(matches)}"
        )
    matches[0].require(intended_use)
    return matches[0]
