from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
from switchsure_ml.data_governance import (
    ApprovalDecision,
    DatasetApproval,
    DatasetUse,
    DatasetUseNotApproved,
    require_approved_dataset,
)


def _approval() -> DatasetApproval:
    return DatasetApproval(
        dataset_id="common-voice-ta",
        version="27.0-2026-09-11",
        reviewed_on=date(2026, 11, 23),
        evidence_uris=("https://example.test/datasheet",),
        decision=ApprovalDecision.APPROVED,
        permitted_uses=frozenset({DatasetUse.MONOLINGUAL_CONTROL}),
        rationale="Exact release is suitable for a real-speech Tamil control.",
    )


def test_approval_is_scoped_to_an_exact_intended_use() -> None:
    approval = _approval()

    approval.require(DatasetUse.MONOLINGUAL_CONTROL)
    with pytest.raises(DatasetUseNotApproved, match="code_switch_evaluation"):
        approval.require(DatasetUse.CODE_SWITCH_EVALUATION)


def test_conditional_review_cannot_silently_grant_access() -> None:
    conditional = replace(
        _approval(),
        decision=ApprovalDecision.CONDITIONAL,
        permitted_uses=frozenset(),
        conditions=("Obtain the signed data-use agreement.",),
    )

    with pytest.raises(DatasetUseNotApproved, match="conditional"):
        conditional.require(DatasetUse.MONOLINGUAL_CONTROL)


def test_invalid_review_records_fail_closed() -> None:
    with pytest.raises(ValueError, match="permitted use"):
        replace(_approval(), permitted_uses=frozenset())
    with pytest.raises(ValueError, match="unmet conditions"):
        replace(
            _approval(),
            decision=ApprovalDecision.CONDITIONAL,
            permitted_uses=frozenset(),
        )
    with pytest.raises(ValueError, match="only approved"):
        replace(_approval(), decision=ApprovalDecision.REJECTED)


def test_registry_lookup_requires_one_exact_version_record() -> None:
    approval = _approval()

    assert (
        require_approved_dataset(
            (approval,),
            dataset_id=approval.dataset_id,
            version=approval.version,
            intended_use=DatasetUse.MONOLINGUAL_CONTROL,
        )
        is approval
    )
    with pytest.raises(DatasetUseNotApproved, match="found 0"):
        require_approved_dataset(
            (approval,),
            dataset_id=approval.dataset_id,
            version="wrong-version",
            intended_use=DatasetUse.MONOLINGUAL_CONTROL,
        )
    with pytest.raises(DatasetUseNotApproved, match="found 2"):
        require_approved_dataset(
            (approval, approval),
            dataset_id=approval.dataset_id,
            version=approval.version,
            intended_use=DatasetUse.MONOLINGUAL_CONTROL,
        )
