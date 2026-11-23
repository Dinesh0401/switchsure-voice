"""Mathematical utilities used by SwitchSure ML adapters."""

from .uncertainty import (
    conformal_quantile,
    normalized_entropy,
    pairwise_sequence_disagreement,
    prediction_set,
    predictive_mutual_information,
)

__all__ = [
    "conformal_quantile",
    "normalized_entropy",
    "pairwise_sequence_disagreement",
    "prediction_set",
    "predictive_mutual_information",
]
