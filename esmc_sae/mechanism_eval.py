"""Deterministic traceability and optional reference metrics for Task 9."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_CLAIM_FIELDS = (
    "a_side_function",
    "b_side_function",
    "joint_vulnerability",
    "cell_context",
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2.0 * precision * recall / (precision + recall)


def _ngram_f1(predicted: list[str], reference: list[str], n: int) -> float:
    predicted_counts = Counter(
        tuple(predicted[index : index + n])
        for index in range(max(0, len(predicted) - n + 1))
    )
    reference_counts = Counter(
        tuple(reference[index : index + n])
        for index in range(max(0, len(reference) - n + 1))
    )
    overlap = sum((predicted_counts & reference_counts).values())
    return _f1(overlap, sum(predicted_counts.values()), sum(reference_counts.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _feature_set(value: object, label: str) -> set[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result: set[int] = set()
    for item in value:
        if type(item) is not int or item < 0:
            raise ValueError(f"{label} contains an invalid feature")
        result.add(item)
    return result


def evaluate_mechanism(
    report: str,
    packet: Mapping[str, object],
    reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate deterministic traceability and optional matched-reference text."""

    if not isinstance(report, str) or not report:
        raise ValueError("report must be non-empty text")
    if not isinstance(packet, Mapping):
        raise ValueError("packet must be a mapping")
    features = _feature_set(packet.get("model_features"), "model_features")
    mentioned = {
        feature_id
        for feature_id in features
        if re.search(rf"\bfeature\s+{feature_id}\b", report, re.IGNORECASE)
    }
    claim_count = len(_CLAIM_FIELDS)
    traced_claims = 0
    for field in _CLAIM_FIELDS:
        claim = packet.get(field)
        if isinstance(claim, Mapping) and claim.get("evidence_ids"):
            traced_claims += 1

    result: dict[str, object] = {
        "feature_traceability": (
            len(mentioned) / len(features) if features else 0.0
        ),
        "claim_traceability": traced_claims / claim_count,
        "counterevidence_present": int(
            isinstance(packet.get("counterevidence"), Mapping)
            and bool(packet["counterevidence"].get("text"))
        ),
        "context_present": int(
            isinstance(packet.get("cell_context"), Mapping)
            and bool(packet["cell_context"].get("text"))
        ),
        "testable_prediction_present": int(
            isinstance(packet.get("testable_prediction"), str)
            and bool(packet["testable_prediction"].strip())
        ),
        "expert_rubric": {
            "factual_consistency": None,
            "pair_specificity": None,
            "causal_chain_completeness": None,
            "uncertainty_calibration": None,
            "needs_adjudication": None,
            "second_reviewer_required": None,
        },
        "reference_metrics": None,
    }
    if reference is None:
        return result
    if not isinstance(reference, Mapping):
        raise ValueError("reference must be a mapping")
    if reference.get("row_id") != packet.get("row_id"):
        raise ValueError("reference row_id does not match packet")
    reference_text = reference.get("text")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("reference text must be non-empty")
    reference_features = _feature_set(
        reference.get("feature_ids"),
        "reference feature_ids",
    )
    predicted_tokens = _tokens(report)
    reference_tokens = _tokens(reference_text)
    shared_features = features & reference_features
    feature_precision = (
        len(shared_features) / len(features) if features else 0.0
    )
    feature_recall = (
        len(shared_features) / len(reference_features)
        if reference_features
        else 0.0
    )
    result["reference_metrics"] = {
        "rouge_1_f1": _ngram_f1(predicted_tokens, reference_tokens, 1),
        "rouge_2_f1": _ngram_f1(predicted_tokens, reference_tokens, 2),
        "rouge_l_f1": _f1(
            _lcs_length(predicted_tokens, reference_tokens),
            len(predicted_tokens),
            len(reference_tokens),
        ),
        "feature_precision": feature_precision,
        "feature_recall": feature_recall,
        "feature_f1": (
            2.0 * feature_precision * feature_recall
            / (feature_precision + feature_recall)
            if feature_precision + feature_recall
            else 0.0
        ),
    }
    return result
