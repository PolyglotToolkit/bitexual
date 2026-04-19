import numpy as np

from .types import Scores


def compute_spans(sentences: list[str]) -> list[tuple[float, float]]:
    """
    Map each sentence to a normalized [start, end] span in [0, 1]
    based on cumulative character length.

    TODO: support word/token length as well.
    """
    lengths: np.ndarray = np.array([len(s) for s in sentences], dtype=float)
    total: float = lengths.sum()
    if total == 0:
        return [(0.0, 0.0)] * len(sentences)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)]) / total
    return [(cumulative[i], cumulative[i + 1]) for i in range(len(sentences))]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap length between two spans, normalized by the smaller span."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0.0:
        return 0.0
    smaller = min(a_end - a_start, b_end - b_start)
    return intersection / smaller if smaller > 0 else 0.0


def get_weighted_candidates(
    sentences_a: list[str],
    sentences_b: list[str],
    min_weight: float = 0.01,
) -> tuple[Scores, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Return all (i, j) candidate pairs whose normalized spans overlap,
    weighted by the relative degree of overlap (normalized to shorter span).

    A weight of 1.0 means the smaller sentence is fully contained within
    the span of the larger. A weight near 0 means they barely touch.

    Args:
        src:        Source sentence list.
        tgt:        Target sentence list.
        min_weight: Discard pairs below this overlap threshold.

    Returns:
        List of WeightedCandidate(i, j, weight), sorted by (i, j).
    """
    spans_a = compute_spans(sentences_a)
    spans_b = compute_spans(sentences_b)

    candidates: Scores = {}

    # two-pointer sweep to avoid O(m * n) checks
    j_start = 0
    for i, (s0, s1) in enumerate(spans_a):
        # j_start should begin past spans that end before s0
        while j_start < len(spans_b) and spans_b[j_start][1] <= s0:
            j_start += 1

        # ... and end before s1
        for j in range(j_start, len(spans_b)):
            t0, t1 = spans_b[j]
            if t0 >= s1:
                break
            w = overlap(s0, s1, t0, t1)
            if w >= min_weight:
                candidates.update({(i, j): float(w)})

    return candidates, spans_a, spans_b
