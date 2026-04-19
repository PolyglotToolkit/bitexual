from itertools import product
from typing import cast

import numpy as np

from .types import Pairs, Scores


def suggest_block_candidates() -> list[tuple[int, int]]: ...


def infer_holes(
    anchors: Pairs,
    probable: Pairs,
    unknowns: Pairs,
) -> Pairs:
    good = set(anchors) | set(probable)
    inferred: set[tuple[int, int]] = set()
    for i, j in unknowns:
        if ((i - 1, j - 1) in good) and ((i + 1, j + 1) in good):
            inferred.add((i, j))
    return sorted(good)


def select_alignments(
    scores: Scores,
    len_a: int,
    len_b: int,
    thresholds: tuple[float, float, float] = (0.9, 0.8, 0.2),
    auto_anchors: int = 5,
) -> np.ndarray:
    lower_threshold, middle_threshold, upper_threshold = sorted(thresholds)
    alignment: np.ndarray = np.zeros((len_a, len_b))
    max_distance = min(len_a, len_b) // 4
    for i, j in product(range(len_a), range(len_b)):
        if abs(i - j) > max_distance:
            alignment[i, j] = -1
    pairs = cast(Pairs, sorted(scores.keys(), key=scores.get, reverse=True))  # type: ignore
    anchors: Pairs = pairs[:auto_anchors]
    pairs = pairs[auto_anchors:]
    probable: Pairs = []
    unknowns: Pairs = []
    rejects: Pairs = []
    for pair in pairs:
        score = scores[pair]
        if score >= upper_threshold:
            anchors.append(pair)
        elif score >= middle_threshold:
            probable.append(pair)
        elif score >= lower_threshold:
            unknowns.append(pair)
        else:
            rejects.append(pair)
    good = infer_holes(anchors, probable, unknowns)
    probable += good
    print(anchors)
    for a, b in anchors:
        if alignment[a, b] == 0:
            alignment[a, :] = -1
            alignment[:, b] = -1
            alignment[a:, :b] = -1
            alignment[:a, b:] = -1
            alignment[a, b] = 2
        else:
            print(
                f"Conflict at {a}, {b}: alignment value {alignment[a, b]}; expected 0."
            )
    for a, b in rejects:
        if alignment[a, b] <= 0:
            alignment[a, b] = -1
        else:
            print(
                f"Conflict at {a}, {b}: alignment value {alignment[a, b]}; expected 0 or -1."
            )
    for a, b in probable:
        if alignment[a, b] == 0:
            alignment[a, b] = 1
    return alignment
