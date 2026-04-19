from itertools import product


def make_candidates(
    lines_a: list[str],
    lines_b: list[str],
    additional_diagonals: int = 2,
) -> list[tuple[int, int]]:
    """
    Return all `(i, j)` index pairs near the scaled diagonal of an (m x n) grid,
    sorted and deduplicated.

    In the alignment matrix, the 'diagonal' maps position `i` in lines_a to its
    proportionally equivalent position in lines_b. A band of 1 adds the
    immediate sub- and superdiagonal, which extends the search space to include
    1-to-1, 1-to-2, and 2-to-1 alignments.
    """
    m, n = len(lines_a), len(lines_b)
    if m == 0 or n == 0:
        return []

    candidates: set[tuple[int, int]] = set()

    # sweep along i and find proportional j
    for i in range(m):
        j_center = i * (n - 1) / (m - 1) if m > 1 else 0.0
        for dj in range(-additional_diagonals, additional_diagonals + 1):
            j = round(j_center) + dj
            if 0 <= j < n:
                candidates.add((i, j))

    # sweep along j and find proportional i (catches edges on the longer axis)
    for j in range(n):
        i_center = j * (m - 1) / (n - 1) if n > 1 else 0.0
        for di in range(-additional_diagonals, additional_diagonals + 1):
            i = round(i_center) + di
            if 0 <= i < m:
                candidates.add((i, j))

    return sorted(candidates)


def select_block_candidates(
    i_min: int, j_min: int, i_max: int, j_max: int
) -> list[tuple[int, int]]:
    len_i = i_max - i_min + 1
    len_j = j_max - j_min + 1
    required_proximity = max(3, min(len_i, len_j) // 2)

    # TODO
    product_ = product(range(i_min, i_max + 1), range(j_min, j_max + 1))
    candidates = [p for p in product_ if abs(p[0] - p[1]) < required_proximity]
    return candidates
