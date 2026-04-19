"""
Translation invariants are the words that are always translated with an equivalent word in the target language.
In other words, invariants are the word pairs that are always present (or present with some high probability `p`)
in a correct sentence pair. This does not require word-level alignment; only a consistent word tokenization,
because presence in a sentence is boolean: either the word is there, or it is not.

Invariant-based alignment uses pre-computed translation probabilities to align bitexts.

Scoring can follow acceptance-based or rejection-based criteria.

In acceptance-based scoring, we use each invariant word (for each direction) to compute
an answer to the question, "How probable is it that sentence pair (x, y) is a member of the correct alignment?"
Conversely, rejection-based criteria compute the probability that a sentence pair is NOT a member
of the correct alignment.

These two approaches can be used together, to successively eliminate the most obviously incorrect alignment
candidates, while adding the highest-probability candidates to the accepted alignment.

Note that the word sets used will not be the same. For example, a highly-inflected language such
as Russian will contain many words whose translation into a low-inflection language such as English
is nearly always the same English word, but the same will not be true in the EN->RU direction,
because the English word has many inflected variants. Each ordered language pair will need to have its
own scores computed.
"""

import math
from collections import defaultdict
from typing import Callable

type Tokenizer = Callable[[str], list[str]]

# outer key: word in language A.; inner key: word in language B
# value: (p_correct, p_random)
type InvariantScores = dict[str, dict[str, tuple[float, float]]]


def find_cooccurrence_candidates(
    data: list[tuple[str, str]],
    tokenizer_a: Tokenizer,
    tokenizer_b: Tokenizer,
    min_count: int = 5,
    min_pmi: float = 1.0,
) -> set[tuple[str, str]]:
    """
    Screen a bitext corpus for word pairs worth scoring as invariant candidates.

    A pair (w_a, w_b) passes the screen if:
        1. it co-occurs in at least `min_count` aligned sentence pairs
        2. its pointwise mutual information (PMI) exceeds `min_pmi`
    """
    n = len(data)
    if n == 0:
        return set()

    count_a: dict[str, int] = defaultdict(int)
    count_b: dict[str, int] = defaultdict(int)
    cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sent_a, sent_b in data:
        words_a = set(tokenizer_a(sent_a))
        words_b = set(tokenizer_b(sent_b))
        for w_a in words_a:
            count_a[w_a] += 1
        for w_b in words_b:
            count_b[w_b] += 1
        for w_a in words_a:
            for w_b in words_b:
                cooccur[w_a][w_b] += 1

    candidates: set[tuple[str, str]] = set()
    for w_a, w_b_counts in cooccur.items():
        for w_b, cnt in w_b_counts.items():
            if cnt < min_count:
                continue
            pmi = math.log((cnt * n) / (count_a[w_a] * count_b[w_b]))
            if pmi >= min_pmi:
                candidates.add((w_a, w_b))

    return candidates


def compute_scores(
    data: list[tuple[str, str]],
    tokenizer_a: Tokenizer,
    tokenizer_b: Tokenizer,
    candidates: set[tuple[str, str]] | None = None,  # None -> score everything
) -> InvariantScores:
    """
    Estimate invariant scores from a list of correctly-aligned sentence pairs.

    For each (w_a, w_b) pair:

        p_correct := (# aligned pairs where both w_a and w_b appear)
                    / (# aligned pairs where w_a appears)
        p_random  := (# pairs where w_b appears) / (# total pairs)

    `p_random` is the baseline: how often we would see `w_b` just by chance,
    regardless of what is in language A. A strong invariant has p_correct
    close to 1 and `p_random` close to 0.
    """
    n = len(data)
    if n == 0:
        return {}

    # sentence-level presence 'counts' (not strictly counts; presence is boolean)
    count_a: dict[str, int] = defaultdict(int)
    count_b: dict[str, int] = defaultdict(int)
    cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sent_a, sent_b in data:
        words_a = set(tokenizer_a(sent_a))
        words_b = set(tokenizer_b(sent_b))
        for w_a in words_a:
            count_a[w_a] += 1
            for w_b in words_b:
                cooccur[w_a][w_b] += 1
        for w_b in words_b:
            count_b[w_b] += 1

    scores: InvariantScores = {}
    for w_a, w_b_counts in cooccur.items():
        scores[w_a] = {}
        for w_b, cnt in w_b_counts.items():
            if candidates is not None and (w_a, w_b) not in candidates:
                continue
            p_correct = cnt / count_a[w_a]
            p_random = count_b[w_b] / n
            scores[w_a][w_b] = (p_correct, p_random)

    return scores


def compute_correctness_score(
    words_a: list[str],
    words_b: list[str],
    invariant_scores_a2b: InvariantScores,
    invariant_scores_b2a: InvariantScores,
) -> float:
    """ """
    set_a = set(words_a)
    set_b = set(words_b)

    total = 0.0
    count = 0

    def accumulate(scores: InvariantScores, present: set[str], other: set[str]) -> None:
        nonlocal total, count
        for w_src in present:
            if w_src not in scores:
                continue
            for w_tgt, (p_correct, p_random) in scores[w_src].items():
                if w_tgt not in other:
                    continue
                if p_correct <= 0 or p_random <= 0:
                    continue
                total += math.log(p_correct / p_random)
                count += 1

    accumulate(invariant_scores_a2b, set_a, set_b)
    accumulate(invariant_scores_b2a, set_b, set_a)

    return total / count if count > 0 else 0.0
