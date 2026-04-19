import torch
from sentence_transformers import SentenceTransformer

from typing import cast

from sentence_transformers import util


def get_embeddings(model: SentenceTransformer, lines: list[str]) -> torch.Tensor:
    return cast(torch.Tensor, model.encode(lines, convert_to_tensor=True))  # type: ignore


def get_candidate_scores(
    candidates: list[tuple[int, int]],
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for idx_a, idx_b in candidates:
        similarity = util.cos_sim(embeddings_a[idx_a], embeddings_b[idx_b])  # type: ignore
        scores.update({(idx_a, idx_b): float(similarity[0][0])})
    return scores
