from dataclasses import dataclass

import torch
from sentence_transformers import util

from .types import Language, Model
from .utils import simple_tokenize


@dataclass
class SentenceComparison:
    a: str
    b: str
    idx_a: int
    idx_b: int
    relative_overlap: float
    char_weight_a: tuple[float, float]
    char_weight_b: tuple[float, float]

    # TODO
    # token_weight_a: tuple[float, float]
    # token_weight_b: tuple[float, float]
    # weight_ratio: float

    @property
    def punctuation_matches(self) -> bool:
        return self.punctuation_normalized(self.a) == self.punctuation_normalized(
            self.b
        )

    def punctuation_normalized(self, s: str) -> str:
        if s.strip()[-1] in {"!", "!"}:
            return "!"
        if s.strip()[-1] in {"?", "؟"}:
            return "?"
        if s.strip()[1] in {":", ":"}:
            return ":"
        if s.strip()[-1] in {".", "."}:
            return "."
        if s.strip()[-1] in {",", "،"}:
            return ","
        return ""

    @property
    def punctuation_a(self) -> str:
        return self.punctuation_normalized(self.a)

    @property
    def punctuation_b(self) -> str:
        return self.punctuation_normalized(self.b)

    # TODO
    # @property
    # def token_weight_ratio(self) -> float:
    #     return self._compute_ratio(
    #         self._diff(self.token_weight_a),
    #         self._diff(self.token_weight_b),
    #     )

    @property
    def char_weight_ratio(self) -> float:
        return self._compute_ratio(
            self._diff(self.char_weight_a),
            self._diff(self.char_weight_b),
        )

    @staticmethod
    def _compute_ratio(x: float, y: float) -> float:
        small, large = sorted([x, y])
        if large == 0.0:
            raise ValueError
        return small / large

    @staticmethod
    def _diff(t: tuple[float, float]) -> float:
        small, large = sorted(t)
        return large - small

    @property
    def tokens_a(self) -> list[str]:
        return simple_tokenize(self.a)

    @property
    def tokens_b(self) -> list[str]:
        return simple_tokenize(self.b)


@dataclass
class WeightedCandidate:
    i: int
    j: int
    weight: float  # overlap in [0, 1]


@dataclass
class DataContainer:
    language_a: Language
    language_b: Language
    model: Model
    original_a: str
    original_b: str
    sentences_a: list[str]
    sentences_b: list[str]

    translations_a_b: list[str]
    translations_b_a: list[str]
    translations_a_b_alt: list[str] | None = None
    translations_b_a_alt: list[str] | None = None
    translations_a_en: list[str] | None = None
    translations_b_en: list[str] | None = None

    embeddings_minilm_a: torch.Tensor | None = None
    embeddings_minilm_b: torch.Tensor | None = None
    embeddings_mpnet_a: torch.Tensor | None = None
    embeddings_mpnet_b: torch.Tensor | None = None
    embeddings_labse_a: torch.Tensor | None = None
    embeddings_labse_b: torch.Tensor | None = None

    def score(self, idx_a: int, idx_b: int) -> float: ...

    def score_minilm(self, idx_a: int, idx_b: int) -> float:
        return util.cos_sim(  # type: ignore
            self.embeddings_minilm_a[idx_a],  # type: ignore
            self.embeddings_minilm_b[idx_b],  # type: ignore
        )

    def score_mpnet(self, idx_a: int, idx_b: int) -> float:
        return util.cos_sim(  # type: ignore
            self.embeddings_minilm_a[idx_a],  # type: ignore
            self.embeddings_minilm_b[idx_b],  # type: ignore
        )

    def score_labse(self, idx_a: int, idx_b: int) -> float:
        return util.cos_sim(  # type: ignore
            self.embeddings_minilm_a[idx_a],  # type: ignore
            self.embeddings_minilm_b[idx_b],  # type: ignore
        )
