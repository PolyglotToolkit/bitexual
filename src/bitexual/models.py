# type: ignore
"""
For future reference:
- `deep-translator` from Google/DeepL/etc.: easiest to use, good for evaluation
- `googletrans` from Google (unofficial): frequently breaks when Google API changes
- `argostranslate`: local neural MT: fully offline, no rate limits
- `Helsinki-NLP` via `transformers`: local neural MT, per-language-pair models, heavy but solid
"""

from typing import Callable

from adiumentum.typing import Endofunction
from sentence_transformers import SentenceTransformer

from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator  # type: ignore

from .types import Language, Model, Translator


def load_embedding_model(model: Model) -> SentenceTransformer:
    model_names = {
        Model.MINILM: "paraphrase-multilingual-MiniLM-L12-v2",
        Model.MPNET: "paraphrase-multilingual-mpnet-base-v2",
        Model.LABSE: "LaBSE",
    }
    if isinstance(model, str):
        model = Model(model.lower())
    return SentenceTransformer(model_names[model])


def make_helsinki_translator(
    source: Language, target: Language = Language.EN
) -> Callable[[str], str]:
    model_name = f"Helsinki-NLP/opus-mt-{source!s}-{target!s}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    def translate(text: str) -> str:
        tokens = tokenizer([text], return_tensors="pt", padding=True)
        output = model.generate(**tokens)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate


def make_google_translator(
    source: Language, target: Language = Language.EN
) -> Callable[[str], str]:
    def translate(text: str) -> str:
        return GoogleTranslator(source=str(source), target=str(target)).translate(text)

    return translate


class LazyModels:
    """Tasked with loading and storing models as soon as they are needed. But no sooner!"""

    def __init__(self) -> None:
        self._minilm: SentenceTransformer | None = None
        self._mpnet: SentenceTransformer | None = None
        self._labse: SentenceTransformer | None = None

        self.embedding_models: dict[Model, SentenceTransformer | None] = {
            Model.MINILM: self._minilm,
            Model.MPNET: self._mpnet,
            Model.LABSE: self._labse,
        }
        self.translator_factories: dict[
            Translator, Callable[[Language, Language], Endofunction[str]]
        ] = {
            Translator.HELSINKI: make_helsinki_translator,
            Translator.GOOGLE: make_google_translator,
        }
        self.translators: dict[
            tuple[Translator, Language, Language], Endofunction[str]
        ] = {}

    def get_model(self, m: Model) -> SentenceTransformer:
        if self.embedding_models[m] is None:
            self.embedding_models[m] = load_embedding_model(m)
        return self.embedding_models[m]  # type: ignore

    def get_translator(
        self, t: Translator, source: Language, target: Language
    ) -> Endofunction[str]:
        key = (t, source, target)
        if self.translators.get(key) is None:
            factory = self.translator_factories[t]
            self.translators.update({key: factory(source, target)})
        return self.translators[key]
