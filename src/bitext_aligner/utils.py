from typing import Callable, cast


import re


from arabic_buckwalter_transliteration.transliteration import (  # type: ignore
    buckwalter_to_arabic,  # type: ignore
    arabic_to_buckwalter,  # type: ignore
)


arabic2ascii = cast(Callable[[str], str], arabic_to_buckwalter)
ascii2arabic = cast(Callable[[str], str], buckwalter_to_arabic)


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def simple_tokenize_cjk(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[\uac00-\ud7af]|\w+|[^\w\s]", text, re.UNICODE)


def make_lines(text: str) -> list[str]:
    return [line for line in text.strip().split("\n") if line]
