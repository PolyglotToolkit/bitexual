import enum

type Pair = tuple[int, int]
type Pairs = list[Pair]
type Scores = dict[Pair, float]


class Language(enum.Enum):
    EN = enum.auto()
    AR = enum.auto()
    FR = enum.auto()
    ZH_S = enum.auto()
    ZH_T = enum.auto()
    RU = enum.auto()


class Model(enum.Enum):
    MINILM = enum.auto()
    MPNET = enum.auto()
    LABSE = enum.auto()


class Translator(enum.Enum):
    HELSINKI = enum.auto()
    GOOGLE = enum.auto()
