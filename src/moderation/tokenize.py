"""Word and bigram featurizer.

Livestream comments are short, but the Jigsaw training comments are Wikipedia
talk pages with a long tail (p90 is about 180 words). The encoder keeps a
head+tail token window so a slur at either end still reaches the model, then
appends word bigrams of that window. Mean-pooling makes the feature order
irrelevant; bigrams are what recover phrases like "go away".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and squash long character runs.

    "sooooo" and "soo" share a token. Spam and insults use those runs; the
    raw text is still available to the spam rules, which want the original.
    """
    lowered = text.lower().replace("\x00", " ")
    collapsed = _SPACE_RE.sub(" ", lowered).strip()
    return _REPEAT_RE.sub(r"\1\1", collapsed)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def select_tokens(tokens: list[str], max_tokens: int, tail_tokens: int) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if tail_tokens < 0 or tail_tokens >= max_tokens:
        raise ValueError("tail_tokens must be in [0, max_tokens)")
    if len(tokens) <= max_tokens:
        return tokens
    head = max_tokens - tail_tokens
    return tokens[:head] + tokens[-tail_tokens:]


def featurize(text: str, max_tokens: int, tail_tokens: int) -> tuple[list[str], bool]:
    tokens = tokenize(text)
    truncated = len(tokens) > max_tokens
    kept = select_tokens(tokens, max_tokens, tail_tokens)
    bigrams = [f"{left}_{right}" for left, right in zip(kept, kept[1:], strict=False)]
    return kept + bigrams, truncated


class Vocabulary:
    def __init__(
        self,
        token_to_id: dict[str, int],
        max_len: int,
        max_tokens: int,
        tail_tokens: int,
    ) -> None:
        self.token_to_id = token_to_id
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.tail_tokens = tail_tokens

    @property
    def size(self) -> int:
        return len(self.token_to_id) + 2

    @classmethod
    def build(
        cls,
        texts: list[str],
        *,
        max_size: int,
        max_len: int,
        max_tokens: int,
        tail_tokens: int,
        min_freq: int,
    ) -> Vocabulary:
        if max_size < 3:
            raise ValueError("max_size must leave room for pad and unk")
        counts: Counter[str] = Counter()
        for text in texts:
            features, _ = featurize(text, max_tokens, tail_tokens)
            counts.update(features)
        most_common = [tok for tok, freq in counts.most_common(max_size - 2) if freq >= min_freq]
        token_to_id = {tok: index + 2 for index, tok in enumerate(most_common)}
        return cls(token_to_id, max_len, max_tokens, tail_tokens)

    def encode(self, text: str) -> tuple[list[int], bool]:
        features, truncated = featurize(text, self.max_tokens, self.tail_tokens)
        ids = [self.token_to_id.get(feature, UNK_ID) for feature in features[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([PAD_ID] * (self.max_len - len(ids)))
        return ids, truncated

    def encode_batch(self, texts: list[str]) -> tuple[list[list[int]], list[bool]]:
        encoded = [self.encode(text) for text in texts]
        ids = [row for row, _ in encoded]
        truncated = [flag for _, flag in encoded]
        return ids, truncated

    def save(self, path: Path) -> None:
        payload = {
            "max_len": self.max_len,
            "max_tokens": self.max_tokens,
            "tail_tokens": self.tail_tokens,
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
            max_len=int(payload["max_len"]),
            max_tokens=int(payload["max_tokens"]),
            tail_tokens=int(payload["tail_tokens"]),
        )
