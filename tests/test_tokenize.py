from __future__ import annotations

from moderation.tokenize import Vocabulary, featurize, normalize, select_tokens, tokenize


def test_normalize_collapses_character_runs_and_case() -> None:
    assert normalize("SOOOOO cool!!!") == "soo cool!!"


def test_tokenizer_keeps_contractions() -> None:
    assert "don't" in tokenize("Please don't shout")


def test_head_and_tail_window_keeps_both_ends() -> None:
    tokens = [f"w{index}" for index in range(6)]
    assert select_tokens(tokens, max_tokens=4, tail_tokens=2) == ["w0", "w1", "w4", "w5"]


def test_bigrams_are_built_from_the_kept_window() -> None:
    features, truncated = featurize("one two three four five six", max_tokens=4, tail_tokens=2)
    assert truncated is True
    assert features[:4] == ["one", "two", "five", "six"]
    assert "two_five" in features


def test_short_text_is_not_truncated() -> None:
    _, truncated = featurize("hello there", max_tokens=8, tail_tokens=2)
    assert truncated is False


def test_encode_pads_and_roundtrips(tmp_path) -> None:
    vocab = Vocabulary.build(
        ["hello there friend", "goodbye friend"],
        max_size=50,
        max_len=16,
        max_tokens=8,
        tail_tokens=2,
        min_freq=1,
    )
    ids, truncated = vocab.encode("hello there")
    assert len(ids) == 16
    assert truncated is False
    assert ids[0] != 0
    assert ids[-1] == 0
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.encode("hello there") == (ids, truncated)


def test_unknown_token_uses_unk_id() -> None:
    vocab = Vocabulary.build(
        ["hello there"],
        max_size=50,
        max_len=8,
        max_tokens=8,
        tail_tokens=2,
        min_freq=1,
    )
    ids, _ = vocab.encode("zzzz-not-in-vocab")
    assert 1 in ids
