"""Label taxonomy shared by training and serving.

The neural net predicts the six Jigsaw toxicity heads. The product API exposes
three scores a livestream moderator actually acts on: toxic (any toxicity
head), hate (identity_hate), and spam (rules, not a learned head — the public
Jigsaw release has no spam labels).
"""

MODEL_LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

# Heads that roll up into the product "toxic" score.
TOXIC_FAMILY: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
)

PRODUCT_LABELS: tuple[str, ...] = ("toxic", "hate", "spam")

# Hard ceiling accepted by the HTTP schema. Settings can tighten this further.
SCHEMA_MAX_TEXT_CHARS = 8000
