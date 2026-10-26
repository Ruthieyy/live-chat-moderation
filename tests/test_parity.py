from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from moderation.network import TextClassifier  # noqa: E402
from moderation.settings import default_model_dir  # noqa: E402


def test_onnx_matches_pytorch_logits() -> None:
    import onnxruntime as ort

    model_dir = default_model_dir()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    network = config["network"]
    model = TextClassifier(
        vocab_size=int(network["vocab_size"]),
        embed_dim=int(network["embed_dim"]),
        hidden_dim=int(network["hidden_dim"]),
        num_labels=int(network["num_labels"]),
        pad_idx=int(network["pad_idx"]),
        dropout=float(network["dropout"]),
    )
    state_path = Path(model_dir) / "model.pt"
    if not state_path.exists():
        pytest.skip("model.pt is not in the tree")
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    ids = torch.randint(0, int(network["vocab_size"]), (4, int(config["max_len"])))
    with torch.no_grad():
        expected = model(ids).numpy()
    session = ort.InferenceSession(
        str(Path(model_dir) / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(["logits"], {"input_ids": ids.numpy()})[0]
    assert np.max(np.abs(expected - actual)) < 1e-3
