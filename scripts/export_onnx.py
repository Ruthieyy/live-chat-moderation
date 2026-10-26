"""Export the trained classifier to ONNX and an INT8 weight-quantized copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from moderation.network import TextClassifier


def export_fp32(model: TextClassifier, path: Path, max_len: int) -> None:
    model.eval()
    dummy = torch.zeros(2, max_len, dtype=torch.long)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def export_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def max_logit_gap(model: TextClassifier, onnx_path: Path, *, max_len: int, vocab_size: int) -> float:
    import onnxruntime as ort

    model.eval()
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab_size, size=(16, max_len), dtype=np.int64)
    ids[:, -8:] = 0
    with torch.no_grad():
        expected = model(torch.from_numpy(ids.copy())).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"input_ids": ids})[0]
    return float(np.max(np.abs(expected - actual)))
