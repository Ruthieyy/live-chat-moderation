"""Compare PyTorch eager, ONNX Runtime FP32, and INT8 forward latency.

Times are wall clock around the forward call only (tokenization excluded),
except ``engine_predict`` which is the full in-process serving path. Each
process is warmed up, then measured sequentially with ``time.perf_counter``.
ORT sessions use one intra-op thread, matching the API default.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moderation.inference import InferenceEngine  # noqa: E402
from moderation.network import TextClassifier  # noqa: E402
from moderation.settings import default_model_dir  # noqa: E402
from moderation.tokenize import Vocabulary  # noqa: E402


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def percentiles(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "n": int(values.size),
        "mean_ms": round(float(values.mean()), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "p99_ms": round(float(np.percentile(values, 99)), 4),
        "min_ms": round(float(values.min()), 4),
        "max_ms": round(float(values.max()), 4),
    }


def time_calls(fn, iterations: int, warmup: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return percentiles(samples)


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_texts() -> list[str]:
    path = ROOT / "loadtest" / "comments.jsonl"
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(json.loads(line)["text"])
    if len(texts) < 32:
        raise SystemExit(f"need at least 32 comments in {path}")
    return texts


def main() -> None:
    model_dir = default_model_dir()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocabulary = Vocabulary.load(model_dir / "vocab.json")
    network = config["network"]
    model = TextClassifier(
        vocab_size=int(network["vocab_size"]),
        embed_dim=int(network["embed_dim"]),
        hidden_dim=int(network["hidden_dim"]),
        num_labels=int(network["num_labels"]),
        pad_idx=int(network["pad_idx"]),
        dropout=float(network["dropout"]),
    )
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    texts = load_texts()
    single_ids, _ = vocabulary.encode(texts[0])
    batch_ids, _ = vocabulary.encode_batch(texts[:32])
    single = np.asarray([single_ids], dtype=np.int64)
    batch = np.asarray(batch_ids, dtype=np.int64)
    single_tensor = torch.from_numpy(single.copy())
    batch_tensor = torch.from_numpy(batch.copy())

    fp32 = session(model_dir / "model.onnx", 1)
    int8_path = model_dir / "model.int8.onnx"
    int8 = session(int8_path, 1) if int8_path.exists() else None

    def pytorch_single() -> None:
        with torch.no_grad():
            model(single_tensor)

    def pytorch_batch() -> None:
        with torch.no_grad():
            model(batch_tensor)

    def onnx_single(active: ort.InferenceSession) -> None:
        active.run(["logits"], {"input_ids": single})

    def onnx_batch(active: ort.InferenceSession) -> None:
        active.run(["logits"], {"input_ids": batch})

    report: dict = {
        "cpu": cpu_model(),
        "cpu_count": os_cpu(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "method": (
            "Sequential wall clock with time.perf_counter. "
            "Warmup 100 calls. Forward rows exclude tokenization. "
            "ORT intra_op_num_threads=1. "
            "pytorch_1thread sets torch.set_num_threads(1); "
            "pytorch_all_threads uses the machine CPU count."
        ),
        "served_onnx_file": config["onnx_file"],
        "forward": {},
        "engine_predict_batch_1": {},
    }

    with torch.no_grad():
        expected = model(single_tensor).numpy()
    fp32_out = fp32.run(["logits"], {"input_ids": single})[0]
    report["parity"] = {
        "fp32_max_abs_logit": float(np.max(np.abs(expected - fp32_out))),
    }
    if int8 is not None:
        int8_out = int8.run(["logits"], {"input_ids": single})[0]
        int8_prob = 1.0 / (1.0 + np.exp(-np.clip(int8_out, -30, 30)))
        fp32_prob = 1.0 / (1.0 + np.exp(-np.clip(fp32_out, -30, 30)))
        report["parity"]["int8_max_abs_prob"] = float(np.max(np.abs(fp32_prob - int8_prob)))

    def forward_block(iterations: int) -> dict:
        torch.set_num_threads(1)
        block = {
            "pytorch_1thread": {
                "batch_1": time_calls(pytorch_single, iterations, 100),
                "batch_32": time_calls(pytorch_batch, max(50, iterations // 4), 30),
            }
        }
        torch.set_num_threads(os_cpu())
        block["pytorch_all_threads"] = {
            "batch_1": time_calls(pytorch_single, iterations, 100),
            "batch_32": time_calls(pytorch_batch, max(50, iterations // 4), 30),
        }
        torch.set_num_threads(1)
        block["onnx_fp32"] = {
            "batch_1": time_calls(lambda: onnx_single(fp32), iterations, 100),
            "batch_32": time_calls(lambda: onnx_batch(fp32), max(50, iterations // 4), 30),
        }
        if int8 is not None:
            block["onnx_int8"] = {
                "batch_1": time_calls(lambda: onnx_single(int8), iterations, 100),
                "batch_32": time_calls(lambda: onnx_batch(int8), max(50, iterations // 4), 30),
            }
        return block

    report["forward"] = forward_block(500)
    engine = InferenceEngine.from_model_dir(model_dir, intra_op_threads=1, max_concurrent=1)
    report["engine_predict_batch_1"] = time_calls(lambda: engine.predict([texts[0]]), 500, 50)
    report["engine_predict_batch_32"] = time_calls(lambda: engine.predict(texts[:32]), 200, 20)

    destination = ROOT / "benchmarks" / "runtime.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["forward"], indent=2))
    print("engine", json.dumps(report["engine_predict_batch_1"], indent=2))
    print(f"wrote {destination}")


def os_cpu() -> int:
    import os

    return max(1, os.cpu_count() or 1)


if __name__ == "__main__":
    main()
