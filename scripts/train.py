"""Train the livestream toxicity classifier and write serving artifacts.

The dataset is the public Jigsaw Toxic Comment Classification Challenge
(Wikipedia talk pages), mirrored without a Kaggle login by
Heliosoph/Jigsaw-Toxic-Comments. The model is a mean-pooled word-and-bigram
embedding small enough to finish on CPU. Official test rows whose labels are
-1 are dropped, matching the Kaggle scoring split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import random
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_onnx import export_fp32, export_int8, max_logit_gap  # noqa: E402
from moderation.calibration import (  # noqa: E402
    clamp_review_below_block,
    operating_point,
    threshold_for_precision,
    threshold_for_recall,
)
from moderation.drift import DEFAULT_EDGES, histogram  # noqa: E402
from moderation.labels import MODEL_LABELS  # noqa: E402
from moderation.network import TextClassifier  # noqa: E402
from moderation.policy import Thresholds, decide  # noqa: E402
from moderation.spam import score_spam  # noqa: E402
from moderation.tokenize import Vocabulary  # noqa: E402

REVISION = "78e67fb081bc05846a90e944d7bb8f2f4d36e438"
DATASET_BASE = "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/" + REVISION
DATASET_FILES = ("train.csv.gz", "test.csv.gz", "test_labels.csv.gz")
LABELS = MODEL_LABELS
TOXIC_INDEXES = [LABELS.index(name) for name in LABELS if name != "identity_hate"]
HATE_INDEX = LABELS.index("identity_hate")
THREAT_INDEX = LABELS.index("threat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tail-tokens", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--pos-weight-cap", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=0, help="Debug cap on training rows. 0 uses all.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        destination = data_dir / name
        if destination.exists() and destination.stat().st_size > 0:
            continue
        url = f"{DATASET_BASE}/{name}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, destination)


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if int(record["toxic"]) < 0:
                continue
            row = {"text": record["comment_text"]}
            for label in LABELS:
                row[label] = int(record[label])
            rows.append(row)
    return rows


def load_train(path: Path) -> list[dict]:
    return _read_rows(path)


def load_test(data_dir: Path) -> list[dict]:
    labels: dict[str, dict[str, int]] = {}
    with gzip.open(data_dir / "test_labels.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if int(record["toxic"]) < 0:
                continue
            labels[record["id"]] = {label: int(record[label]) for label in LABELS}
    rows: list[dict] = []
    with gzip.open(data_dir / "test.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            gold = labels.get(record["id"])
            if gold is None:
                continue
            row = {"text": record["comment_text"]}
            row.update(gold)
            rows.append(row)
    return rows


def stratified_split(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    positive = [row for row in rows if row["toxic"] == 1]
    negative = [row for row in rows if row["toxic"] == 0]
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_pos = max(1, int(len(positive) * val_fraction))
    n_neg = max(1, int(len(negative) * val_fraction))
    validation = positive[:n_pos] + negative[:n_neg]
    train = positive[n_pos:] + negative[n_neg:]
    rng.shuffle(validation)
    rng.shuffle(train)
    return train, validation


def encode_split(rows: list[dict], vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray, float]:
    features = np.zeros((len(rows), vocabulary.max_len), dtype=np.int64)
    targets = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    truncated = 0
    for index, row in enumerate(rows):
        ids, was_truncated = vocabulary.encode(row["text"])
        features[index] = ids
        truncated += int(was_truncated)
        for label_index, label in enumerate(LABELS):
            targets[index, label_index] = row[label]
    return features, targets, truncated / max(len(rows), 1)


def label_rates(targets: np.ndarray) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        count = int(targets[:, index].sum())
        report[label] = {"positives": count, "rate": count / max(len(targets), 1)}
    return report


def per_label_scores(targets: np.ndarray, probs: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, label in enumerate(LABELS):
        y_true = targets[:, index]
        if y_true.min() == y_true.max():
            continue
        report[label] = {
            "roc_auc": float(roc_auc_score(y_true, probs[:, index])),
            "average_precision": float(average_precision_score(y_true, probs[:, index])),
        }
    if report:
        report["mean_roc_auc"] = {
            "roc_auc": float(np.mean([item["roc_auc"] for item in report.values()])),
            "average_precision": float(np.mean([item["average_precision"] for item in report.values()])),
        }
    return report


def predict_probs(model: TextClassifier, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size])
            outputs.append(model(batch).numpy())
    logits = np.concatenate(outputs, axis=0)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def train_model(
    model: TextClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight_cap: float,
) -> tuple[TextClassifier, list[dict], list[float]]:
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, pos_weight_cap).astype(np.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_auc = -1.0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(x_train))
        total_loss = 0.0
        started = time.perf_counter()
        for step, start in enumerate(range(0, len(order), batch_size), start=1):
            batch_index = order[start : start + batch_size]
            features = torch.from_numpy(x_train[batch_index])
            target = torch.from_numpy(y_train[batch_index])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_index)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step} loss {float(loss.detach()):.4f}", flush=True)
        probs = predict_probs(model, x_val)
        scores = per_label_scores(y_val, probs)
        mean_auc = float(scores["mean_roc_auc"]["roc_auc"])
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(x_train),
            "val_mean_roc_auc": mean_auc,
            "seconds": round(elapsed, 2),
        }
        history.append(row)
        print(
            f"epoch {epoch} loss {row['train_loss']:.4f} val_auc {mean_auc:.4f} ({elapsed:.1f}s)",
            flush=True,
        )
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, [float(value) for value in pos_weight]


def toxic_aggregate(probs: np.ndarray) -> np.ndarray:
    return probs[:, TOXIC_INDEXES].max(axis=1)


def any_toxic(targets: np.ndarray) -> np.ndarray:
    return targets[:, TOXIC_INDEXES].max(axis=1) > 0.5


def choose_thresholds(targets: np.ndarray, probs: np.ndarray) -> tuple[Thresholds, dict]:
    toxic_y = any_toxic(targets)
    toxic_s = toxic_aggregate(probs)
    hate_y = targets[:, HATE_INDEX] > 0.5
    hate_s = probs[:, HATE_INDEX]
    threat_y = targets[:, THREAT_INDEX] > 0.5
    threat_s = probs[:, THREAT_INDEX]

    toxic_block = threshold_for_precision(toxic_y, toxic_s, min_precision=0.85, min_predicted=30)
    toxic_review = threshold_for_recall(toxic_y, toxic_s, min_recall=0.80, min_predicted=30)
    hate_block = threshold_for_precision(hate_y, hate_s, min_precision=0.60, min_predicted=10)
    hate_review = threshold_for_recall(hate_y, hate_s, min_recall=0.50, min_predicted=10)
    threat_block = threshold_for_precision(threat_y, threat_s, min_precision=0.40, min_predicted=8)

    hate_block_threshold = hate_block.threshold
    if hate_block.precision < 0.60:
        hate_block_threshold = max(hate_block.threshold, 0.75)
    threat_block_threshold = threat_block.threshold
    if threat_block.precision < 0.40:
        threat_block_threshold = max(threat_block.threshold, 0.70)

    thresholds = Thresholds(
        toxic_review=clamp_review_below_block(toxic_review.threshold, toxic_block.threshold),
        toxic_block=float(toxic_block.threshold),
        hate_review=clamp_review_below_block(hate_review.threshold, hate_block_threshold),
        hate_block=float(hate_block_threshold),
        threat_block=float(threat_block_threshold),
        spam_review=0.70,
    )
    final = {
        "toxic_block": operating_point(toxic_y, toxic_s, thresholds.toxic_block).as_dict(),
        "toxic_review": operating_point(toxic_y, toxic_s, thresholds.toxic_review).as_dict(),
        "hate_block": operating_point(hate_y, hate_s, thresholds.hate_block).as_dict(),
        "hate_review": operating_point(hate_y, hate_s, thresholds.hate_review).as_dict(),
        "threat_block": operating_point(threat_y, threat_s, thresholds.threat_block).as_dict(),
        "candidates": {
            "toxic_block": toxic_block.as_dict(),
            "toxic_review": toxic_review.as_dict(),
            "hate_block": hate_block.as_dict(),
            "hate_review": hate_review.as_dict(),
            "threat_block": threat_block.as_dict(),
        },
    }
    return thresholds, final


def actions_for(rows: list[dict], probs: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    actions = []
    for index, row in enumerate(rows):
        attributes = {label: float(probs[index, label_index]) for label_index, label in enumerate(LABELS)}
        spam_score, spam_reasons = score_spam(row["text"])
        decision = decide(attributes, spam_score, spam_reasons, thresholds)
        actions.append(decision.action)
    return np.asarray(actions)


def decision_report(actions: np.ndarray, y_bad: np.ndarray) -> dict[str, float | int | None]:
    blocked = actions == "block"
    reviewed = actions == "review"
    flagged = blocked | reviewed
    clean = ~y_bad

    def ratio(hit: np.ndarray, base: np.ndarray) -> float | None:
        denom = int(base.sum())
        if denom == 0:
            return None
        return float(np.sum(hit & base) / denom)

    return {
        "n": int(len(actions)),
        "n_bad": int(y_bad.sum()),
        "n_block": int(blocked.sum()),
        "n_review": int(reviewed.sum()),
        "n_allow": int((actions == "allow").sum()),
        "block_precision": ratio(y_bad, blocked),
        "block_recall": ratio(blocked, y_bad),
        "flag_precision": ratio(y_bad, flagged),
        "flag_recall": ratio(flagged, y_bad),
        "clean_block_rate": ratio(blocked, clean),
        "clean_flag_rate": ratio(flagged, clean),
    }


def reference_counts(rows: list[dict], probs: np.ndarray) -> dict:
    edges = np.asarray(DEFAULT_EDGES, dtype=np.float64)
    toxic = toxic_aggregate(probs)
    hate = probs[:, HATE_INDEX]
    spam = np.asarray([score_spam(row["text"])[0] for row in rows], dtype=np.float64)
    series = {"toxic": toxic, "hate": hate, "spam": spam}
    return {
        "edges": list(DEFAULT_EDGES),
        "sample_size": len(rows),
        "counts": {label: histogram(values, edges).astype(int).tolist() for label, values in series.items()},
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return round(value, 6) if abs(value) < 1e6 else value
    return value


def prob_gap(left_path: Path, right_path: Path, features: np.ndarray) -> float:
    import onnxruntime as ort

    sample = features[:256]
    left = ort.InferenceSession(str(left_path), providers=["CPUExecutionProvider"])
    right = ort.InferenceSession(str(right_path), providers=["CPUExecutionProvider"])
    left_probs = 1.0 / (1.0 + np.exp(-np.clip(left.run(["logits"], {"input_ids": sample})[0], -30, 30)))
    right_probs = 1.0 / (1.0 + np.exp(-np.clip(right.run(["logits"], {"input_ids": sample})[0], -30, 30)))
    return float(np.max(np.abs(left_probs - right_probs)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(os_cpu())
    started = time.perf_counter()
    download_dataset(args.data_dir)
    print("loading train split", flush=True)
    rows = load_train(args.data_dir / "train.csv.gz")
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = stratified_split(rows, args.val_fraction, args.seed)
    print(f"rows train={len(train_rows)} val={len(val_rows)}", flush=True)
    vocabulary = Vocabulary.build(
        [row["text"] for row in train_rows],
        max_size=args.vocab_size,
        max_len=args.max_len,
        max_tokens=args.max_tokens,
        tail_tokens=args.tail_tokens,
        min_freq=args.min_freq,
    )
    print(f"vocab size {vocabulary.size}", flush=True)
    x_train, y_train, train_trunc = encode_split(train_rows, vocabulary)
    x_val, y_val, val_trunc = encode_split(val_rows, vocabulary)
    model = TextClassifier(
        vocab_size=vocabulary.size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(LABELS),
    )
    model, history, pos_weight = train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight_cap=args.pos_weight_cap,
    )
    val_probs = predict_probs(model, x_val)
    val_scores = per_label_scores(y_val, val_probs)
    thresholds, operating = choose_thresholds(y_val, val_probs)
    print("loading official test split", flush=True)
    test_rows = load_test(args.data_dir)
    x_test, y_test, test_trunc = encode_split(test_rows, vocabulary)
    test_probs = predict_probs(model, x_test)
    test_scores = per_label_scores(y_test, test_probs)
    y_bad = y_test.max(axis=1) > 0.5
    policy_actions = actions_for(test_rows, test_probs, thresholds)
    model_only = Thresholds(
        toxic_review=thresholds.toxic_review,
        toxic_block=thresholds.toxic_block,
        hate_review=thresholds.hate_review,
        hate_block=thresholds.hate_block,
        threat_block=thresholds.threat_block,
        spam_review=2.0,
    )
    model_actions = actions_for(test_rows, test_probs, model_only)
    test_decisions = {
        "with_spam_rules": decision_report(policy_actions, y_bad),
        "model_only": decision_report(model_actions, y_bad),
    }
    print("test decisions", json.dumps(jsonable(test_decisions)), flush=True)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(args.model_dir / "vocab.json")
    torch.save(model.state_dict(), args.model_dir / "model.pt")
    fp32_path = args.model_dir / "model.onnx"
    int8_path = args.model_dir / "model.int8.onnx"
    export_fp32(model, fp32_path, args.max_len)
    fp32_gap = max_logit_gap(model, fp32_path, max_len=args.max_len, vocab_size=vocabulary.size)
    print(f"fp32 logit gap {fp32_gap:.6e}", flush=True)
    if fp32_gap > 1e-3:
        raise SystemExit(f"ONNX parity failed: max logit gap {fp32_gap}")
    int8_gap = None
    int8_error = None
    try:
        export_int8(fp32_path, int8_path)
        int8_gap = prob_gap(fp32_path, int8_path, x_val)
        print(f"int8 prob gap {int8_gap:.6e}", flush=True)
    except Exception as exc:  # quantization is optional; FP32 still serves
        int8_error = str(exc)
        print(f"int8 export failed: {exc}", flush=True)
    onnx_file = "model.onnx"
    if int8_gap is not None and int8_gap <= 0.03:
        onnx_file = "model.int8.onnx"
    reference = reference_counts(val_rows, val_probs)
    (args.model_dir / "reference.json").write_text(json.dumps(reference), encoding="utf-8")

    config = {
        "model_version": "bow-bigram-v1",
        "trained_at": datetime.now(UTC).isoformat(),
        "onnx_file": onnx_file,
        "labels": list(LABELS),
        "thresholds": thresholds.as_dict(),
        "max_len": args.max_len,
        "max_tokens": args.max_tokens,
        "tail_tokens": args.tail_tokens,
        "network": {
            "vocab_size": vocabulary.size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_labels": len(LABELS),
            "pad_idx": 0,
            "dropout": model.dropout_p,
        },
        "dataset": {
            "name": "jigsaw-toxic-comment-classification",
            "huggingface": "Heliosoph/Jigsaw-Toxic-Comments",
            "revision": REVISION,
            "license": "CC0 packaging; comment text is Wikipedia CC BY-SA 3.0",
        },
        "seed": args.seed,
        "metrics": {
            "validation": val_scores,
            "test": test_scores,
            "operating_points": operating,
            "test_decisions": test_decisions,
            "truncation_rate": {
                "train": train_trunc,
                "validation": val_trunc,
                "test": test_trunc,
            },
            "label_rates_train": label_rates(y_train),
            "pos_weight": pos_weight,
            "history": history,
            "onnx": {
                "fp32_logit_gap": fp32_gap,
                "int8_prob_gap": int8_gap,
                "int8_error": int8_error,
                "served_file": onnx_file,
            },
        },
    }
    payload = jsonable(config)
    (args.model_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "cpu": cpu_model(),
        "cpu_count": os_cpu(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "train_seconds": round(time.perf_counter() - started, 2),
        "rows": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "config": payload,
    }
    (args.benchmark_dir / "training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.model_dir} served={onnx_file}", flush=True)
    print(f"val mean ROC-AUC {val_scores['mean_roc_auc']['roc_auc']:.4f}", flush=True)
    print(f"test mean ROC-AUC {test_scores['mean_roc_auc']['roc_auc']:.4f}", flush=True)


def os_cpu() -> int:
    import os

    return max(1, os.cpu_count() or 1)


if __name__ == "__main__":
    main()
