from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix and probability visualizations for a TFLite classifier."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Folder containing class subfolders (one subfolder per class).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("meat_model.tflite"),
        help="Path to the .tflite model file.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("labels.txt"),
        help="Path to labels file (one label per line).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_reports"),
        help="Where to save plots and CSV files.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images to evaluate (0 means all).",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels: list[str] = []
    for raw_line in labels_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line)

    if not labels:
        raise ValueError("No labels found in labels file.")

    return labels


def list_samples(dataset_dir: Path, labels: list[str]) -> list[tuple[Path, int]]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    label_to_idx = {name: idx for idx, name in enumerate(labels)}
    samples: list[tuple[Path, int]] = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in label_to_idx:
            print(f"Skipping folder '{class_name}' (not found in labels.txt)")
            continue

        true_idx = label_to_idx[class_name]
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, true_idx))

    if not samples:
        raise ValueError(
            "No images found. Ensure dataset layout is dataset_dir/<class_name>/<images>."
        )

    return samples


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def looks_like_probabilities(x: np.ndarray, tol: float = 1e-3) -> bool:
    if x.ndim != 1 or x.size == 0:
        return False
    if not np.all(np.isfinite(x)):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    return abs(float(np.sum(x)) - 1.0) <= 1e-2


def preprocess_image(image_path: Path, input_shape: np.ndarray, input_dtype: np.dtype) -> np.ndarray:
    if len(input_shape) != 4:
        raise ValueError(f"Unexpected model input shape: {input_shape}")

    height = int(input_shape[1])
    width = int(input_shape[2])

    img = Image.open(image_path).convert("RGB").resize((width, height))
    arr = np.asarray(img)

    if input_dtype == np.float32:
        x = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        x = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    return np.expand_dims(x, axis=0)


def run_inference(
    interpreter: tf.lite.Interpreter,
    input_details: list[dict],
    output_details: list[dict],
    x: np.ndarray,
) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"]).reshape(-1)

    out_quant = output_details[0].get("quantization")
    if out_quant and isinstance(out_quant, tuple) and len(out_quant) == 2:
        scale, zero_point = out_quant
        if scale not in (0, 0.0):
            y = (y.astype(np.float32) - float(zero_point)) * float(scale)

    y = y.astype(np.float32)
    if looks_like_probabilities(y):
        return y
    return softmax(y)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    np.divide(cm_norm, row_sums, out=cm_norm, where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_classification_report_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", index=True)

    class_metrics = report_df.loc[labels, ["precision", "recall", "f1-score"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.55)))
    sns.heatmap(
        class_metrics,
        annot=True,
        fmt=".2f",
        cmap="PuBuGn",
        vmin=0,
        vmax=1,
        cbar=True,
        ax=ax,
    )
    ax.set_title("Classification Report Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Class")
    plt.tight_layout()
    fig.savefig(output_dir / "classification_report_heatmap.png", dpi=220)
    plt.close(fig)

    return report_df


def plot_probability_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    max_conf = probs.max(axis=1)
    correct = y_true == y_pred

    rows: list[dict[str, float | int | str]] = []
    for idx, class_name in enumerate(labels):
        mask = y_true == idx
        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "class": class_name,
                    "samples": 0,
                    "class_accuracy": np.nan,
                    "mean_true_class_probability": np.nan,
                    "mean_top1_confidence": np.nan,
                }
            )
            continue

        class_acc = float((y_pred[mask] == idx).mean())
        mean_true_prob = float(probs[mask, idx].mean())
        mean_top1_conf = float(max_conf[mask].mean())

        rows.append(
            {
                "class": class_name,
                "samples": count,
                "class_accuracy": class_acc,
                "mean_true_class_probability": mean_true_prob,
                "mean_top1_confidence": mean_top1_conf,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "prediction_probability_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    if correct.any():
        sns.histplot(
            max_conf[correct],
            bins=18,
            color="#2E7D32",
            alpha=0.6,
            label="Correct",
            ax=axes[0],
        )
    if (~correct).any():
        sns.histplot(
            max_conf[~correct],
            bins=18,
            color="#C62828",
            alpha=0.6,
            label="Incorrect",
            ax=axes[0],
        )
    axes[0].set_title("Top-1 Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        stats_df["class_accuracy"].values,
        width,
        label="Class Accuracy",
        color="#1565C0",
    )
    axes[1].bar(
        x + width / 2,
        stats_df["mean_true_class_probability"].values,
        width,
        label="Mean True-Class Probability",
        color="#EF6C00",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-Class Probability Stats")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "prediction_probability_stats.png", dpi=220)
    plt.close(fig)

    return stats_df


def main() -> None:
    args = parse_args()

    model_path = args.model_path.resolve()
    labels_path = args.labels_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path)
    samples = list_samples(dataset_dir, labels)

    if args.max_images > 0:
        samples = samples[: args.max_images]

    print(f"Evaluating {len(samples)} images...")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[np.ndarray] = []
    all_paths: list[str] = []

    for image_path, true_idx in samples:
        x = preprocess_image(image_path, input_details[0]["shape"], input_details[0]["dtype"])
        probs = run_inference(interpreter, input_details, output_details, x)
        pred_idx = int(np.argmax(probs))

        all_true.append(true_idx)
        all_pred.append(pred_idx)
        all_probs.append(probs)
        all_paths.append(str(image_path))

    y_true = np.asarray(all_true, dtype=np.int32)
    y_pred = np.asarray(all_pred, dtype=np.int32)
    probs_matrix = np.vstack(all_probs).astype(np.float32)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {overall_acc:.4f}")

    plot_confusion_matrices(y_true, y_pred, labels, output_dir)
    report_df = plot_classification_report_heatmap(y_true, y_pred, labels, output_dir)
    stats_df = plot_probability_stats(y_true, y_pred, probs_matrix, labels, output_dir)

    pred_df = pd.DataFrame(
        {
            "image_path": all_paths,
            "true_index": y_true,
            "true_label": [labels[i] for i in y_true],
            "pred_index": y_pred,
            "pred_label": [labels[i] for i in y_pred],
            "pred_confidence": probs_matrix.max(axis=1),
        }
    )

    for idx, label in enumerate(labels):
        pred_df[f"prob_{label}"] = probs_matrix[:, idx]

    pred_df.to_csv(output_dir / "per_image_predictions.csv", index=False)

    summary = {
        "num_images": int(len(samples)),
        "accuracy": float(overall_acc),
        "labels": labels,
        "outputs": {
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
            "classification_report_heatmap": str(output_dir / "classification_report_heatmap.png"),
            "prediction_probability_stats": str(output_dir / "prediction_probability_stats.png"),
            "classification_report_csv": str(output_dir / "classification_report.csv"),
            "probability_stats_csv": str(output_dir / "prediction_probability_stats.csv"),
            "per_image_predictions_csv": str(output_dir / "per_image_predictions.csv"),
        },
    }

    (output_dir / "summary.txt").write_text(
        "\n".join([
            f"num_images: {summary['num_images']}",
            f"accuracy: {summary['accuracy']:.6f}",
            f"labels: {', '.join(labels)}",
            "",
            "Generated files:",
            f"- {summary['outputs']['confusion_matrix']}",
            f"- {summary['outputs']['classification_report_heatmap']}",
            f"- {summary['outputs']['prediction_probability_stats']}",
            f"- {summary['outputs']['classification_report_csv']}",
            f"- {summary['outputs']['probability_stats_csv']}",
            f"- {summary['outputs']['per_image_predictions_csv']}",
        ]),
        encoding="utf-8",
    )

    print("Saved outputs to:", output_dir)
    print(report_df[["precision", "recall", "f1-score", "support"]].head())
    print(stats_df.head())


if __name__ == "__main__":
    main()
