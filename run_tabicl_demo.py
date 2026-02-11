import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split


def is_hf_download_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    indicators = [
        "huggingface",
        "hf_hub",
        "connection",
        "timed out",
        "name resolution",
        "offline",
        "download",
        "checkpoint",
    ]
    return any(token in msg for token in indicators)


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def compute_quantile_bin_stats(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y_prob, quantiles)
    bin_idx = np.digitize(y_prob, edges[1:-1], right=True)

    counts = np.bincount(bin_idx, minlength=n_bins)
    mean_pred_prob = np.full(n_bins, np.nan, dtype=float)
    frac_positives = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            mean_pred_prob[i] = float(np.mean(y_prob[mask]))
            frac_positives[i] = float(np.mean(y_true[mask]))

    return counts, mean_pred_prob, frac_positives


def compute_ece_from_calibration_curve(
    prob_true: np.ndarray, prob_pred: np.ndarray, counts: np.ndarray
) -> float:
    total = int(np.sum(counts))
    ece = 0.0
    j = 0

    for i in range(len(counts)):
        if counts[i] <= 0:
            continue
        acc_k = float(prob_true[j])
        conf_k = float(prob_pred[j])
        ece += (counts[i] / total) * abs(acc_k - conf_k)
        j += 1

    return float(ece)


def fit_predict_with_retry(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    initial_estimators: int = 4,
) -> tuple[np.ndarray, int, float]:
    from tabicl import TabICLClassifier

    estimators = initial_estimators
    has_retried_for_oom = False

    while True:
        try:
            clf = TabICLClassifier(n_estimators=estimators, random_state=42)
            t0 = time.perf_counter()
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)[:, 1]
            elapsed = time.perf_counter() - t0
            return y_prob, estimators, elapsed
        except Exception as exc:  # noqa: BLE001 - keep robust fallback behavior
            if is_cuda_oom(exc) and not has_retried_for_oom and estimators > 1:
                print("CUDA OOM detected. Retrying with n_estimators=1.")
                has_retried_for_oom = True
                estimators = 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise


def main() -> int:
    hf_cache = Path(".hf_cache").resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_cache / "assets")
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_ASSETS_CACHE"]).mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    n_bins = 10
    initial_estimators = 4
    try:
        y_prob, used_estimators, runtime_sec = fit_predict_with_retry(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            initial_estimators=initial_estimators,
        )
    except Exception as exc:  # noqa: BLE001 - explicit user-facing fallback
        if is_hf_download_error(exc):
            print(
                "Checkpoint download failed; please pre-download on a login node or set HF cache."
            )
            print(f"Original error: {exc}")
            return 1
        raise

    y_test_arr = y_test.to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_arr, y_pred)
    brier = brier_score_loss(y_test_arr, y_prob)
    prob_true, prob_pred = calibration_curve(
        y_test_arr, y_prob, n_bins=n_bins, strategy="quantile"
    )

    counts, mean_pred_prob, frac_positives = compute_quantile_bin_stats(
        y_test_arr, y_prob, n_bins=n_bins
    )
    ece = compute_ece_from_calibration_curve(prob_true, prob_pred, counts)

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"n_estimators (requested): {initial_estimators}")
    print(f"n_estimators (used): {used_estimators}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Brier score: {brier:.6f}")
    print(f"ECE (quantile, {n_bins} bins): {ece:.6f}")
    print(f"Runtime fit+predict_proba (seconds): {runtime_sec:.3f}")
    print(f"GPU used flag (torch.cuda.is_available): {torch.cuda.is_available()}")

    table_df = pd.DataFrame(
        {
            "bin_index": np.arange(n_bins),
            "count": counts,
            "mean_pred_prob": mean_pred_prob,
            "frac_positives": frac_positives,
        }
    )
    print("\nBin stats:")
    print(table_df.to_string(index=False))

    predictions_df = pd.DataFrame(
        {
            "sample_index": x_test.index.to_numpy(),
            "y_true": y_test_arr,
            "y_prob_1": y_prob,
            "y_pred": y_pred,
        }
    ).sort_values("sample_index")
    predictions_df["y_true_name"] = np.where(
        predictions_df["y_true"] == 1, "benign", "malignant"
    )
    predictions_df["y_pred_name"] = np.where(
        predictions_df["y_pred"] == 1, "benign", "malignant"
    )
    predictions_df.to_csv("tabicl_predictions.csv", index=False)

    print("\nTabICL predictions (first 20 test rows):")
    print(
        predictions_df.head(20).to_string(
            index=False, formatters={"y_prob_1": lambda x: f"{x:.6f}"}
        )
    )
    print("Saved predictions to: tabicl_predictions.csv")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="TabICL")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability Diagram - TabICL (n_estimators={used_estimators})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reliability.png", dpi=200)
    plt.close()

    table_plot_df = table_df.copy()
    table_plot_df["mean_pred_prob"] = table_plot_df["mean_pred_prob"].round(6)
    table_plot_df["frac_positives"] = table_plot_df["frac_positives"].round(6)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    mat_table = ax.table(
        cellText=table_plot_df.values,
        colLabels=table_plot_df.columns,
        cellLoc="center",
        loc="center",
    )
    mat_table.auto_set_font_size(False)
    mat_table.set_fontsize(9)
    mat_table.scale(1, 1.25)
    plt.title(f"TabICL Bin Stats (n_estimators={used_estimators})")
    plt.tight_layout()
    plt.savefig("bin_stats_table.png", dpi=200)
    plt.close(fig)

    print("Saved reliability diagram to: reliability.png")
    print("Saved bin table visualization to: bin_stats_table.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
