import os
import time
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def compute_attention_weights(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    attn_mask: torch.Tensor | int | None = None,
    rope=None,
) -> torch.Tensor | None:
    if isinstance(attn_mask, int):
        return None

    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    q, k, _ = F._in_projection_packed(
        query, key, value, module.in_proj_weight, module.in_proj_bias
    )
    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)

    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

    q = q.float()
    k = k.float()
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            mask = attn_mask
            if mask.dim() == 2:
                view_shape = (1,) * len(batch_shape) + (1, tgt_len, src_len)
                mask = mask.view(*view_shape)
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            additive = attn_mask.to(scores.dtype)
            if additive.dim() == 2:
                view_shape = (1,) * len(batch_shape) + (1, tgt_len, src_len)
                additive = additive.view(*view_shape)
            scores = scores + additive

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.bool:
            mask = (
                key_padding_mask.view(*batch_shape, 1, 1, src_len)
                .expand(*batch_shape, num_heads, tgt_len, src_len)
            )
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            additive = (
                key_padding_mask.to(scores.dtype)
                .view(*batch_shape, 1, 1, src_len)
                .expand(*batch_shape, num_heads, tgt_len, src_len)
            )
            scores = scores + additive

    return torch.softmax(scores, dim=-1)


def extract_feature_attention_matrix(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame
) -> tuple[np.ndarray, list[str], list[tuple[str, float]]]:
    from tabicl import TabICLClassifier

    feature_names = list(x_train.columns)
    n_features = len(feature_names)

    devices_to_try: list[str | None]
    if torch.cuda.is_available():
        devices_to_try = [None, "cpu"]
    else:
        devices_to_try = [None]

    last_exc: Exception | None = None
    for device in devices_to_try:
        try:
            attn_clf = TabICLClassifier(
                n_estimators=1,
                feat_shuffle_method="none",
                class_shift=False,
                batch_size=1,
                random_state=42,
                device=device,
            )
            attn_clf.fit(x_train, y_train)

            row_block = attn_clf.model_.row_interactor.tf_row.blocks[-1]
            row_attn = row_block.attn
            num_cls = attn_clf.model_.row_interactor.num_cls
            captured_weights: list[torch.Tensor] = []
            original_forward = row_attn.forward

            def wrapped_forward(
                query, key, value, key_padding_mask=None, attn_mask=None, rope=None
            ):
                out = original_forward(
                    query,
                    key,
                    value,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                )
                with torch.no_grad():
                    weights = compute_attention_weights(
                        module=row_attn,
                        query=query,
                        key=key,
                        value=value,
                        key_padding_mask=key_padding_mask,
                        attn_mask=attn_mask,
                        rope=rope,
                    )
                    if weights is not None:
                        captured_weights.append(weights.detach().cpu())
                return out

            row_attn.forward = wrapped_forward
            try:
                _ = attn_clf.predict_proba(x_test)
            finally:
                row_attn.forward = original_forward

            if not captured_weights:
                raise RuntimeError("Attention instrumentation captured no weights.")

            weights_all = torch.cat(captured_weights, dim=0)
            train_size = len(y_train)
            if weights_all.shape[1] > train_size:
                weights_rows = weights_all[:, train_size:, ...]
            else:
                weights_rows = weights_all

            feat_start = num_cls
            feat_end = num_cls + n_features
            cls_to_feat = weights_rows[..., :num_cls, feat_start:feat_end]

            matrix = cls_to_feat.mean(dim=(0, 1, 2)).numpy()  # (num_cls, n_features)
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix_norm = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix),
                where=row_sums > 0,
            )

            avg_row = matrix_norm.mean(axis=0, keepdims=True)
            heatmap_matrix = np.vstack([matrix_norm, avg_row])
            row_labels = [f"CLS_{i}" for i in range(num_cls)] + ["CLS_avg"]

            top_idx = np.argsort(avg_row[0])[::-1][:10]
            top_features = [
                (feature_names[idx], float(avg_row[0, idx])) for idx in top_idx
            ]
            return heatmap_matrix, row_labels, top_features
        except Exception as exc:  # noqa: BLE001
            if is_cuda_oom(exc) and device is None and torch.cuda.is_available():
                print("CUDA OOM during attention extraction. Retrying on CPU.")
                torch.cuda.empty_cache()
                continue
            last_exc = exc
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Attention extraction failed unexpectedly.")


def save_attention_heatmap(
    heatmap_matrix: np.ndarray, row_labels: list[str], feature_names: list[str]
) -> None:
    fig_width = max(12, 0.35 * len(feature_names))
    fig_height = 2.5 + 0.55 * len(row_labels)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heatmap_matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Input features")
    ax.set_ylabel("Row transformer CLS queries")
    ax.set_title("TabICL Attention Heatmap (CLS -> Features, last row-attention block)")
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Normalized attention mass over feature keys",
    )
    fig.tight_layout()
    fig.savefig("attention_heatmap.png", dpi=220)
    plt.close(fig)


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

    try:
        feature_names = list(x_train.columns)
        attn_matrix, attn_row_labels, top_features = extract_feature_attention_matrix(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
        )
        save_attention_heatmap(
            heatmap_matrix=attn_matrix,
            row_labels=attn_row_labels,
            feature_names=feature_names,
        )
        print("\nTop 10 features by CLS_avg attention:")
        for name, score in top_features:
            print(f"  {name:>24s}  {score:.6f}")
        print("Saved attention heatmap to: attention_heatmap.png")
    except Exception as exc:  # noqa: BLE001
        if is_hf_download_error(exc):
            print(
                "Checkpoint download failed; please pre-download on a login node or set HF cache."
            )
        print(f"Attention extraction skipped due to error: {exc}")

    print("Saved reliability diagram to: reliability.png")
    print("Saved bin table visualization to: bin_stats_table.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
