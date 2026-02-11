import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def setup_hf_cache() -> None:
    hf_cache = Path(".hf_cache").resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_cache / "assets")
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_ASSETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def fit_tabicl_with_retry(
    x_train: pd.DataFrame, y_train: pd.Series, initial_estimators: int = 4
):
    from tabicl import TabICLClassifier

    estimators = initial_estimators
    has_retried_for_oom = False

    while True:
        try:
            clf = TabICLClassifier(n_estimators=estimators, random_state=42)
            t0 = time.perf_counter()
            clf.fit(x_train, y_train)
            fit_sec = time.perf_counter() - t0
            return clf, estimators, fit_sec
        except Exception as exc:  # noqa: BLE001
            if is_cuda_oom(exc) and not has_retried_for_oom and estimators > 1:
                print("CUDA OOM detected during fit. Retrying with n_estimators=1.")
                has_retried_for_oom = True
                estimators = 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if is_hf_download_error(exc):
                print(
                    "Checkpoint download failed; please pre-download on a login node or set HF cache."
                )
                print(f"Original error: {exc}")
                raise SystemExit(1) from exc
            raise


def plot_feature_boxplots(
    x: pd.DataFrame, y: pd.Series, top_features: list[str], out_path: Path
) -> None:
    n_features = len(top_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(top_features):
        ax = axes[i]
        malignant = x.loc[y == 0, feature].to_numpy()
        benign = x.loc[y == 1, feature].to_numpy()
        ax.boxplot([malignant, benign], tick_labels=["malignant(0)", "benign(1)"])
        ax.set_title(feature)
        ax.grid(alpha=0.3)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Feature Distributions by Class", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_corr_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mat = corr_df.to_numpy()
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=8)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pca_views(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(x_train_scaled)
    pcs = pca.transform(x_test_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax0, ax1 = axes

    m0 = y_test == 0
    m1 = y_test == 1
    ax0.scatter(pcs[m0, 0], pcs[m0, 1], s=28, alpha=0.8, label="malignant(0)")
    ax0.scatter(pcs[m1, 0], pcs[m1, 1], s=28, alpha=0.8, label="benign(1)")
    ax0.set_title("PCA: Test Rows by True Class")
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.legend(loc="best")
    ax0.grid(alpha=0.3)

    mis = y_test != y_pred
    sc = ax1.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=y_prob,
        cmap="viridis",
        s=30,
        alpha=0.85,
        edgecolors=np.where(mis, "red", "none"),
        linewidths=np.where(mis, 1.0, 0.0),
    )
    ax1.set_title("PCA: Predicted Probability of Benign (Red Edge = Error)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="P(y=1)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_probability_hist(y_test: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        y_prob[y_test == 0],
        bins=20,
        alpha=0.75,
        label="true malignant(0)",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.hist(
        y_prob[y_test == 1],
        bins=20,
        alpha=0.75,
        label="true benign(1)",
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.set_xlabel("Predicted probability of benign class (y=1)")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_permutation_importance(
    clf,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    baseline_acc: float,
    baseline_brier: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    features = list(x_test.columns)

    for i, feature in enumerate(features, start=1):
        x_perm = x_test.copy()
        x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
        y_prob_perm = clf.predict_proba(x_perm)[:, 1]
        y_pred_perm = (y_prob_perm >= 0.5).astype(int)
        acc_perm = accuracy_score(y_test, y_pred_perm)
        brier_perm = brier_score_loss(y_test, y_prob_perm)
        rows.append(
            {
                "feature": feature,
                "acc_perm": acc_perm,
                "brier_perm": brier_perm,
                "delta_acc": baseline_acc - acc_perm,
                "delta_brier": brier_perm - baseline_brier,
            }
        )
        if i % 5 == 0 or i == len(features):
            print(f"Permutation importance progress: {i}/{len(features)}")

    imp_df = pd.DataFrame(rows).sort_values("delta_brier", ascending=False).reset_index(drop=True)
    return imp_df


def plot_permutation_importance(imp_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top["feature"], top["delta_brier"], color="#4c78a8")
    ax.set_title("Permutation Importance (Higher Delta Brier = More Important)")
    ax.set_xlabel("Brier score increase after shuffling one feature")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_local_perturbation_effects(
    clf,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    medians = x_train.median(numeric_only=True)
    features = list(x_test.columns)
    index_values = x_test.index.to_numpy()

    most_benign_idx = int(index_values[int(np.argmax(y_prob))])
    most_malignant_idx = int(index_values[int(np.argmin(y_prob))])

    mis_positions = np.where(y_test != y_pred)[0]
    selected = [
        ("most_confident_benign", most_benign_idx),
        ("most_confident_malignant", most_malignant_idx),
    ]
    if len(mis_positions) > 0:
        mis_idx = int(index_values[int(mis_positions[0])])
        selected.append(("first_misclassified", mis_idx))

    rows = []
    for sample_label, sample_idx in selected:
        row_df = x_test.loc[[sample_idx]].copy()
        baseline_pos = int(np.where(index_values == sample_idx)[0][0])
        baseline_prob = float(y_prob[baseline_pos])
        true_label = int(y_test[baseline_pos])
        pred_label = int(y_pred[baseline_pos])

        perturbed_batch = []
        feat_order = []
        for feature in features:
            pert = row_df.copy()
            pert[feature] = medians[feature]
            perturbed_batch.append(pert)
            feat_order.append(feature)
        perturbed_df = pd.concat(perturbed_batch, ignore_index=True)
        pert_probs = clf.predict_proba(perturbed_df)[:, 1]

        for feature, pert_prob in zip(feat_order, pert_probs):
            delta = float(pert_prob - baseline_prob)
            rows.append(
                {
                    "sample_label": sample_label,
                    "sample_index": sample_idx,
                    "feature": feature,
                    "baseline_prob_1": baseline_prob,
                    "perturbed_prob_1": float(pert_prob),
                    "delta_prob_1": delta,
                    "abs_delta_prob_1": abs(delta),
                    "y_true": true_label,
                    "y_pred": pred_label,
                }
            )

    return pd.DataFrame(rows)


def plot_local_effect_heatmap(local_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    top_feats = (
        local_df.groupby("feature")["abs_delta_prob_1"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    sample_order = local_df["sample_label"].drop_duplicates().tolist()
    mat_df = local_df[local_df["feature"].isin(top_feats)].pivot_table(
        index="sample_label", columns="feature", values="delta_prob_1", aggfunc="mean"
    )
    mat_df = mat_df.reindex(index=sample_order, columns=top_feats)

    fig, ax = plt.subplots(figsize=(1.2 * len(top_feats), 1.7 + 0.7 * len(sample_order)))
    mat = mat_df.to_numpy()
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(mat_df.columns)))
    ax.set_xticklabels(mat_df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat_df.index)))
    ax.set_yticklabels(mat_df.index, fontsize=9)
    ax.set_title("Local Feature Perturbation Effects on P(y=1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delta prob (perturbed - baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    setup_hf_cache()
    out_dir = Path("eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    x = data.data
    y = data.target
    feature_names = list(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    y_test_arr = y_test.to_numpy()

    print("Fitting TabICL...")
    clf, used_estimators, fit_sec = fit_tabicl_with_retry(x_train, y_train, initial_estimators=4)

    t0 = time.perf_counter()
    y_prob = clf.predict_proba(x_test)[:, 1]
    pred_sec = time.perf_counter() - t0
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_arr, y_pred)
    brier = brier_score_loss(y_test_arr, y_prob)

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"n_estimators used: {used_estimators}")
    print(f"Fit seconds: {fit_sec:.3f}")
    print(f"Predict seconds: {pred_sec:.3f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Brier score: {brier:.6f}")

    pred_df = x_test.copy()
    pred_df["y_true"] = y_test_arr
    pred_df["y_prob_1"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df["correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df.to_csv(out_dir / "predictions_with_features.csv", index=True)

    corr_to_target = x.copy()
    corr_to_target["target"] = y
    corr_series = corr_to_target.corr(numeric_only=True)["target"].drop("target")
    top_dist_features = corr_series.abs().sort_values(ascending=False).head(10).index.tolist()
    corr_summary = (
        corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
        .rename("corr_with_target")
        .rename_axis("feature")
        .reset_index()
    )
    corr_summary.to_csv(out_dir / "feature_target_correlations.csv", index=False)

    plot_feature_boxplots(x, y, top_dist_features, out_dir / "feature_distributions.png")

    heatmap_features = corr_series.abs().sort_values(ascending=False).head(15).index.tolist()
    corr_heat = x[heatmap_features].corr(numeric_only=True)
    plot_corr_heatmap(corr_heat, out_dir / "correlation_heatmap.png")

    plot_pca_views(
        x_train=x_train,
        x_test=x_test,
        y_test=y_test_arr,
        y_prob=y_prob,
        y_pred=y_pred,
        out_path=out_dir / "pca_prediction_view.png",
    )
    plot_probability_hist(y_test_arr, y_prob, out_dir / "prediction_probability_hist.png")

    print("Running permutation importance...")
    perm_df = compute_permutation_importance(
        clf=clf,
        x_test=x_test,
        y_test=y_test_arr,
        baseline_acc=accuracy,
        baseline_brier=brier,
    )
    perm_df.to_csv(out_dir / "permutation_importance.csv", index=False)
    plot_permutation_importance(perm_df, out_dir / "permutation_importance.png", top_n=15)

    print("Running local perturbation analysis...")
    local_df = compute_local_perturbation_effects(
        clf=clf,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test_arr,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    local_df.to_csv(out_dir / "local_perturbation_effects.csv", index=False)
    plot_local_effect_heatmap(local_df, out_dir / "local_effects_heatmap.png", top_n=12)

    top_perm = perm_df.head(10)[["feature", "delta_brier"]].copy()
    top_local = (
        local_df.groupby("feature")["abs_delta_prob_1"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="mean_abs_delta_prob_1")
    )
    top_perm.to_csv(out_dir / "top10_permutation_features.csv", index=False)
    top_local.to_csv(out_dir / "top10_local_effect_features.csv", index=False)

    summary_lines = [
        "TabICL EDA + Interpretability Summary",
        f"Rows total: {len(x)}, train: {len(x_train)}, test: {len(x_test)}",
        f"Accuracy: {accuracy:.6f}",
        f"Brier score: {brier:.6f}",
        f"Fit seconds: {fit_sec:.3f}",
        f"Predict seconds: {pred_sec:.3f}",
        "",
        "How to interpret outputs:",
        "- permutation_importance.csv: global feature influence by performance drop after shuffling each feature.",
        "- local_perturbation_effects.csv: local sensitivity; each feature is replaced with train median and probability shift is measured.",
        "- local_effects_heatmap.png: positive delta means feature median-replacement increases P(y=1) for that sample.",
        "",
        "Top global features (delta_brier):",
    ]
    summary_lines.extend(
        [f"- {row.feature}: {row.delta_brier:.6f}" for row in top_perm.itertuples(index=False)]
    )
    summary_lines.append("")
    summary_lines.append("Top local sensitivity features (mean abs delta_prob_1):")
    summary_lines.extend(
        [f"- {row.feature}: {row.mean_abs_delta_prob_1:.6f}" for row in top_local.itertuples(index=False)]
    )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved EDA artifacts:")
    for file_path in sorted(out_dir.glob("*")):
        print(f"- {file_path}")

    print("\nTop 10 permutation features by delta_brier:")
    print(top_perm.to_string(index=False, formatters={"delta_brier": lambda x: f"{x:.6f}"}))

    print("\nTop 10 local sensitivity features (mean |delta_prob_1|):")
    print(top_local.to_string(index=False, formatters={"mean_abs_delta_prob_1": lambda x: f"{x:.6f}"}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
