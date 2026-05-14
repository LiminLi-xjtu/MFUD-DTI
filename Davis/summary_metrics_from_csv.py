import os
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 服务器/终端环境下使用无界面后端
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

BASE_DIR = "/ssd2/lxy_code/DTI/MFUD/result"
DATASET = "Davis"
SETTINGS = [ "random", "protein", "scaffold", "scaffold_protein"]
SEED = 3
RUN_IDS = [1, 2, 3, 4, 5]
# 要计算和绘图的指标
METRIC_KEYS = [
    "auc",       # ROC-AUC
    "aupr",      # AUPR
    "f1",
    "accuracy",
    "mcc",
    "recall",
    "precision",
]

# 图里重点展示的几个
PLOT_METRICS = ["auc", "aupr", "f1", "accuracy", "mcc"]


def compute_metrics_from_csv(csv_path: str) -> Dict[str, float]:
    """
    从 test_prediction_seed<SEED>.csv 中计算各项二分类指标。
    需要列: classification_label, predicted_binary_interaction
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if "classification_label" not in df.columns or "predicted_binary_interaction" not in df.columns:
        raise ValueError(f"文件缺少必要列: {csv_path}")

    y = df["classification_label"].values.astype(float)
    p = df["predicted_binary_interaction"].values.astype(float)

    # 过滤 NaN
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]
    p = p[mask]

    # 概率 -> 0/1 预测标签
    y_pred = (p > 0.5).astype(int)

    if len(np.unique(y)) < 2:
        # 全 0 或全 1 时，指标无意义，统一置 0
        return {k: 0.0 for k in METRIC_KEYS}

    metrics = {
        "auc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, y_pred)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
    }
    return metrics


def collect_metrics_for_setting(setting: str) -> Dict[str, List[float]]:
    """
    对某个 setting (scaffold/protein/random)，收集所有 RUN 的指标:
    返回 {metric_name: [run1, ..., run5]}。
    """
    metrics_per_key = {k: [] for k in METRIC_KEYS}

    for run_id in RUN_IDS:
        # 直接拼接为 {setting}_RUN_{run_id}
        result_dir = f"{setting}_RUN_{run_id}"
        csv_path = os.path.join(result_dir, f"test_prediction_seed{SEED}.csv")
        m = compute_metrics_from_csv(csv_path)
        for k in METRIC_KEYS:
            metrics_per_key[k].append(m[k])

    return metrics_per_key


def plot_metrics(metrics_all_settings: Dict[str, Dict[str, List[float]]]):
    """
    画对比图:
    x 轴是 setting (scaffold/protein/random)，
    每个 setting 上 5 个散点 + 均值柱状。
    """
    settings = list(metrics_all_settings.keys())
    n_set = len(settings)
    n_run = len(RUN_IDS)

    x = np.arange(n_set)

    # 2 行 3 列的子图，展示 PLOT_METRICS 前 5 个
    n_plot = len(PLOT_METRICS)
    n_rows, n_cols = 2, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    jitter = 0.06

    for idx_metric, metric in enumerate(PLOT_METRICS):
        ax = axes[idx_metric]
        # 收集成 [n_set, n_run]
        vals = np.array(
            [metrics_all_settings[s][metric] for s in settings], dtype=float
        )  # shape: [n_set, n_run]

        for i in range(n_set):
            xs = x[i] + (np.arange(n_run) - (n_run - 1) / 2) * jitter
            ax.scatter(xs, vals[i], alpha=0.7, zorder=3)

            mean_v = vals[i].mean()
            ax.bar(x[i], mean_v, width=0.2, alpha=0.3, zorder=2)
            ax.text(
                x[i],
                mean_v + 0.005,
                f"{mean_v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(metric.upper())
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 如果子图多于需要的，关掉多余的
    for j in range(n_plot, n_rows * n_cols):
        fig.delaxes(axes[j])

    for ax in axes[:n_plot]:
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=20)

    fig.suptitle(
        f"Davis / {CONFIG_SUFFIX} - metrics from CSV ({n_run} runs per setting)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(
        BASE_DIR,
        "result",
        DATASET,
        f"{CONFIG_SUFFIX}_metrics_from_csv.png",
    )
    plt.savefig(out_path, dpi=300)
    print(f"保存图像到: {out_path}")


def main():
    print(f"=== Davis 数据集 - 基于 CSV 的 {len(RUN_IDS)} 次运行汇总 ===\n")

    metrics_all_settings: Dict[str, Dict[str, List[float]]] = {}

    for setting in SETTINGS:
        print(f"--- Setting: {setting} ---")
        metrics_per_key = collect_metrics_for_setting(setting)
        metrics_all_settings[setting] = metrics_per_key

        for k in METRIC_KEYS:
            vals = np.array(metrics_per_key[k], dtype=float)
            mean = vals.mean()
            std = vals.std()
            runs_str = "  ".join(f"{v:.4f}" for v in vals)
            print(f"{k:9s} | runs: {runs_str} | mean={mean:.4f}, std={std:.4f}")
        print()

    print("说明：")
    print("- auc   : ROC-AUC")
    print("- aupr  : AUPR")
    print("- mcc   : Matthews 相关系数")
    print("- f1    : F1-score")
    print("- accuracy: 准确率")
    print("- recall / precision: 召回率 / 精确率\n")

    # 画图
    # plot_metrics(metrics_all_settings)


if __name__ == "__main__":
    main()