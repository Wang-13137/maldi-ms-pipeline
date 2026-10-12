"""特征选择、差异检验与交叉验证评估。

变换与缩放
----------
峰面积的动态范围跨 3-4 个数量级，且右偏严重。先取 log10(x+1) 把乘性差异
变成加性差异，t 检验和 SVM 都建立在正态分布假设上，这一步是必要的。
缩放按每列的 sqrt(std) 而非 std：MALDI 峰面积的方差与均值强相关，用 std
缩放会过度放大低丰度特征的噪声；sqrt(std) 是一种折中，弱化量纲差异的同时
不把噪声拉到与信号同一量级。

为什么要交叉验证
----------------
37 例样本里感染组只有 8 例，留出法的一次划分结果方差极大。5 折分层交叉
验证保证每一折里两组比例与总体一致，AUC 取 5 折均值。注意这里的 AUC 是
"调参过程中的评估值"而非独立测试集结果，样本量限制见 README 已知局限。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


# ---------------------------------------------------------------- 变换


def transform(frame: pd.DataFrame) -> np.ndarray:
    """log10 变换 + 按列 sqrt(std) 缩放。"""
    X = np.nan_to_num(np.asarray(frame.values, dtype=float), nan=0.0)
    X_log = np.log10(X + 1.0)
    X_scaled = X_log.copy()
    for j in range(X_scaled.shape[1]):
        col_std = np.std(X_scaled[:, j])
        if col_std > 0:
            X_scaled[:, j] = X_scaled[:, j] / np.sqrt(col_std)
    return X_scaled


def log_matrix(frame: pd.DataFrame) -> np.ndarray:
    """仅做 log10 变换，用于差异检验。"""
    X = np.nan_to_num(np.asarray(frame.values, dtype=float), nan=0.0)
    return np.log10(X + 1.0)


# ---------------------------------------------------------------- 差异检验


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH 法 FDR 校正，并保证单调性。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    fdr = np.ones(n)
    for rank, idx in enumerate(order):
        fdr[idx] = min(1.0, p[idx] * n / (rank + 1))
    for i in range(n - 1, 0, -1):
        fdr[order[i - 1]] = min(fdr[order[i - 1]], fdr[order[i]])
    return fdr


def differential_test(
    frame: pd.DataFrame,
    y: np.ndarray,
    mzs: Sequence[float],
) -> pd.DataFrame:
    """逐特征做两组 t 检验（log 空间），返回含 FDR 的结果表。"""
    X_log = log_matrix(frame)
    mask_case = np.asarray(y) == 1
    mask_ctrl = ~mask_case

    rows = []
    for j, col in enumerate(frame.columns):
        try:
            _, p_val = ttest_ind(
                X_log[mask_ctrl, j], X_log[mask_case, j], nan_policy="omit"
            )
        except Exception:
            p_val = 1.0
        p_val = float(p_val) if np.isfinite(p_val) else 1.0

        mean_ctrl = float(np.mean(X_log[mask_ctrl, j]))
        mean_case = float(np.mean(X_log[mask_case, j]))
        delta = mean_case - mean_ctrl
        rows.append({
            "Feature": col,
            "m/z": float(mzs[j]),
            "健康组_mean(log10)": mean_ctrl,
            "感染组_mean(log10)": mean_case,
            "log2(FC)": delta / np.log10(2),
            "Fold Change": 10.0 ** delta,
            "p_value": p_val,
        })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    df["FDR"] = benjamini_hochberg(df["p_value"].values)
    return df


# ---------------------------------------------------------------- 建模


@dataclass
class PanelResult:
    """panel 评估结果。"""

    best_size: int
    best_auc: float
    best_indices: List[int]
    curve: Dict[int, float] = field(default_factory=dict)
    ttest_curve: Dict[int, float] = field(default_factory=dict)
    best_ttest_auc: float = 0.0
    best_ttest_size: int = 0
    importances: np.ndarray = None


def make_cv(cfg: dict, y: np.ndarray) -> StratifiedKFold:
    """分层 K 折。折数不能超过少数类的样本数，否则 sklearn 会直接报错。"""
    m = cfg.get("modeling", {})
    folds = int(m.get("cv_folds", 5))
    counts = np.bincount(np.asarray(y, dtype=int))
    minority = int(counts[counts > 0].min()) if len(counts) > 1 else 0
    folds = max(2, min(folds, minority))
    return StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=m.get("random_state", 42)
    )


def _make_svm(cfg: dict) -> SVC:
    m = cfg.get("modeling", {})
    s = m.get("svm", {})
    kwargs = dict(
        kernel=s.get("kernel", "rbf"),
        class_weight=s.get("class_weight", "balanced"),
        random_state=m.get("random_state", 42),
    )
    # roc_auc 走 decision_function，不需要概率输出。
    # sklearn 1.9 起只要显式传入 probability（哪怕传 False）就会 FutureWarning，
    # 1.11 移除该参数。因此默认不传；仅当配置显式要求概率输出时才传，
    # 届时应改用 CalibratedClassifierCV(SVC(), ensemble=False)。
    if s.get("probability", False):
        kwargs["probability"] = True
    return SVC(**kwargs)


def random_forest_importance(X: np.ndarray, y: np.ndarray, cfg: dict) -> np.ndarray:
    """拟合随机森林，返回特征重要性。"""
    m = cfg.get("modeling", {})
    rf_cfg = m.get("rf", {})
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 500),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=m.get("random_state", 42),
    )
    rf.fit(X, y)
    return rf.feature_importances_


def evaluate_panels(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    ttest_order: Sequence[int] = (),
    verbose: bool = True,
) -> PanelResult:
    """按 RF 重要性与 t 检验两种排序，逐个 panel 规模评估 SVM 的 CV AUC。"""
    m = cfg.get("modeling", {})
    cv = make_cv(cfg, y)
    sizes = list(m.get("panel_sizes", [3, 5, 10, 15, 20, 30, 50, 80]))

    importances = random_forest_importance(X, y, cfg)
    rf_order = np.argsort(importances)[::-1]
    t_order = np.asarray(list(ttest_order), dtype=int)

    rf_curve: Dict[int, float] = {}
    t_curve: Dict[int, float] = {}
    best_auc, best_size, best_idx = -1.0, 0, []
    best_t_auc, best_t_size = -1.0, 0

    if verbose:
        print(f"  {'Top N':<8}{'RF Panel':<12}{'t-test Panel':<12}")

    for k in sizes:
        k_use = min(int(k), X.shape[1])
        auc = float(np.mean(cross_val_score(
            _make_svm(cfg), X[:, rf_order[:k_use]], y, cv=cv, scoring="roc_auc"
        )))
        rf_curve[k_use] = auc
        if auc > best_auc:
            best_auc, best_size, best_idx = auc, k_use, list(rf_order[:k_use])

        t_auc = float("nan")
        if len(t_order):
            t_auc = float(np.mean(cross_val_score(
                _make_svm(cfg), X[:, t_order[:k_use]], y, cv=cv, scoring="roc_auc"
            )))
            t_curve[k_use] = t_auc
            if t_auc > best_t_auc:
                best_t_auc, best_t_size = t_auc, k_use

        if verbose:
            t_txt = f"{t_auc:.4f}" if len(t_order) else "-"
            print(f"  {k_use:<8}{auc:<12.4f}{t_txt:<12}")

    if verbose:
        print(f"  Best RF panel: Top {best_size}, AUC={best_auc:.4f}")

    return PanelResult(
        best_size=best_size,
        best_auc=best_auc,
        best_indices=best_idx,
        curve=rf_curve,
        ttest_curve=t_curve,
        best_ttest_auc=best_t_auc if len(t_order) else float("nan"),
        best_ttest_size=best_t_size,
        importances=importances,
    )
