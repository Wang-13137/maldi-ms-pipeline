"""跨样本 m/z 特征对齐。

为什么要对齐：同一代谢物在不同样本里的实测 m/z 会有几个 ppm 到上百 ppm
的漂移（质量校准误差、基质结晶不均、离子起始位置差异）。不做对齐的话，
同一个分子在 37 条谱图上会变成 37 个不同特征，特征矩阵里绝大多数格子是 0，
后面的统计检验和建模都无从谈起。

为什么用 ppm 而不是固定 Da：MALDI-TOF 的质量误差随 m/z 近似线性增长。
100 ppm 在 m/z 100 处是 0.01 Da，在 m/z 900 处是 0.09 Da。固定 Da 窗口在
低质量端会把相邻代谢物错误合并、在高质量端又会漏合并。

归并算法：按 m/z 排序后单遍扫描，与当前簇中位数比较 ppm 偏差，落在容差内
就并入并更新中位数，否则开新簇。用中位数而非均值，是因为个别离群的
m/z 不会把簇中心拽走。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def _ppm_diff(a: float, b: float) -> float:
    denom = b if b else 1e-12
    return abs(a - b) / denom * 1e6


def align_peaks(peaks_flat: Sequence[dict], ppm_tolerance: float) -> List[List[dict]]:
    """把扁平化的 (sample, mz, intensity) 列表归并成簇。

    每个元素形如 ``{"sample": str, "mz": float, "intensity": float}``。
    输入需已按 m/z 升序排列，函数内部会再排一次。
    """
    if not peaks_flat:
        return []

    sorted_peaks = sorted(peaks_flat, key=lambda x: x["mz"])
    clusters: List[List[dict]] = []
    current = [sorted_peaks[0]]
    current_median = sorted_peaks[0]["mz"]

    for peak in sorted_peaks[1:]:
        if _ppm_diff(peak["mz"], current_median) <= ppm_tolerance:
            current.append(peak)
            current_median = float(np.median([p["mz"] for p in current]))
        else:
            clusters.append(current)
            current = [peak]
            current_median = peak["mz"]

    clusters.append(current)
    return clusters


def cluster_summary(clusters: Iterable[Sequence[dict]]) -> List[dict]:
    """每个簇的中位 m/z、覆盖样本数、检出次数。"""
    info = []
    for c in clusters:
        c = list(c)
        mzs = [p["mz"] for p in c]
        info.append({
            "mz_median": float(np.median(mzs)),
            "mz_mean": float(np.mean(mzs)),
            "mz_std": float(np.std(mzs)),
            "n_samples": len({p["sample"] for p in c}),
            "total_detections": len(c),
        })
    return info


def build_feature_table(
    clusters: Sequence[Sequence[dict]],
    sample_names: Sequence[str],
) -> pd.DataFrame:
    """把簇转成 样本 x 特征 的强度矩阵。

    同一簇内一个样本若有多条记录（不该发生，但保险起见）取最大值。
    未检出的格子填 0，表示"低于检出限"而不是"强度为零"。
    """
    sample_names = list(sample_names)
    row_of = {name: i for i, name in enumerate(sample_names)}

    matrix = np.zeros((len(sample_names), len(clusters)))
    for j, cluster in enumerate(clusters):
        by_sample: Dict[str, List[float]] = {}
        for p in cluster:
            by_sample.setdefault(p["sample"], []).append(p["intensity"])
        for name, values in by_sample.items():
            i = row_of.get(name)
            if i is not None:
                matrix[i, j] = max(values)

    mz_medians = [float(np.median([p["mz"] for p in c])) for c in clusters]
    columns = [f"F_{i + 1}_M{mz:.4f}" for i, mz in enumerate(mz_medians)]
    return pd.DataFrame(matrix, index=sample_names, columns=columns)


def feature_mz(frame: pd.DataFrame) -> np.ndarray:
    """从列名 ``F_12_M345.6789`` 还原 m/z 数组。"""
    return np.array([float(c.split("M")[-1]) for c in frame.columns])
