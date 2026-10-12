"""特征过滤：阴性对照背景剔除 + 检出率过滤。

两步的取舍逻辑不同：

背景过滤解决"这个信号来自样本还是来自基质/溶剂"。MALDI 基质（如 CHCA、DHB）
本身会在低质量区产生大量峰，塑料耗材和溶剂也会带来污染物。阴性对照里同样
出现、且强度与样本相当的特征，没有生物学意义。判断时给样本留了余地：
只要样本最高强度超过对照的 2 倍，就认为样本里确有额外来源，予以保留。

检出率过滤解决"这个特征能不能用于统计"。只在一两个样本里出现的特征，
既可能是随机的离子化波动，也会在交叉验证里造成训练/测试折之间的分布漂移。
要求特征至少在 15% 的样本中出现，是把"偶然出现"和"稳定存在"分开的最低门槛。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .alignment import feature_mz


def filter_background(
    frame: pd.DataFrame,
    negative_peaks: Sequence[dict],
    cfg: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """剔除与阴性对照强度相当的特征。"""
    if not negative_peaks:
        if verbose:
            print("  未提供阴性对照数据，跳过背景过滤")
        return frame

    f = cfg.get("filtering", {}).get("background", {})
    if not f.get("enabled", True):
        return frame

    ppm_tol = cfg.get("alignment", {}).get("ppm_tolerance", 100)
    dominant_ratio = f.get("sample_dominant_ratio", 2.0)
    blank_ratio = f.get("blank_intensity_ratio", 0.5)

    mzs = feature_mz(frame)
    blank_intensities = np.zeros(len(mzs))
    for j, fmz in enumerate(mzs):
        window = fmz * ppm_tol / 1e6
        best = 0.0
        for bp in negative_peaks:
            if abs(bp["mz"] - fmz) <= window:
                best = max(best, bp["intensity"])
        blank_intensities[j] = best

    sample_max = frame.max(axis=0).values
    sample_median = frame.median(axis=0).values

    keep: List[int] = []
    n_bg = 0
    for j in range(len(mzs)):
        blank = blank_intensities[j]
        smax = sample_max[j]
        smed = sample_median[j]

        if smax == 0:
            n_bg += 1
            continue
        if blank == 0:
            keep.append(j)
            continue
        if smax / (blank + 1) > dominant_ratio:
            keep.append(j)
            continue
        if blank > smed * blank_ratio and smed > 0:
            n_bg += 1
            continue
        keep.append(j)

    if verbose:
        n_blank_detected = int(np.sum(blank_intensities > 0))
        print(f"  总feature: {len(mzs)}")
        print(f"  阴性对照中检测到: {n_blank_detected}")
        print(f"  阴性对照中未检测到(直接保留): {len(mzs) - n_blank_detected}")
        print(f"  标记为背景: {n_bg}")
        print(f"  保留: {len(keep)}")

    if not keep:
        if verbose:
            print("  [警告] 所有feature被标记为背景，跳过该步")
        return frame

    return frame.iloc[:, keep]


def filter_by_detection(
    frame: pd.DataFrame,
    cfg: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """按检出率过滤特征。"""
    fc = cfg.get("filtering", {})
    ratio = fc.get("min_detection_ratio", 0.15)
    abs_min = fc.get("min_detection_abs", 2)

    n_samples = frame.shape[0]
    min_detection = max(abs_min, int(n_samples * ratio))
    counts = (frame > 0).sum(axis=0).values
    keep = [j for j in range(frame.shape[1]) if counts[j] >= min_detection]

    if verbose:
        print(f"  过滤前: {frame.shape[1]}")
        print(f"  检出率阈值: {min_detection}/{n_samples} 样本")
        print(f"  低覆盖: {frame.shape[1] - len(keep)}")
        print(f"  过滤后: {len(keep)}")

    return frame.iloc[:, keep]
