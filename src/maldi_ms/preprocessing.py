"""谱图预处理：Savitzky-Golay 平滑、基线校正、归一化。

平滑放在最前面，原因是后续所有步骤（噪声估计、顶点检测、质心）
都假设谱图轮廓是连续的。原始 MALDI 轮廓谱在峰顶附近的高频抖动会让
prominence 计算失真，先平滑再找顶点是 flexAnalysis / MALDIquant 的共同做法。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savitzky_golay(intensity: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """SG 平滑，负值截零。

    window 必须是奇数且大于 polyorder；若数据点太少则退化为原样返回。
    """
    x = np.asarray(intensity, dtype=float)
    n = len(x)
    if n < 5 or window <= polyorder:
        return np.maximum(x, 0.0)

    w = int(window)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= polyorder:
        return np.maximum(x, 0.0)

    smoothed = savgol_filter(x, window_length=w, polyorder=polyorder)
    return np.maximum(smoothed, 0.0)


def baseline_correction(intensity: np.ndarray, window: int = 501) -> np.ndarray:
    """移动最小值基线校正。

    用滑动窗口最小值作为基线估计再减去。MALDI 谱基线漂移通常不明显，
    默认关闭；只在需要时显式打开。
    """
    x = np.asarray(intensity, dtype=float)
    if len(x) < window:
        window = max(3, len(x) // 10 * 2 + 1)
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    base = np.array([np.min(padded[i:i + window]) for i in range(len(x))])
    return np.maximum(x - base, 0.0)


def normalize(intensity: np.ndarray, method: str = "none") -> np.ndarray:
    """归一化。none 表示不做处理，tic 表示按总离子流缩放。"""
    x = np.asarray(intensity, dtype=float)
    if method is None or method == "none":
        return x
    if method == "tic":
        total = np.sum(x)
        if total <= 0:
            return x
        return x / total * 1e6
    raise ValueError(f"未知归一化方式: {method}")


def prepare_spectrum(
    intensity: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """按配置顺序执行预处理，返回可直接进入峰检测的强度数组。"""
    p = cfg.get("preprocessing", {})
    x = np.asarray(intensity, dtype=float)

    if p.get("baseline_correction", False):
        x = baseline_correction(x)

    x = savitzky_golay(
        x,
        window=p.get("sg_window", 41),
        polyorder=p.get("sg_polyorder", 2),
    )
    return normalize(x, p.get("normalization", "none"))
