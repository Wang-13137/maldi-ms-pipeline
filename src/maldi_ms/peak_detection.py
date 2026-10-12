"""峰检测：对标 Bruker flexAnalysis 的七步法。

流程与每一步的动机：

1. SG 平滑
   轮廓谱（profile spectrum）在峰顶附近存在高频抖动，直接按局部极大值
   找峰会检出大量假峰。先做 Savitzky-Golay 平滑能在保留峰形的前提下
   压掉高频噪声。

2. 残差 MAD 噪声估计
   噪声水平不能取全谱标准差：真实峰会把它抬高好几倍，导致弱峰被漏检。
   这里用 ``raw - smoothed`` 的残差，残差里几乎只剩噪声，再取
   ``1.4826 * MAD``。1.4826 是 MAD 到正态分布标准差的换算系数，
   用 MAD 而非标准差是为了让少数大残差（峰位置）不污染估计。

3. SNR >= 3 顶点检测
   在平滑谱上找 prominence 不低于 ``noise * snr_threshold`` 的局部极大值，
   同时限制相邻峰间距 >= 0.2 Da（对应 MALDI-TOF 10000-20000 的分辨率）。

4. 质心法定位
   顶点 m/z 受采样网格限制，且峰形不对称时会偏。用峰区间内的强度加权
   平均 m/z（质心）作为该峰的 m/z，这是商业软件的通行做法。

5. 峰面积积分
   用峰区间内的强度和作为定量值，比顶点高度稳健：峰形展宽或轻微
   质量偏移时，面积比高度稳定得多。

6. 五重质控
   峰宽 >= 3 点（剔除单点尖峰）、真实 SNR >= 阈值、绝对强度下限、
   相对基峰强度 >= 0.01%、质心与顶点位移 <= 0.05 Da（峰形对称性）。

7. 同位素去除
   在 +1.003355 Da（以及 +2 倍）处以 10 ppm 窗口搜索，若子峰强度与母峰
   之比落在合理区间，判定为同位素峰并剔除。否则同位素会被当成独立代谢物，
   虚增特征数并重复计入同一分子的信息。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from .preprocessing import savitzky_golay


# ---------------------------------------------------------------- 噪声估计


def estimate_noise_mad(intensity: np.ndarray) -> float:
    """基于低强度半区的 MAD 噪声估计（不含峰）。

    取非零点的低强度一半计算 MAD，避开峰区。异常值（极高的峰）不会进入
    低强度半区，因此估计值稳健。

    代价是在低强度半区内做截断，估计值偏保守（约为真实 sigma 的 0.6 倍）。
    流水线默认使用 :func:`estimate_noise_residual`，本函数作为无平滑谱时的
    兜底与对照。
    """
    x = np.asarray(intensity, dtype=float)
    nonzero = x[x > 0]
    if len(nonzero) < 10:
        top = np.max(x) if len(x) else 0.0
        return max(top * 0.01, 1.0) if top > 0 else 1.0

    sorted_nz = np.sort(nonzero)
    low = sorted_nz[: max(1, len(sorted_nz) // 2)]
    if len(low) <= 1:
        return max(float(low[0]) * 0.1, 1.0)
    mad = np.median(np.abs(low - np.median(low)))
    return max(1.4826 * float(mad), 1.0)


def estimate_noise_residual(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """基于平滑残差的 MAD 噪声估计。"""
    residuals = np.asarray(raw, dtype=float) - np.asarray(smoothed, dtype=float)
    if len(residuals) < 2:
        return 1.0
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return max(1.4826 * float(mad), 1.0)


# ---------------------------------------------------------------- 峰检测


def _enforce_min_distance(
    mz: np.ndarray,
    smooth: np.ndarray,
    peaks_idx: np.ndarray,
    props: dict,
    min_distance_da: float,
) -> Tuple[np.ndarray, dict]:
    """按 m/z 间距（而非点数）剔除过近的峰，保留其中更高的那个。

    在非均匀 m/z 网格上，scipy 的 distance 参数按数据点计数，等价于
    按 m/z 距离过滤时才需要这一步。
    """
    if len(peaks_idx) <= 1:
        return peaks_idx, props

    order = np.argsort(mz[peaks_idx])
    keep = np.ones(len(peaks_idx), dtype=bool)
    for k in range(1, len(order)):
        cur, prev = order[k], order[k - 1]
        if keep[cur] and (mz[peaks_idx[cur]] - mz[peaks_idx[prev]]) < min_distance_da:
            if smooth[peaks_idx[cur]] > smooth[peaks_idx[prev]]:
                keep[prev] = False
            else:
                keep[cur] = False

    idx = peaks_idx[keep]
    new_props = {k: np.asarray(v)[keep] for k, v in props.items()}
    return idx, new_props


def _peak_boundaries(
    intensity: np.ndarray,
    apex_idx: int,
    search_pts: int,
    boundary_fraction: float,
) -> Tuple[int, int]:
    """从顶点向两侧走，直到强度低于顶点的 boundary_fraction 或开始回升。"""
    n = len(intensity)
    apex_int = intensity[apex_idx]

    left = apex_idx
    for j in range(apex_idx - 1, max(0, apex_idx - search_pts), -1):
        if intensity[j] < apex_int * boundary_fraction:
            break
        if intensity[j] > intensity[j + 1]:
            break
        left = j

    right = apex_idx
    for j in range(apex_idx + 1, min(n, apex_idx + search_pts)):
        if intensity[j] < apex_int * boundary_fraction:
            break
        if intensity[j] > intensity[j - 1]:
            break
        right = j

    return left, right


def detect_peaks(mz: np.ndarray, intensity: np.ndarray, cfg: dict) -> Tuple[List[dict], float]:
    """单条谱图的峰检测。返回 (峰列表, 噪声估计值)。

    峰字典包含：mz（质心）、intensity（峰面积）、intensity_apex、
    prominence、snr、peak_width_pts。
    """
    p = cfg.get("peak_detection", {})
    snr_threshold = p.get("snr_threshold", 3)
    min_distance = p.get("min_peak_distance", 0.2)
    min_intensity = p.get("min_intensity", 0)
    min_width_pts = p.get("min_peak_width_pts", 3)
    min_relative = p.get("min_relative_intensity", 0.0001)
    max_shift = p.get("max_centroid_shift", 0.05)
    boundary_fraction = p.get("boundary_fraction", 0.1)
    boundary_da = p.get("boundary_search_da", 0.05)

    mz = np.asarray(mz, dtype=float)
    raw = np.asarray(intensity, dtype=float)

    # 1. 平滑
    smooth = savitzky_golay(
        raw,
        window=cfg.get("preprocessing", {}).get("sg_window", 41),
        polyorder=cfg.get("preprocessing", {}).get("sg_polyorder", 2),
    )

    # 2. 残差 MAD 噪声
    noise = estimate_noise_residual(raw, smooth)

    # 3. 顶点检测
    avg_step = float(np.median(np.diff(mz))) if len(mz) > 1 else 1.0
    if avg_step <= 0:
        return [], noise
    distance_pts = max(1, int(min_distance / avg_step))

    peaks_idx, props = find_peaks(
        smooth,
        prominence=noise * snr_threshold,
        distance=distance_pts,
        height=0,
    )
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    peaks_idx, props = _enforce_min_distance(mz, smooth, peaks_idx, props, min_distance)

    if len(peaks_idx) == 0:
        return [], noise

    prominences = props.get("prominences", props.get("prominence"))
    if prominences is None or len(prominences) != len(peaks_idx):
        return [], noise

    search_pts = max(10, int(boundary_da / avg_step))
    base_peak = float(np.max(raw[peaks_idx])) if len(peaks_idx) else 1.0
    min_abs_relative = base_peak * min_relative

    results: List[dict] = []
    for i, apex_idx in enumerate(peaks_idx):
        # 4-5. 峰边界
        left, right = _peak_boundaries(raw, apex_idx, search_pts, boundary_fraction)

        # 6a. 峰宽质控
        width_pts = int(right - left + 1)
        if width_pts < min_width_pts:
            continue

        seg_mz = mz[left: right + 1]
        seg_int = raw[left: right + 1]
        area = float(np.sum(seg_int))
        if area <= 0:
            continue

        # 质心定位
        centroid_mz = float(np.sum(seg_mz * seg_int) / area)
        apex_height = float(np.max(seg_int))
        snr = apex_height / noise if noise > 0 else 0.0

        # 6b. 真实 SNR
        if snr < snr_threshold:
            continue
        # 6c. 绝对强度下限
        if min_intensity and apex_height < min_intensity:
            continue
        # 6d. 相对基峰强度
        if apex_height < min_abs_relative:
            continue
        # 6e. 质心位移
        if abs(centroid_mz - mz[apex_idx]) > max_shift:
            continue

        results.append({
            "mz": centroid_mz,
            "intensity": area,
            "intensity_apex": apex_height,
            "prominence": float(prominences[i]),
            "snr": snr,
            "peak_width_pts": width_pts,
        })

    return results, noise


# ---------------------------------------------------------------- 同位素


def remove_isotope_peaks(peaks: Sequence[dict], cfg: dict) -> List[dict]:
    """剔除 +1 / +2 同位素峰。"""
    iso = cfg.get("peak_detection", {}).get("isotope", {})
    if not iso.get("enabled", True) or len(peaks) < 2:
        return list(peaks)

    shift = iso.get("mass_shift", 1.003355)
    ppm_tol = iso.get("ppm_tolerance", 10)
    max_ratio = iso.get("max_ratio", 0.9)
    min_ratio = iso.get("min_ratio", 0.01)

    sp = sorted(peaks, key=lambda x: x["mz"])
    flagged = [False] * len(sp)

    for i, peak in enumerate(sp):
        for ds, mr in ((shift, max_ratio), (2 * shift, max_ratio * 0.5)):
            target = peak["mz"] + ds
            window = target * ppm_tol / 1e6
            for j in range(i + 1, len(sp)):
                if sp[j]["mz"] > target + window:
                    break
                if abs(sp[j]["mz"] - target) <= window:
                    ratio = sp[j]["intensity"] / peak["intensity"] if peak["intensity"] else 0.0
                    if min_ratio < ratio < mr:
                        flagged[j] = True

    return [p for i, p in enumerate(sp) if not flagged[i]]


# ---------------------------------------------------------------- 肩峰


def merge_shoulder_peaks(
    peaks: Sequence[dict],
    mz_full: np.ndarray,
    intensity_full: np.ndarray,
    cfg: dict,
) -> List[dict]:
    """合并肩峰：两峰很近且中间的谷很浅，说明它们属于同一个未分离的峰。

    判据是谷深与较矮峰的比值，以及两峰强度比。谷很浅说明中间没有真正的
    分界，此时保留矮峰只会引入重复的定量信息。
    """
    sh = cfg.get("peak_detection", {}).get("shoulder", {})
    if not sh.get("enabled", True) or len(peaks) < 2:
        return list(peaks)

    max_gap = sh.get("max_gap_da", 0.15)
    valley_thresh = sh.get("valley_ratio", 0.5)
    int_ratio_thresh = sh.get("intensity_ratio", 3.0)

    sp = sorted(peaks, key=lambda x: x["mz"])
    n = len(sp)
    drop = [False] * n

    for i in range(n - 1):
        if drop[i] or drop[i + 1]:
            continue
        if sp[i + 1]["mz"] - sp[i]["mz"] > max_gap:
            continue

        idx1 = int(np.argmin(np.abs(mz_full - sp[i]["mz"])))
        idx2 = int(np.argmin(np.abs(mz_full - sp[i + 1]["mz"])))
        if idx2 - idx1 < 2:
            continue

        valley = float(np.min(intensity_full[idx1: idx2 + 1]))
        smaller = min(sp[i]["intensity"], sp[i + 1]["intensity"])
        if smaller <= 0:
            continue
        valley_ratio = valley / smaller
        int_ratio = max(sp[i]["intensity"], sp[i + 1]["intensity"]) / smaller

        if valley_ratio > valley_thresh and (
            int_ratio > int_ratio_thresh or valley_ratio > 0.8
        ):
            if sp[i]["intensity"] < sp[i + 1]["intensity"]:
                drop[i] = True
            else:
                drop[i + 1] = True

    return [p for i, p in enumerate(sp) if not drop[i]]


def pick_peaks(mz: np.ndarray, intensity: np.ndarray, cfg: dict) -> Tuple[List[dict], float]:
    """峰检测完整流程：检测 + 去同位素 + 并肩峰。"""
    peaks, noise = detect_peaks(mz, intensity, cfg)
    peaks = remove_isotope_peaks(peaks, cfg)
    peaks = merge_shoulder_peaks(peaks, mz, intensity, cfg)
    return peaks, noise
