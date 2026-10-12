"""噪声估计的鲁棒性。"""

import numpy as np

from maldi_ms.peak_detection import estimate_noise_mad, estimate_noise_residual
from maldi_ms.preprocessing import savitzky_golay


def test_mad_robust_to_outliers():
    """注入极端异常值后，MAD 估计几乎不变，而普通标准差会被带偏几个数量级。"""
    rng = np.random.default_rng(42)
    true_sigma = 10.0
    x = 100.0 + rng.normal(0, true_sigma, 10000)
    est_clean = estimate_noise_mad(x)
    std_clean = float(np.std(x))

    # 注入 1% 的极端异常值（模拟峰和尖峰）
    x[:100] = 50000.0
    est_outlier = estimate_noise_mad(x)
    std_outlier = float(np.std(x))

    assert std_outlier / std_clean > 5, "对照组：普通标准差应被异常值严重抬高"
    drift = abs(est_outlier - est_clean) / est_clean
    assert drift < 0.2, f"MAD 估计被异常值带偏: {est_clean:.2f} -> {est_outlier:.2f}"
    assert 0.4 * true_sigma < est_outlier < 1.5 * true_sigma, f"量级不合理: {est_outlier:.2f}"


def test_mad_scales_with_noise_level():
    """噪声水平翻倍，估计值也应大致翻倍。"""
    rng = np.random.default_rng(0)
    lo = estimate_noise_mad(100.0 + rng.normal(0, 5.0, 20000))
    hi = estimate_noise_mad(100.0 + rng.normal(0, 10.0, 20000))
    assert 1.6 < hi / lo < 2.4


def test_residual_noise_ignores_peaks():
    """残差法估计的噪声不应被真实峰抬高。"""
    rng = np.random.default_rng(1)
    mz = np.linspace(100, 200, 20001)
    x = 100.0 + rng.normal(0, 5.0, len(mz))
    # 加一个极高的峰
    x += 100000.0 * np.exp(-0.5 * ((mz - 150.0) / 0.05) ** 2)

    smooth = savitzky_golay(x, window=41, polyorder=2)
    est = estimate_noise_residual(x, smooth)

    assert 2.0 < est < 12.0, f"残差噪声估计被峰污染: {est:.2f}"
