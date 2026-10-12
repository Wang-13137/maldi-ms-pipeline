"""合成信号上的峰检测正确性。"""

import numpy as np

from maldi_ms.peak_detection import detect_peaks, pick_peaks, remove_isotope_peaks

from conftest import gaussian_spectrum, synthetic_grid


def _detected_mzs(peaks):
    return np.array([p["mz"] for p in peaks], dtype=float)


def test_known_peaks_are_detected(cfg):
    """三个已知位置的高斯峰都应被检出，定位误差在 0.05 Da 内。"""
    mz = synthetic_grid(100.0, 200.0, 0.02)
    truth = [(120.0, 5000.0), (150.0, 3000.0), (180.5, 2000.0)]
    intensity = gaussian_spectrum(mz, truth, sigma=0.06, noise_sigma=3.0, seed=1)

    peaks, noise = detect_peaks(mz, intensity, cfg)
    found = _detected_mzs(peaks)

    assert len(peaks) >= 3, f"检出峰数不足: {len(peaks)}"
    for center, _ in truth:
        assert np.min(np.abs(found - center)) < 0.05, f"m/z {center} 未被检出"

    assert noise > 0
    # 定量值取峰面积，应明显大于顶点高度
    for p in peaks:
        assert p["intensity"] >= p["intensity_apex"] > 0
        assert p["snr"] >= cfg["peak_detection"]["snr_threshold"]
        assert p["peak_width_pts"] >= cfg["peak_detection"]["min_peak_width_pts"]


def test_noise_only_spectrum_yields_no_peaks(cfg):
    """纯噪声谱不应产生峰，否则 SNR 阈值形同虚设。"""
    mz = synthetic_grid(100.0, 200.0, 0.02)
    rng = np.random.default_rng(7)
    intensity = np.maximum(50.0 + rng.normal(0, 3.0, len(mz)), 0.0)

    peaks, _ = detect_peaks(mz, intensity, cfg)
    assert len(peaks) == 0


def test_isotope_peaks_are_removed(cfg):
    """母峰 +1.003 Da 处的弱峰应被判为同位素剔除，母峰保留。"""
    mz = synthetic_grid(100.0, 200.0, 0.02)
    intensity = gaussian_spectrum(
        mz, [(150.0, 5000.0), (151.0034, 1500.0)], sigma=0.06, noise_sigma=3.0, seed=3
    )

    peaks, _ = detect_peaks(mz, intensity, cfg)
    kept = remove_isotope_peaks(peaks, cfg)

    assert len(kept) < len(peaks), "同位素峰未被剔除"
    assert np.min(np.abs(_detected_mzs(kept) - 150.0)) < 0.05, "母峰被误删"


def test_pick_peaks_returns_sorted_unique(cfg):
    """完整流程（含并肩峰）输出的峰互不重复。"""
    mz = synthetic_grid(100.0, 200.0, 0.02)
    intensity = gaussian_spectrum(
        mz, [(120.0, 5000.0), (150.0, 3000.0), (180.5, 2000.0)],
        sigma=0.06, noise_sigma=3.0, seed=5,
    )
    peaks, _ = pick_peaks(mz, intensity, cfg)
    found = _detected_mzs(peaks)
    gaps = np.diff(np.sort(found))
    assert len(found) >= 3
    assert np.all(gaps >= cfg["peak_detection"]["min_peak_distance"] - 1e-9)
