"""跨样本 m/z 对齐逻辑。"""

import numpy as np

from maldi_ms.alignment import align_peaks, build_feature_table, cluster_summary, feature_mz


def _flat(pairs, sample="S1"):
    return [{"sample": sample, "mz": m, "intensity": i} for m, i in pairs]


def test_close_mz_are_merged():
    """相差 80 ppm 的两个 m/z 应归入同一特征。"""
    peaks = _flat([(100.000, 1000.0)]) + _flat([(100.008, 900.0)], sample="S2")
    clusters = align_peaks(peaks, ppm_tolerance=100)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_distant_mz_are_not_merged():
    """相差 5000 ppm 的两个 m/z 必须保持为两个特征。"""
    peaks = _flat([(100.000, 1000.0)]) + _flat([(100.500, 900.0)], sample="S2")
    clusters = align_peaks(peaks, ppm_tolerance=100)
    assert len(clusters) == 2


def test_ppm_window_is_relative_to_mz():
    """同样的 0.01 Da 偏差，在高质量端超出 100 ppm 时不合并。"""
    low = _flat([(200.000, 1000.0)]) + _flat([(200.010, 900.0)], sample="S2")
    high = _flat([(900.000, 1000.0)]) + _flat([(900.010, 900.0)], sample="S2")

    assert len(align_peaks(low, ppm_tolerance=100)) == 1    # 50 ppm
    assert len(align_peaks(high, ppm_tolerance=100)) == 1   # 11 ppm，同样在容差内

    # 高质量端 0.1 Da 偏差 = 111 ppm，超出容差
    far = _flat([(900.000, 1000.0)]) + _flat([(900.100, 900.0)], sample="S2")
    assert len(align_peaks(far, ppm_tolerance=100)) == 2


def test_feature_matrix_shape_and_values():
    """特征矩阵应为 样本数 x 特征数，未检出填 0，重复记录取最大值。"""
    peaks = (
        _flat([(120.0, 500.0), (150.0, 700.0)], sample="A")
        + _flat([(120.002, 800.0)], sample="B")
        + _flat([(150.001, 300.0)], sample="C")
    )
    clusters = align_peaks(peaks, ppm_tolerance=100)
    frame = build_feature_table(clusters, ["A", "B", "C"])

    assert frame.shape == (3, 2)
    assert frame.loc["A", frame.columns[0]] == 500.0
    assert frame.loc["B", frame.columns[0]] == 800.0
    assert frame.loc["C", frame.columns[0]] == 0.0
    assert np.allclose(feature_mz(frame), [120.001, 150.0005], atol=1e-3)


def test_cluster_summary_counts_samples():
    """簇摘要统计的覆盖样本数应排除同一样本的重复检出。"""
    peaks = (
        _flat([(120.0, 500.0)], sample="A")
        + _flat([(120.004, 600.0)], sample="A")
        + _flat([(120.008, 400.0)], sample="B")
    )
    clusters = align_peaks(peaks, ppm_tolerance=100)
    info = cluster_summary(clusters)
    assert len(info) == 1
    assert info[0]["n_samples"] == 2
    assert info[0]["total_detections"] == 3
