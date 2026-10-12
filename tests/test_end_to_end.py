"""端到端冒烟测试：用合成的小样本数据跑通完整流程。

只断言流程不报错、输出形状与文件齐全，不锁定具体数值。
真实数据上的复现结果见 README「复现步骤」。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from maldi_ms.cli import run_pipeline

from conftest import gaussian_spectrum, synthetic_grid

SAMPLES = ["UTI11", "UTI13", "UTI20", "UTI21", "UTI22", "UTI23"]


def _write_dataset(data_dir: str, n_shared: int = 3) -> None:
    """生成 6 条合成谱图写成一个 xlsx，感染组另加一个差异峰。"""
    os.makedirs(data_dir, exist_ok=True)
    mz = synthetic_grid(100.0, 200.0, 0.02)

    shared = [(120.0, 4000.0), (150.0, 3000.0), (180.0, 2000.0)][:n_shared]
    df = pd.DataFrame({"横坐标(m/z)": mz})

    for i, name in enumerate(SAMPLES):
        peaks = list(shared)
        if name in ("UTI11", "UTI13"):
            peaks.append((160.0, 2500.0))
        df[f"样品{i + 1}_{name}"] = gaussian_spectrum(
            mz, peaks, sigma=0.06, noise_sigma=3.0, seed=100 + i
        )

    df.to_excel(os.path.join(data_dir, "synthetic.xlsx"), index=False)


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory, cfg):
    base = tmp_path_factory.mktemp("smoke")
    data_dir = str(base / "data")
    out_dir = str(base / "out")
    _write_dataset(data_dir)

    local_cfg = json.loads(json.dumps(cfg))
    local_cfg["modeling"]["panel_sizes"] = [2, 3]
    local_cfg["modeling"]["cv_folds"] = 2  # 感染组只有 2 例
    return run_pipeline(local_cfg, data_dir, out_dir, verbose=False), out_dir


def test_pipeline_runs_and_outputs_files(smoke_result):
    result, out_dir = smoke_result
    for fname in ("aligned_feature_table.xlsx", "feature_stats.xlsx", "metrics.json"):
        assert os.path.exists(os.path.join(out_dir, fname)), f"缺少输出 {fname}"


def test_feature_matrix_shape(smoke_result):
    result, _ = smoke_result
    frame = result["feature_table"]
    assert frame.shape[0] == len(SAMPLES)
    assert frame.shape[1] >= 1
    assert list(frame.index) == sorted(SAMPLES)
    assert np.all(np.isfinite(frame.values))
    assert np.all(frame.values >= 0)


def test_metrics_are_consistent(smoke_result):
    result, out_dir = smoke_result
    m = result["metrics"]
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as fh:
        on_disk = json.load(fh)

    assert m["n_samples"] == len(SAMPLES)
    assert m["n_infected"] == 2
    assert m["n_healthy"] == 4
    assert 0.0 <= m["best_panel_auc"] <= 1.0
    assert m["n_features_final"] <= m["n_features_aligned"]
    assert on_disk["n_features_final"] == m["n_features_final"]


def test_diff_table_columns(smoke_result):
    result, _ = smoke_result
    diff = result["diff"]
    assert len(diff) == result["feature_table"].shape[1]
    for col in ("Feature", "m/z", "p_value", "FDR", "log2(FC)"):
        assert col in diff.columns
    assert diff["FDR"].min() >= diff["p_value"].min() - 1e-12
