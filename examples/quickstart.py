"""快速上手：读谱图、检峰、对齐、过滤、建模，五步跑完。

    python examples/quickstart.py              # 无参数：自动生成合成谱图演示
    python examples/quickstart.py <数据目录>    # 用真实数据（格式见 README）

合成演示生成 6 例样本（感染 2 / 健康 4），含 3 个共有峰和 1 个只在
感染组出现的差异峰，全程不依赖任何外部数据。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from maldi_ms import alignment, filtering, modeling  # noqa: E402
from maldi_ms.io import load_config, read_directory  # noqa: E402
from maldi_ms.peak_detection import pick_peaks  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "uti8_50.yaml")

DEMO_SAMPLES = ["UTI11", "UTI13", "UTI20", "UTI21", "UTI22", "UTI23"]
DEMO_INFECTED = {"UTI11", "UTI13"}  # 与 configs/uti8_50.yaml 的 infection_uti 一致


def make_demo_dataset(data_dir: str, seed: int = 7) -> None:
    """生成 6 例合成谱图并写成 xlsx，感染组额外带一个差异峰。"""
    rng = np.random.default_rng(seed)
    mz = np.arange(100.0, 200.0, 0.02)
    shared = [(120.0, 4000.0), (150.0, 3000.0), (180.0, 2000.0)]

    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame({"横坐标(m/z)": mz})
    for i, name in enumerate(DEMO_SAMPLES):
        peaks = list(shared)
        if name in DEMO_INFECTED:
            peaks.append((160.0, 2500.0))
        y = rng.normal(0.0, 3.0, mz.size)
        for center, amplitude in peaks:
            y = y + amplitude * np.exp(-0.5 * ((mz - center) / 0.06) ** 2)
        df[f"样品{i + 1}_{name}"] = y

    df.to_excel(os.path.join(data_dir, "synthetic_demo.xlsx"), index=False)


def main(data_dir: str) -> None:
    cfg = load_config(CONFIG)

    # 1. 读取
    dataset = read_directory(data_dir, cfg)
    names = sorted(dataset.samples)
    print(dataset.summary())

    # 2. 峰检测
    peaks_by_sample = {
        name: pick_peaks(dataset.samples[name].mz, dataset.samples[name].intensity, cfg)[0]
        for name in names
    }
    print("检出峰:", sum(len(p) for p in peaks_by_sample.values()))

    # 3. 对齐
    flat = [
        {"sample": name, "mz": p["mz"], "intensity": p["intensity"]}
        for name in names for p in peaks_by_sample[name]
    ]
    clusters = alignment.align_peaks(flat, cfg["alignment"]["ppm_tolerance"])
    frame = alignment.build_feature_table(clusters, names)
    print("对齐后特征:", frame.shape[1])

    # 4. 过滤
    frame = filtering.filter_background(frame, [], cfg, verbose=False)
    frame = filtering.filter_by_detection(frame, cfg, verbose=False)
    print("过滤后特征:", frame.shape[1])

    # 5. 建模
    y = np.array([1 if dataset.samples[n].group == "感染组" else 0 for n in frame.index])
    X = modeling.transform(frame)
    diff = modeling.differential_test(frame, y, alignment.feature_mz(frame))
    panel = modeling.evaluate_panels(X, y, cfg)
    print(f"最优 panel: Top {panel.best_size}, AUC={panel.best_auc:.4f}")
    print(diff.head(5)[["Feature", "m/z", "log2(FC)", "p_value", "FDR"]].to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        demo_dir = os.path.join(tempfile.mkdtemp(prefix="maldi_demo_"), "data")
        print("未提供数据目录，演示模式：生成 6 例合成谱图（感染 2 / 健康 4）\n")
        make_demo_dataset(demo_dir)
        main(demo_dir)
