"""快速上手：读谱图、检峰、对齐、过滤、建模，五步跑完。

    PYTHONPATH=src python examples/quickstart.py <数据目录>

数据格式见 README「输入数据格式」。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from maldi_ms import alignment, filtering, modeling  # noqa: E402
from maldi_ms.io import load_config, read_directory  # noqa: E402
from maldi_ms.peak_detection import pick_peaks  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "uti8_50.yaml")


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
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
