"""命令行入口与流程编排。

用法::

    python -m maldi_ms.cli --config configs/uti8_50.yaml \
        --data "C:/Users/Jenny/Desktop/UTI/8-50" --out outputs/uti8_50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from . import alignment, filtering, modeling
from .io import Dataset, Spectrum, load_config, read_directory
from .peak_detection import pick_peaks


def run_pipeline(cfg: dict, data_dir: str, out_dir: str, verbose: bool = True) -> dict:
    """跑通完整流程，返回结果字典。"""
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 1. 读取 ----------------
    if verbose:
        print("=" * 60)
        print("Step 1: 读取原始谱图")
        print("=" * 60)
    dataset = read_directory(data_dir, cfg)
    sample_names = sorted(dataset.samples.keys())
    if verbose:
        print(f"  {dataset.summary()}")

    # ---------------- 2. 峰检测 ----------------
    if verbose:
        print("\n" + "=" * 60)
        print(f"Step 2: 峰检测 (SNR>={cfg['peak_detection']['snr_threshold']}, "
              f"dist>={cfg['peak_detection']['min_peak_distance']} Da)")
        print("=" * 60)

    peaks_by_sample: Dict[str, List[dict]] = {}
    peak_stats = []
    for name in sample_names:
        spec: Spectrum = dataset.samples[name]
        peaks, noise = pick_peaks(spec.mz, spec.intensity, cfg)
        peaks_by_sample[name] = peaks
        peak_stats.append({
            "sample": name, "group": spec.group,
            "n_peaks": len(peaks), "noise": noise,
        })

    negative_peaks: List[dict] = []
    for nname in sorted(dataset.negatives.keys()):
        spec = dataset.negatives[nname]
        peaks, _ = pick_peaks(spec.mz, spec.intensity, cfg)
        negative_peaks.extend(peaks)
        if verbose:
            print(f"  阴性对照 {nname[:30]}: {len(peaks)} peaks")

    total_detections = sum(len(p) for p in peaks_by_sample.values())
    if verbose:
        for g in ("感染组", "健康组"):
            counts = [s["n_peaks"] for s in peak_stats if s["group"] == g]
            if counts:
                print(f"  {g}: 平均峰数 {np.mean(counts):.0f} "
                      f"(范围 {min(counts)}-{max(counts)})")
        print(f"  样本总检出峰: {total_detections}")
        print(f"  阴性对照总检出峰: {len(negative_peaks)}")

    if total_detections == 0:
        raise RuntimeError("未检测到任何峰，请检查数据或降低 snr_threshold")

    # ---------------- 3. 对齐 ----------------
    if verbose:
        print("\n" + "=" * 60)
        print(f"Step 3: 特征对齐 ("
              f"{cfg['alignment']['ppm_tolerance']} ppm)")
        print("=" * 60)

    flat = []
    for name in sample_names:
        for p in peaks_by_sample[name]:
            flat.append({"sample": name, "mz": p["mz"], "intensity": p["intensity"]})

    clusters = alignment.align_peaks(flat, cfg["alignment"]["ppm_tolerance"])
    summary = alignment.cluster_summary(clusters)
    frame = alignment.build_feature_table(clusters, sample_names)
    mzs = alignment.feature_mz(frame)

    if verbose:
        counts = [f["n_samples"] for f in summary]
        print(f"  原始检出峰: {total_detections}")
        print(f"  对齐后特征: {len(clusters)}")
        print(f"  平均覆盖样本数: {np.mean(counts):.1f}/{len(sample_names)}")
        print(f"  仅在 1 个样本中出现: {sum(1 for c in counts if c == 1)}")

    # ---------------- 4. 背景过滤 ----------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 4: 阴性对照背景过滤")
        print("=" * 60)
    frame = filtering.filter_background(frame, negative_peaks, cfg, verbose=verbose)

    # ---------------- 5. 检出率过滤 ----------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 5: 检出率过滤")
        print("=" * 60)
    frame = filtering.filter_by_detection(frame, cfg, verbose=verbose)

    frame.to_excel(os.path.join(out_dir, "aligned_feature_table.xlsx"))

    # ---------------- 6. 建模 ----------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 6: 差异检验 + panel 评估")
        print("=" * 60)

    y = np.array([1 if dataset.samples[n].group == "感染组" else 0 for n in frame.index])
    X = modeling.transform(frame)
    kept_mzs = alignment.feature_mz(frame)

    diff = modeling.differential_test(frame, y, kept_mzs)
    diff.to_excel(os.path.join(out_dir, "feature_stats.xlsx"), index=False)

    d = cfg["modeling"]["diff"]
    sig_mask = (diff["p_value"] < d["p_value"]) & (diff["log2(FC)"].abs() > d["log2_fc"])
    fdr_mask = (diff["FDR"] < d["fdr_p_value"]) & (diff["log2(FC)"].abs() > d["fdr_log2_fc"])
    n_sig = int(sig_mask.sum())
    n_fdr = int(fdr_mask.sum())
    diff[sig_mask].to_excel(os.path.join(out_dir, "diff_features.xlsx"), index=False)

    panel = modeling.evaluate_panels(
        X, y, cfg, ttest_order=list(range(len(diff))), verbose=verbose
    )

    metrics = {
        "n_samples": len(sample_names),
        "n_infected": int(y.sum()),
        "n_healthy": int((y == 0).sum()),
        "n_negative_controls": len(dataset.negatives),
        "data_points_per_spectrum": int(dataset.samples[sample_names[0]].n_points),
        "total_detected_peaks": int(total_detections),
        "n_features_aligned": len(clusters),
        "n_features_final": int(frame.shape[1]),
        "compression_ratio": 1 - frame.shape[1] / max(len(clusters), 1),
        "n_diff_p05": n_sig,
        "n_diff_fdr05": n_fdr,
        "min_fdr": float(diff["FDR"].min()) if len(diff) else None,
        "best_panel_size": panel.best_size,
        "best_panel_auc": panel.best_auc,
        "panel_curve": {str(k): v for k, v in panel.curve.items()},
        "best_ttest_panel_auc": panel.best_ttest_auc,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("结果摘要")
        print("=" * 60)
        print(f"  样本数: {metrics['n_samples']} "
              f"(感染 {metrics['n_infected']} / 健康 {metrics['n_healthy']})")
        print(f"  检出峰: {metrics['total_detected_peaks']}")
        print(f"  对齐特征: {metrics['n_features_aligned']}")
        print(f"  最终特征: {metrics['n_features_final']}")
        print(f"  差异特征 p<0.05: {n_sig}，FDR<0.05: {n_fdr}")
        print(f"  Best RF panel: Top {panel.best_size}, AUC={panel.best_auc:.4f}")
        print(f"  输出目录: {out_dir}")

    return {
        "dataset": dataset,
        "feature_table": frame,
        "peaks_by_sample": peaks_by_sample,
        "negative_peaks": negative_peaks,
        "diff": diff,
        "panel": panel,
        "metrics": metrics,
        "X": X,
        "y": y,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="maldi_ms.cli",
        description="MALDI-TOF 代谢组学分析流水线",
    )
    p.add_argument("--config", required=True, help="YAML 配置文件路径")
    p.add_argument("--data", required=True, help="原始数据目录（含 xlsx）")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_pipeline(cfg, args.data, args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
