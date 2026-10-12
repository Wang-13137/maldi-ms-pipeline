"""原始谱图读取与配置加载。

数据格式：每个 Excel 文件第一列是 m/z 轴，其余每列是一条谱图。
列名形如 ``样品2_UTI11``、``样品10__CN``、``样品19_阴性对照``，
样本身份只能从列名里解析，所以要在这里完成分类。
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class Spectrum:
    """一条谱图。"""

    name: str
    mz: np.ndarray
    intensity: np.ndarray
    group: str = "未分组"
    source_file: str = ""

    @property
    def n_points(self) -> int:
        return len(self.mz)


@dataclass
class Dataset:
    """一次读取的结果：样本谱图 + 阴性对照谱图。"""

    samples: Dict[str, Spectrum] = field(default_factory=dict)
    negatives: Dict[str, Spectrum] = field(default_factory=dict)

    def summary(self) -> str:
        infected = sum(1 for s in self.samples.values() if s.group == "感染组")
        healthy = sum(1 for s in self.samples.values() if s.group == "健康组")
        pts = sum(s.n_points for s in self.samples.values())
        return (
            f"样本 {len(self.samples)} 例（感染 {infected} / 健康 {healthy}），"
            f"阴性对照 {len(self.negatives)} 例，"
            f"每例 {next(iter(self.samples.values())).n_points if self.samples else 0} 数据点，"
            f"合计约 {pts / 1e6:.1f} M 点"
        )


def load_config(path: str) -> dict:
    """读取 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _is_negative(col: str, cfg: dict) -> bool:
    ds = cfg.get("dataset", {})
    for kw in ds.get("negative_keywords", []):
        if kw in col:
            return True
    for suf in ds.get("negative_suffixes", []):
        if col.endswith(suf):
            return True
    return False


def read_directory(data_dir: str, cfg: dict) -> Dataset:
    """扫描目录下的 xlsx，拆成样本谱图与阴性对照谱图。"""
    ds_cfg = cfg.get("dataset", {})
    infection = set(ds_cfg.get("infection_uti", []))
    pattern = ds_cfg.get("sample_pattern", r"UTI(\d+)")
    regex = re.compile(pattern, re.IGNORECASE)

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".xlsx") and not f.startswith("~")
    )

    dataset = Dataset()
    skipped: List[str] = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        df = pd.read_excel(fpath)
        mz = df[df.columns[0]].values.astype(float)

        for col in df.columns[1:]:
            col_str = str(col)
            intensity = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)

            if _is_negative(col_str, cfg):
                neg_name = f"{fname[:10]}_{col_str[:20]}"
                dataset.negatives[neg_name] = Spectrum(
                    name=neg_name, mz=mz.copy(), intensity=intensity,
                    group="阴性对照", source_file=fname,
                )
                continue

            m = regex.search(col_str)
            if not m:
                skipped.append(f"{fname}: {col_str}")
                continue
            uti_num = int(m.group(1))
            sample_name = f"UTI{uti_num}"
            group = "感染组" if uti_num in infection else "健康组"
            dataset.samples[sample_name] = Spectrum(
                name=sample_name, mz=mz.copy(), intensity=intensity,
                group=group, source_file=fname,
            )

    if skipped:
        print(
            f"[警告] {len(skipped)} 个列名无法解析出样本编号，已跳过：\n  "
            + "\n  ".join(skipped[:10])
            + ("\n  ..." if len(skipped) > 10 else ""),
            file=sys.stderr,
        )

    return dataset


def labels_for(dataset: Dataset, sample_order: Optional[List[str]] = None) -> np.ndarray:
    """按给定（或字典序）样本顺序返回 0/1 标签，1 表示感染组。"""
    order = sample_order or sorted(dataset.samples.keys())
    return np.array(
        [1 if dataset.samples[name].group == "感染组" else 0 for name in order]
    )
