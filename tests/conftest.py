"""测试公用夹具。"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maldi_ms.io import load_config  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "uti8_50.yaml")


@pytest.fixture(scope="session")
def cfg():
    return load_config(os.path.abspath(CONFIG_PATH))


def gaussian_spectrum(mz, peaks, sigma=0.02, noise_sigma=3.0, baseline=50.0, seed=0):
    """在给定 m/z 网格上合成高斯峰 + 高斯噪声。"""
    rng = np.random.default_rng(seed)
    y = np.full(len(mz), baseline, dtype=float) + rng.normal(0, noise_sigma, len(mz))
    for center, amplitude in peaks:
        y += amplitude * np.exp(-0.5 * ((mz - center) / sigma) ** 2)
    return np.maximum(y, 0.0)


def synthetic_grid(start=100.0, stop=200.0, step=0.02):
    return np.arange(start, stop + step / 2, step)
