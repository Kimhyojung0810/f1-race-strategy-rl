"""
GRU-based next-lap outcome predictor for ep_next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import os


def build_gru_model(seq_len: int, feature_dim: int, hidden_size: int = 64) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), name="seq")
    # reset_after=False disables cuDNN path to avoid GPU cudnn errors
    x = tf.keras.layers.GRU(hidden_size, reset_after=False, name="gru")(inputs)
    x = tf.keras.layers.Dense(hidden_size, activation="relu")(x)
    y_pos = tf.keras.layers.Dense(1, name="y_pos")(x)
    y_lap = tf.keras.layers.Dense(1, name="y_lap")(x)
    return tf.keras.Model(inputs=inputs, outputs={"y_pos": y_pos, "y_lap": y_lap})


@dataclass
class EpNextPredictor:
    model_dir: Path

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model = tf.keras.models.load_model(self.model_dir / "model")
        with open(self.model_dir / "feature_stats.json", "r") as f:
            stats = json.load(f)
        self.feature_mean = np.array(stats["mean"], dtype=np.float32)
        self.feature_std = np.array(stats["std"], dtype=np.float32)
        self.seq_len = int(stats["seq_len"])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / np.maximum(self.feature_std, 1e-6)

    def predict(self, seq: np.ndarray) -> Dict[str, float]:
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape != (self.seq_len, self.feature_mean.shape[0]):
            raise ValueError(f"Expected seq shape {(self.seq_len, self.feature_mean.shape[0])}, got {seq.shape}")
        x = self.normalize(seq)[None, ...]
        if os.environ.get("EP_NEXT_CPU", "0") == "1":
            with tf.device("/CPU:0"):
                out = self.model.predict(x, verbose=0)
        else:
            out = self.model.predict(x, verbose=0)
        return {
            "y_pos": float(np.squeeze(out["y_pos"])),
            "y_lap": float(np.squeeze(out["y_lap"])),
        }
