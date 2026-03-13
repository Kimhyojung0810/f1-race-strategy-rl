"""
Utilities for ep_next dataset logging and feature building.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
TYRE_MAP = {"SOFT": 0.0, "MED": 1.0, "HARD": 2.0, "I": 3.0, "W": 4.0}

FEATURE_NAMES = [
    "lap_idx",
    "pos_now",
    "lap_time_next",
    "lap_time_prev",
    "pit_loss",
    "wear_delta",
    "tyre_id",
    "fuel_kg",
    "stint_lap",
    "action_code",
    "pit_stop_flag",
    "pit_count",
    "last_pit_lap",
]


def points_for_position(pos: Optional[int]) -> float:
    if pos is None:
        return 0.0
    return float(POINTS_TABLE.get(int(pos), 0))


def build_feature_vector(meta: Dict[str, Any]) -> List[float]:
    def _num(v: Any) -> float:
        try:
            x = float(v)
            if x != x or x in (float("inf"), float("-inf")):
                return 0.0
            return x
        except Exception:
            return 0.0

    tyre = meta.get("tyre")
    tyre_id = TYRE_MAP.get(str(tyre).upper(), -1.0) if tyre is not None else -1.0
    out = [
        _num(meta.get("lap_idx")),
        _num(meta.get("pos_now")),
        _num(meta.get("lap_time_next")),
        _num(meta.get("lap_time_prev")),
        _num(meta.get("pit_loss")),
        _num(meta.get("wear_delta")),
        _num(tyre_id),
        _num(meta.get("fuel_kg")),
        _num(meta.get("stint_lap")),
        _num(meta.get("action_code")),
        _num(meta.get("pit_stop_flag")),
        _num(meta.get("pit_count")),
        _num(meta.get("last_pit_lap")),
    ]
    return out


@dataclass
class EpNextDatasetWriter:
    outdir: Path
    shard_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        self._episode_idx = 0

    def log_step(self, meta: Dict[str, Any]) -> None:
        pos_now = meta.get("pos_now")
        pos_next = meta.get("pos_next")
        y_pos = None
        y_lap = None
        if pos_now is not None and pos_next is not None:
            try:
                y_pos = float(int(pos_now) - int(pos_next))
            except Exception:
                y_pos = None

        lap_prev = meta.get("lap_time_prev")
        lap_next = meta.get("lap_time_next")
        try:
            if lap_prev is not None and lap_next is not None:
                y_lap = float(lap_next) - float(lap_prev)
        except Exception:
            y_lap = None

        row = {
            "track": meta.get("track"),
            "season": meta.get("season"),
            "round": meta.get("round"),
            "seed": meta.get("seed"),
            "lap_idx": meta.get("lap_idx"),
            "pos_now": pos_now,
            "pos_next": pos_next,
            "lap_time_next": meta.get("lap_time_next"),
            "lap_time_prev": meta.get("lap_time_prev"),
            "pit_loss": meta.get("pit_loss"),
            "wear_delta": meta.get("wear_delta"),
            "tyre": meta.get("tyre"),
            "fuel_kg": meta.get("fuel_kg"),
            "stint_lap": meta.get("stint_lap"),
            "action_code": meta.get("action_code"),
            "pit_stop_flag": meta.get("pit_stop_flag"),
            "pit_count": meta.get("pit_count"),
            "last_pit_lap": meta.get("last_pit_lap"),
            "y_pos": y_pos,
            "y_lap": y_lap,
        }
        self.rows.append(row)

    def finalize(self, seed: int) -> Path:
        if not self.rows:
            return self.outdir / "empty.csv"

        df = pd.DataFrame(self.rows)
        fname = self.shard_name or f"seed_{seed:03d}"
        fname = f"{fname}_ep{self._episode_idx:05d}"
        path = self.outdir / f"{fname}.csv"
        df.to_csv(path, index=False)
        self.rows.clear()
        self._episode_idx += 1
        return path
