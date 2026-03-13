# -*- coding: utf-8 -*-
"""EP Logger: (state, action) → final_points 로깅 (샤딩 저장)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

REQUIRED_COLS = [
    "track","season","round","seed",
    "lap_idx","stint_lap","tyre","fuel_kg","pos_now",
    "action_code","pit_stop_flag","compound_pick",
    # filled at episode end:
    "final_points",
]

class EPLogger:
    """에피소드 동안 스텝 레코드를 모았다가, 종료 시 final_points 채워 파일로 저장."""
    def __init__(self, outdir: str | Path, shard_name: Optional[str]=None):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.shard_name = shard_name  # e.g., "seed_003"
        self.rows: List[Dict[str, Any]] = []
        self._episode_idx = 0

    def log_step(self, meta: Dict[str, Any]) -> None:
        """에피소드 중 매 스텝 호출. final_points는 finalize에서 채움."""
        row = {k: meta.get(k, None) for k in REQUIRED_COLS if k != "final_points"}
        row["final_points"] = None
        self.rows.append(row)

    def finalize(self, final_points: float, seed: int) -> Path:
        """에피소드 종료 시 모든 row에 final_points 채우고 저장."""
        if not self.rows:
            return self.outdir / "empty.parquet"

        for r in self.rows:
            r["final_points"] = final_points

        df = pd.DataFrame(self.rows)
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = None

        if self.shard_name:
            fname = f"{self.shard_name}_ep{self._episode_idx:05d}"
        else:
            fname = f"seed_{seed:03d}_ep{self._episode_idx:05d}"
        path_parquet = self.outdir / f"{fname}.parquet"
        try:
            df.to_parquet(path_parquet, index=False)
            saved = path_parquet
        except Exception:
            # pyarrow/fastparquet 미설치 환경 대비
            path_csv = self.outdir / f"{fname}.csv"
            df.to_csv(path_csv, index=False)
            saved = path_csv

        # 다음 에피소드 대비 초기화
        self.rows.clear()
        self._episode_idx += 1
        return saved
