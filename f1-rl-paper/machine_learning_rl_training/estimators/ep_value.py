# -*- coding: utf-8 -*-
"""EP 추정기 (초기 버전): 로그 평균 기반. 로깅이 쌓이면 회귀/롤아웃으로 교체."""
from __future__ import annotations
import pandas as pd
from typing import Any, Optional

class EPFromStateMean:
    """
    간단한 시작 버전:
      - 트랙/타이어 기준으로 final_points 평균을 EP(s)로 사용
      - 데이터가 적을 땐 전체 평균으로 fallback
    """
    def __init__(self, df_logs: Optional[pd.DataFrame] = None, default_pts: float = 6.0):
        self.df = df_logs if df_logs is not None else pd.DataFrame()
        self.default = float(default_pts)

    def __call__(self, state: Any) -> float:
        if self.df is None or self.df.empty:
            return self.default

        df = self.df
        track = getattr(state, "track", None)
        tyre  = getattr(state, "tyre",  None)

        if track is not None and "track" in df.columns:
            df = df[df["track"] == track]
        if tyre is not None and "tyre" in df.columns:
            df = df[df["tyre"] == tyre]

        if "final_points" not in df.columns or len(df) == 0:
            return self.default

        # 표본이 너무 적으면 전체 평균으로
        if len(df) < 20:
            return float(self.df["final_points"].mean())

        return float(df["final_points"].mean())
