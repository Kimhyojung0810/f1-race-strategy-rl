# -*- coding: utf-8 -*-
"""Step-wise Reward (Next-lap feedback) + 간이 레퍼런스 + 터미널 포인트."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

# -------------------------
# Config
# -------------------------
@dataclass
class RewardConfig:
    gamma: float = 1.0
    # weights (λ)
    l_lap: float = 0.2
    l_pos: float = 0.2
    l_pit: float = 0.2
    l_risk: float = 0.0
    l_stint: float = 0.0
    # legacy (kept for backwards compat, effectively off)
    l_deg: float = 0.0
    # unit conversions
    k_t: float = 1 / 7.0   # sec → points
    k_pos: float = 1.0     # 1 pos → 1 pt
    k_pit: float = 1 / 7.0
    k_stint: float = 1.0
    # legacy (kept for backwards compat, effectively off)
    k_deg: float = 0.0
    # short-stint config
    min_stint_laps: int = 2
    # ΔEP_next helper (kept for compatibility; actual predictor may live elsewhere)
    ep_next_clip: float = 5.0
    ep_next_warmup_steps: int = 5000
    # switches
    use_deltaEP_next: bool = True
    use_potential: bool = False  # 처음엔 OFF, 안정화 필요시 ON

# -------------------------
# 간이 레퍼런스 (FastF1 테이블로 교체 예정)
# -------------------------
def ref_laptime(state) -> float:
    tyre = getattr(state, "tyre", "MED")
    fuel = float(getattr(state, "fuel", 30.0))
    base = 90.0  # 1:30.000
    tyre_delta = {"SOFT": -0.6, "MED": 0.0, "HARD": +0.5}.get(tyre, 0.0)
    return base + tyre_delta + 0.03 * fuel  # 1kg = 0.03s

def pit_loss_ref(track: str) -> float:
    table = {"Monza": 18.5, "Imola": 22.0, "Silverstone": 20.5}
    return table.get(track or "", 20.0)

def ref_wear_delta(state) -> float:
    tyre = getattr(state, "tyre", "MED")
    return {"SOFT": 0.9, "MED": 0.6, "HARD": 0.4}.get(tyre, 0.6)

# -------------------------
# 핵심: Step-wise Reward
# -------------------------
def step_reward(
    s: Any, a: Any, s_next: Any, info: Dict,
    cfg: RewardConfig,
    ep_from_state,         # callable(state)->float
    baseline_nextlap,      # callable(state)->float
) -> float:
    # 1) ΔEP_next (RSRL 핵심)
    r_dep = 0.0
    if cfg.use_deltaEP_next:
        ep_next_pred = info.get("ep_next_pred", None)
        if ep_next_pred is not None:
            # Use GRU-based predictor output with clip + warmup
            try:
                r_dep_raw = float(ep_next_pred)
            except Exception:
                r_dep_raw = 0.0
            c = abs(float(cfg.ep_next_clip))
            if c > 0.0:
                r_dep = max(-c, min(c, r_dep_raw))
            else:
                r_dep = r_dep_raw

            step_idx = 0
            try:
                step_idx = int(info.get("ep_next_step", 0) or 0)
            except Exception:
                step_idx = 0
            if step_idx < cfg.ep_next_warmup_steps:
                scale = float(step_idx) / float(max(cfg.ep_next_warmup_steps, 1))
                r_dep *= scale
        else:
            # Fallback: 기존 평균 기반 ΔEP_next
            b_t  = baseline_nextlap(s)   # 휴리스틱 1랩 후 EP
            epn  = ep_from_state(s_next) # 현재 행동 결과 상태의 남은 EP
            r_dep = epn - b_t

    # 2) Lap time 즉시 피드백
    t_next = info.get("lap_time_next", None)
    r_lap = 0.0
    if t_next is not None:
        t_ref = ref_laptime(s_next)
        r_lap = - cfg.k_t * (t_next - t_ref)

    # 3) Position 변화  (개선: pos_now - pos_next)
    r_pos = 0.0
    pos_now, pos_next = info.get("pos_now"), info.get("pos_next")
    if pos_now is not None and pos_next is not None:
        dpos = int(pos_now) - int(pos_next)   # 앞서면 +1
        r_pos = cfg.k_pos * dpos

    # 4) Pit loss 초과분
    pit_excess = max(0.0, (info.get("pit_loss") or 0.0) - pit_loss_ref(info.get("track", "")))
    r_pit = - cfg.k_pit * pit_excess if pit_excess > 0.0 else 0.0
    # 계속 선두이면 어떻게 할 것인지? 기본 포인트 어떻게?
    
    # 5) Short-stint penalty (only if a pit actually happened)
    r_stint = 0.0
    stint_lap = info.get("stint_lap")
    pit_loss = info.get("pit_loss") or 0.0
    if pit_loss > 0.0 and stint_lap is not None:
        shortfall = max(0, int(cfg.min_stint_laps) - int(stint_lap))
        if shortfall > 0:
            r_stint = - cfg.k_stint * float(shortfall)

    # 6) Wear 초과분  비활성화 (deg terms kept but off)
    r_deg = 0.0
    
   # 7) Potential shaping (옵션 꺼두기)
    phi = 0.0
    if getattr(cfg, "use_potential", False):
        phi = cfg.gamma * ep_from_state(s_next) - ep_from_state(s)

    return (
        r_dep
        + cfg.l_lap * r_lap
        + cfg.l_pos * r_pos
        + cfg.l_pit * r_pit
        + cfg.l_stint * r_stint
        + cfg.l_deg * r_deg  # <- 0 곱해져서 영향 없음
        + phi                # <- use_potential=False면 0
    )
# -------------------------
# 터미널 보상 (F1 points)
# -------------------------
def terminal_reward(final_position: int) -> float:
    pts = [25,18,15,12,10,8,6,4,2,1]
    return pts[final_position-1] if 1 <= final_position <= 10 else 0.0
