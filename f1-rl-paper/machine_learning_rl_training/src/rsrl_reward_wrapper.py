# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Optional, Callable, Dict
from collections import deque
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

# 우리가 만든 모듈들
from machine_learning_rl_training.reward.step_reward import (
    RewardConfig, step_reward, terminal_reward
)
from machine_learning_rl_training.ep_next_dataset import build_feature_vector

class RSRLRewardWrapper(py_environment.PyEnvironment):
    """
    기존 RaceSimulation(PyEnv)을 감싸서:
      - step() 보상을 Step-wise RSRL 보상으로 교체
      - EP 로깅을 스텝/에피소드 단위로 수행
    """
    def __init__(
        self,
        base_env: py_environment.PyEnvironment,
        cfg: RewardConfig,
        ep_from_state: Callable[[Any], float],
        baseline_nextlap: Callable[[Any], float],
        ep_logger,                           # EPLogger (또는 None)
        meta: Dict[str, Any],                # {"track":..., "season":..., "round":...}
        ep_next_predictor=None,              # EpNextPredictor (또는 None)
        ep_next_logger=None,                 # EpNextDatasetWriter (또는 None)
    ):
        super().__init__()
        self._env = base_env
        self.cfg = cfg
        self.ep_from_state = ep_from_state
        self.baseline_nextlap = baseline_nextlap
        self.logger = ep_logger
        self.meta = meta or {}
        self.ep_next_predictor = ep_next_predictor
        self.ep_next_logger = ep_next_logger
        self._ep_next_history = deque(maxlen=getattr(ep_next_predictor, "seq_len", 0) or 0)
        self._global_step = 0
        self._last_pit_lap = None
        self._last_lap_time_prev = None
        self._prev_state = None   # step 전 s_t 캐시
        self._last_obs = None        # <<< 추가: 최근 관측을 상태로 쓸 때 필요
        # --- episode stats (1단계) ---
        self._ep_len = 0
        self._ep_pit_cnt = 0
        self._ep_lap_time_sum = 0.0
        self._ep_lap_time_n = 0
        self._ep_final_pos = None
        self._last_episode_summary = {}

    # ------- spec 위임 -------
    def observation_spec(self): return self._env.observation_spec()
    def action_spec(self): return self._env.action_spec()
    def batched(self):
        b = getattr(self._env, "batched", None)
        if callable(b):
            return b()
        return bool(b) if b is not None else False

        
    @property
    def batch_size(self):
        bs = getattr(self._env, "batch_size", None)
        if callable(bs):
            return bs()
        return bs

    def time_step_spec(self):
        # TF-Agents가 time_step_spec()을 직접 부를 때 래퍼가 투명해지도록
        return self._env.time_step_spec()

    def __getattr__(self, name):
        # 래퍼에 없는 속성/메서드는 원 환경으로 위임
        # 예: cat_preprocessor, final_position, 기타 커스텀 필드 등
        return getattr(self._env, name)


    # ------- state helpers (환경마다 메서드명이 조금 다를 수 있음) -------
    def _get_state(self):
        # 1) 환경이 get_state를 제공하면 사용
        if hasattr(self._env, "get_state"):
            try:
                return self._env.get_state()
            except NotImplementedError:
                pass
        # 2) 아니면 최근 관측을 상태로 사용
        return self._last_obs

    def _final_points(self) -> float:
        for name in ("compute_final_points", "get_final_points"):
            if hasattr(self._env, name):
                return float(getattr(self._env, name)())
        # 없으면 포지션으로 변환
        if hasattr(self._env, "final_position"):
            pos = int(self._env.final_position())
            return float(terminal_reward(pos))
        return 0.0

    def _info_from_env(self) -> Dict[str, Any]:
        """환경의 속성들로 info 딕셔너리를 구성 (필드명이 다르면 여기만 맞추면 됨)."""
        e = self._env
        info = {
            "track": getattr(e, "track_name", self.meta.get("track")),
            "lap_idx": getattr(e, "current_lap", None),
            "stint_lap": getattr(e, "stint_lap", None),
            "pos_now": getattr(e, "position_prev", getattr(e, "pos_before", None)),
            "pos_next": getattr(e, "position", None),
            "lap_time_next": getattr(e, "last_lap_time", None),
            "wear_delta": getattr(e, "last_tyre_wear_delta", 0.0),
            "pit_loss": getattr(e, "last_pit_time_loss", 0.0),
        }
        return info

    # ------- PyEnvironment 필수 구현 -------
    def _reset(self):
        ts0 = self._env.reset()
        self._last_obs = ts0.observation
        self._prev_state = self._last_obs
        self._ep_next_history.clear()
        self._last_pit_lap = None
        self._last_lap_time_prev = None
        # --- reset episode stats ---
        self._ep_len = 0
        self._ep_pit_cnt = 0
        self._ep_lap_time_sum = 0.0
        self._ep_lap_time_n = 0
        self._ep_final_pos = None
        self._last_episode_summary = {}

        zero = np.array(0.0, dtype=np.float32).reshape(())
        return ts.TimeStep(step_type=ts0.step_type, reward=zero,
                        discount=ts0.discount, observation=ts0.observation)

        
    def _step(self, action):
        # 0) step 전 상태 (s_t) 먼저 캐시
        s_t = self._prev_state

        # 1) action을 정수 스칼라로 강제 변환
        try:
            action_scalar = int(np.asarray(action).item())
        except Exception:
            action_scalar = int(action) if hasattr(action, "__int__") else action

        # >>>>>>>>>>>>>>>>>>>>>>>>>> 여기부터 추가 <<<<<<<<<<<<<<<<<<<<<<<<<<
        # avg retrun 이 왜 0.330 고정인지 해결 
        # 1) 에이전트 액션 스칼라화
        # 1) 에이전트 액션 스칼라화
        agent_action0 = action_scalar

        # 2) 환경의 합법 액션 범위 파악
        try:
            spec = self._env.action_spec()
            a_min = int(getattr(spec, "minimum", 0))   # 보통 0
            a_max = int(getattr(spec, "maximum", 0))   # 보통 len(available_compounds)
        except Exception:
            a_min, a_max = 0, len(getattr(self._env, "available_compounds", []))

        # 3) ✅ 매핑: 0은 no-pit 그대로, 그 외는 1..a_max로 클램핑
        if agent_action0 <= 0:
            env_action = 0
        else:
            env_action = max(1, min(int(agent_action0), int(a_max)))
        # <<<<<<<<<<<<<<<<<<<<<<<<<< 추가 끝 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # 2) 실제 환경 진행 (스칼라 action 전달)
        lap_time_prev = self._last_lap_time_prev
        ts1 = self._env.step(env_action)
        # --- update per-step episode stats ---
        self._ep_len += 1

        # pit 카운트: last_pit_time_loss > 0이면 pit했다고 간주
        try:
            pit_loss = float(getattr(self._env, "last_pit_time_loss", 0.0) or 0.0)
        except Exception:
            pit_loss = 0.0
        if pit_loss > 0.0:
            self._ep_pit_cnt += 1

        # lap time 누적: last_lap_time이 있을 때만
        lt = getattr(self._env, "last_lap_time", None)
        if lt is not None:
            try:
                lt = float(lt)
                if np.isfinite(lt) and lt > 0:
                    self._ep_lap_time_sum += lt
                    self._ep_lap_time_n += 1
            except Exception:
                pass
        self._last_lap_time_prev = lt if lt is not None else self._last_lap_time_prev

        # 3) 다음 상태: 관측을 상태로 사용 (s_{t+1})
        self._last_obs = ts1.observation
        s_tp1 = self._last_obs
        self._prev_state = s_tp1

        # 4) info 구성
        info = self._info_from_env()
        info["lap_time_prev"] = lap_time_prev
        info["ep_next_step"] = self._global_step
        if pit_loss > 0.0:
            self._last_pit_lap = info.get("lap_idx")

        # ep_next feature vector/history
        if self.ep_next_predictor is not None:
            feat_meta = {
                "lap_idx": info.get("lap_idx"),
                "pos_now": info.get("pos_now"),
                "lap_time_next": info.get("lap_time_next"),
                "lap_time_prev": info.get("lap_time_prev"),
                "pit_loss": info.get("pit_loss"),
                "wear_delta": info.get("wear_delta"),
                "tyre": getattr(s_t, "tyre", None),
                "fuel_kg": getattr(s_t, "fuel", None),
                "stint_lap": info.get("stint_lap"),
                "action_code": int(env_action),
                "pit_stop_flag": int((info.get("pit_loss", 0.0) or 0.0) > 0.0),
                "pit_count": int(self._ep_pit_cnt),
                "last_pit_lap": self._last_pit_lap,
            }
            self._ep_next_history.append(build_feature_vector(feat_meta))
            if len(self._ep_next_history) == self._ep_next_history.maxlen:
                try:
                    pred = self.ep_next_predictor.predict(list(self._ep_next_history))
                    y_lap = pred.get("y_lap")
                    info["ep_next_pred"] = -float(y_lap) if y_lap is not None else None
                except Exception:
                    info["ep_next_pred"] = None

        # 5) RSRL 보상 계산
        try:
            r_step = float(step_reward(
                s=s_t, a=action_scalar, s_next=s_tp1, info=info,
                cfg=self.cfg, ep_from_state=self.ep_from_state, baseline_nextlap=self.baseline_nextlap
            ))
        except Exception:
            r_step = 0.0

        # 6) EP 로깅 (s_t가 위에서 정의돼 있으므로 OK)
        if self.logger is not None and s_t is not None:
            try:
                # 액션 코드 스칼라 보장
                try:
                    acode = int(np.asarray(action_scalar).item())
                except Exception:
                    acode = int(action_scalar) if hasattr(action_scalar, "__int__") else str(action_scalar)

                self.logger.log_step({
                    "track": self.meta.get("track"),
                    "season": self.meta.get("season"),
                    "round": self.meta.get("round"),
                    "seed": self._seed_value(),
                    "lap_idx": info.get("lap_idx"),
                    "stint_lap": info.get("stint_lap"),
                    "tyre": getattr(s_t, "tyre", None),
                    "fuel_kg": getattr(s_t, "fuel", None),
                    "pos_now": info.get("pos_now"),
                    "action_code": int(env_action),
                    "agent_action0": int(agent_action0),  # 에이전트가 낸 값(0..N-1, 디버그용)
                    "pit_stop_flag": int((info.get("pit_loss", 0.0) or 0.0) > 0.0),
                    "compound_pick": getattr(action, "compound", None),
                })
            except Exception:
                pass

        # 6-1) ep_next supervised dataset logging
        if self.ep_next_logger is not None and s_t is not None:
            try:
                self.ep_next_logger.log_step({
                    "track": self.meta.get("track"),
                    "season": self.meta.get("season"),
                    "round": self.meta.get("round"),
                    "seed": self._seed_value(),
                    "lap_idx": info.get("lap_idx"),
                    "stint_lap": info.get("stint_lap"),
                    "tyre": getattr(s_t, "tyre", None),
                    "fuel_kg": getattr(s_t, "fuel", None),
                    "pos_now": info.get("pos_now"),
                    "pos_next": info.get("pos_next"),
                    "lap_time_next": info.get("lap_time_next"),
                    "lap_time_prev": info.get("lap_time_prev"),
                    "pit_loss": info.get("pit_loss"),
                    "wear_delta": info.get("wear_delta"),
                    "action_code": int(env_action),
                    "pit_stop_flag": int((info.get("pit_loss", 0.0) or 0.0) > 0.0),
                    "pit_count": int(self._ep_pit_cnt),
                    "last_pit_lap": self._last_pit_lap,
                })
            except Exception:
                pass

        # 7) 우리 보상으로 TimeStep 교체
        rew = np.array(r_step, dtype=np.float32)
        if getattr(ts1.reward, "shape", ()) != ():
            rew = np.broadcast_to(rew, ts1.reward.shape).astype(np.float32)

        new_ts = ts.TimeStep(
            step_type=ts1.step_type,
            reward=rew,
            discount=ts1.discount,
            observation=ts1.observation,
        )

        # --- episode end summary (1단계 지표 수집) ---
        if ts1.is_last():
            # final position
            if hasattr(self._env, "final_position"):
                try:
                    self._ep_final_pos = int(
                        self._env.final_position()
                        if callable(self._env.final_position)
                        else self._env.final_position
                    )
                except Exception:
                    self._ep_final_pos = None

            avg_lap_time = (
                self._ep_lap_time_sum / self._ep_lap_time_n
                if self._ep_lap_time_n > 0 else None
            )

            self._last_episode_summary = {
                "track": self.meta.get("track"),
                "season": self.meta.get("season"),
                "round": self.meta.get("round"),
                "seed": self._seed_value(),
                "episode_length": int(self._ep_len),
                "pit_count": int(self._ep_pit_cnt),
                "avg_lap_time": float(avg_lap_time) if avg_lap_time is not None else None,
                "final_position": (
                    int(self._ep_final_pos) if self._ep_final_pos is not None else None
                ),
            }

        # 8) 에피소드 종료 시 final_points 저장
        if ts1.is_last() and self.logger is not None:
            try:
                fp = self._final_points()
                self.logger.finalize(final_points=fp, seed=self._seed_value())
            except Exception:
                pass

        if ts1.is_last() and self.ep_next_logger is not None:
            try:
                self.ep_next_logger.finalize(seed=self._seed_value())
            except Exception:
                pass

        # global step for warmup
        self._global_step += 1

        return new_ts

    def get_episode_summary(self) -> Dict[str, Any]:
        base = self._get_env_episode_summary()
        out = dict(base)

        # wrapper에서 수집한 값이 있으면 우선 반영 (없으면 base 유지)
        if self._last_episode_summary:
            out.update({k: v for k, v in self._last_episode_summary.items() if v is not None})

        # meta로 track/season/round 강제 채우기 (빈칸 방지)
        for k in ("track", "season", "round"):
            mv = self.meta.get(k)
            if mv is not None:
                out[k] = mv

        # seed는 무조건 숫자/None만
        out["seed"] = self._seed_value()

        return out


    def _seed_value(self):
        v = getattr(self._env, "seed_value", None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                return None
        return None

    def _get_env_episode_summary(self) -> Dict[str, Any]:
        if hasattr(self._env, "get_episode_summary"):
            try:
                s = self._env.get_episode_summary()
                return dict(s) if s is not None else {}
            except Exception:
                return {}
        return {}
