# -*- coding: utf-8 -*-
"""Baseline EP (next-lap): 휴리스틱으로 1랩 전개한 뒤의 EP(s_next)."""
from __future__ import annotations
from typing import Any, Callable
class BaselineOneLap:
    def __init__(self, env_factory, ep_from_state):
        self.env_factory = env_factory
        self.ep_from_state = ep_from_state

    def __call__(self, state):
        try:
            env = self.env_factory()
            if hasattr(env, "set_state"):
                env.set_state(state)
            elif hasattr(env, "load_state"):
                env.load_state(state)
            else:
                # 상태 고정이 불가하면 baseline = EP(state)로 폴백 (no-op baseline)
                return self.ep_from_state(state)

            if hasattr(env, "step_heuristic_one_lap"):
                s_next = env.step_heuristic_one_lap()
            else:
                if not hasattr(env, "heuristic_policy") or not hasattr(env, "step"):
                    # 휴리스틱이 없으면 역시 폴백
                    return self.ep_from_state(state)
                a = env.heuristic_policy(state)
                ts1 = env.step(a)
                # 관측을 상태로 사용
                if isinstance(ts1, tuple):  # (obs, reward, done, info) 스타일 방지
                    s_next = ts1[0]
                else:
                    s_next = ts1.observation if hasattr(ts1, "observation") else None
            return self.ep_from_state(s_next) if s_next is not None else self.ep_from_state(state)
        except Exception:
            # 어떤 이유로든 실패하면 안전 폴백
            return self.ep_from_state(state)
