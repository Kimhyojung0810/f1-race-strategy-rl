import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class PadObservationWrapper(py_environment.PyEnvironment):
    def __init__(self, env, target_dim: int):
        self._env = env
        self._target_dim = int(target_dim)

        orig = env.observation_spec()
        self._obs_spec = array_spec.BoundedArraySpec(
            shape=(self._target_dim,),
            dtype=orig.dtype,
            minimum=getattr(orig, "minimum", -np.inf),
            maximum=getattr(orig, "maximum", np.inf),
            name="observation",
        )
        
    def get_episode_summary(self):
        if hasattr(self._env, "get_episode_summary"):
            return self._env.get_episode_summary()
        return {}
        
    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def time_step_spec(self):
        base = self._env.time_step_spec()
        return ts.TimeStep(
            step_type=base.step_type,
            reward=base.reward,
            discount=base.discount,
            observation=self._obs_spec,
        )

    def _pad(self, obs):
        obs = np.asarray(obs, dtype=self._obs_spec.dtype)
        if obs.shape[0] == self._target_dim:
            return obs
        out = np.zeros((self._target_dim,), dtype=self._obs_spec.dtype)
        out[: min(len(obs), self._target_dim)] = obs[: self._target_dim]
        return out

    def _reset(self):
        t = self._env.reset()
        return t._replace(observation=self._pad(t.observation))

    def _step(self, action):
        t = self._env.step(action)
        return t._replace(observation=self._pad(t.observation))

    def __getattr__(self, name):
        # Pad wrapper에 없는 속성/메서드는 내부 env로 위임
        return getattr(self._env, name)
