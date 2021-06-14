import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.typing import types
from typing import Text


class TestEnvironment(tf_agents.environments.py_environment.PyEnvironment):
    def __init__(self, discount=1.0):
        super().__init__()
        self._action_spec = tf_agents.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, name="action", minimum=0, maximum=3)
        self._observation_spec = tf_agents.specs.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, name="observation", minimum=0, maximum=1)
        self.discount = discount

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _init_state(self):
        self._state = np.zeros(2, dtype=np.int32)
        obs = np.zeros((4, 4), dtype=np.int32)
        obs[self._state[0], self._state[1]] = 1
        return obs

    def _reset(self):
        obs = self._init_state()
        return tf_agents.trajectories.time_step.restart(obs)

    def _step(self, action):
        self._state += [(-1, 0), (+1, 0), (0, -1), (0, +1)][action]
        reward = -1
        obs = np.zeros((4, 4), dtype=np.int32)
        done = (self._state.min() < 0 or self._state.max() > 3)
        if not done:
            obs[self._state[0], self._state[1]] = 1
        if done or np.all(self._state == np.array([3, 3])):
            reward = -10 if done else +100
            obs = self._init_state()
            return tf_agents.trajectories.time_step.termination(obs, reward)
        else:
            return tf_agents.trajectories.time_step.transition(obs, reward,
                                                               self.discount)

    def seed(self, seed: types.Seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def render(self, mode: Text = 'rgb_array'):
        obs = np.zeros((4, 4), dtype=np.int32)
        obs[self._state[0], self._state[1]] += 1
        print("-" * 40)
        for i in range(4):
            for j in range(4):
                print(obs[i][j], end="\t")
            print()
        print("-" * 40)
