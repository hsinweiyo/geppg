import gym
import numpy as np

from unityagents import UnityEnvironment

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

import os

class UnityEnv(gym.Env):
    def __init__(self, app_name=None, idx=0):
        # parameter
        app_path = os.path.join(os.path.dirname(__file__), 'assets', app_name)
        idx = idx
        no_graphics = False
        #no_graphics = True
        self.num_envs = 1

        # create environment
        self._env = UnityEnvironment(file_name=app_path, worker_id=idx, no_graphics=no_graphics)
        self.name = app_name

        # Only Accept Environment with Only One Brain
        assert len(self._env.brains) == 1
        self.brain_name = self._env.external_brain_names[0]
        self.brain = self._env.brains[self.brain_name]
        
        # viusalization
        self.use_visual = (self.brain.number_visual_observations == 1)

        # action space dimension
        if self.brain.vector_action_space_type == "discrete":
            self._a_dim = Discrete(1)
        else:
            high = np.array([np.inf] * (self.brain.vector_action_space_size))
            self._a_dim = Box(-high, high)
        
        # observation spce dimension
        if self.use_visual and False and no_graphic:
            high = np.array([np.inf] * self.brain.camera_resolutions[0]["height"] * 
                self.brain.camera_resolutions[0]["width"] * 3)
            self._ob_dim = Box(-high, high)
        else:
            high = np.array([np.inf] * self.brain.vector_observation_space_size)
            self._ob_dim = Box(-high, high)

        # video buffer
        self.frames = []

    def reset(self, config=None):
        self.frames = []
        info = self._env.reset(config=config)[self.brain_name] 
        state = info.vector_observations[0]
        return np.array([state])

    def step(self, action):
        info = self._env.step([action])[self.brain_name]
        
        state  = info.vector_observations[0] 
        reward = info.rewards[0]
        done = info.local_done[0] 

        self._collect_frames(info.visual_observations[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([None])

    def close(self):
        self._env.close()

    def _collect_frames(self, frame):
        if self.use_visual:
            self.frames.append(frame)

    @property
    def action_space(self):
        return self._a_dim

    @property
    def observation_space(self):
        return self._ob_dim

