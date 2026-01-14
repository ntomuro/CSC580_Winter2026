"""
dqn-core.py

Collection of core functions for Atari Pong.
"""
#---------------------------------------------------
# 1. Wrappers -- additional features for environment
#---------------------------------------------------
import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import ale_py

class AtariPreprocess(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, 84, 84), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info

class PongActionReducer(gym.ActionWrapper):
    """
    Reduce the action space from six to three (NOOP, FIRE, UP, DOWN)
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self._action_map = [0, 1, 2, 3]  # NOOP, FIRE, UP, DOWN

    def action(self, act):
        return self._action_map[act]
		
#----------------------------------
# 2. Reply Buffer
#----------------------------------
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """TODO (students):
        Store transition in buffer (circular).
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """TODO (students):
        Sample a batch and return arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
			
#--------------------------------------------------------------
# 3. DQN -- same deep learning network used in the Nature paper
#--------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        """TODO (students):
        Normalize input and compute Q-values.
        """
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#--------------------------------------------------
# 4. make_env function -- incorporates the Wrappers
#--------------------------------------------------
def make_env(render_mode=None):
    env = gym.make("PongNoFrameskip-v4", render_mode=render_mode)
    env = PongActionReducer(env)
    env = AtariPreprocess(env)
    env = FrameStack(env, 4)
    return env
	
