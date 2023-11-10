# For env register
import gym_duckietown
import gymnasium

from typing import List

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.resize_observation import ResizeObservation

from gym_duckietown.wrappers import DiscreteWrapper, CropObservation, SegmentMiddleLaneWrapper, SegmentRemoveExtraInfo, SegmentLaneWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import numpy as np

from custom_net import CustomCNN

MODEL_PREFIX = "dqn"
SEED = 123123123

ckpt_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix=f"{MODEL_PREFIX}_ckpt",
    save_replay_buffer=True,
    save_vecnormalize=True,

)

def wrap(env):
    env = DiscreteWrapper(env)
    env = segment(env)
    return env

def segment(env):
    env = CropObservation(env, 140)
    env = SegmentLaneWrapper(env)
    env = SegmentMiddleLaneWrapper(env)
    env = SegmentRemoveExtraInfo(env)
    env = ResizeObservation(env, 120)
    return env

print(f"Running train on {id}")
env = make_vec_env("Duckietown-udem1-v0", n_envs=4, wrapper_class=wrap, seed=SEED)
env = VecFrameStack(env, 5)
print(env.observation_space.shape)
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=3),
)

model = DQN(
    policy="CnnPolicy",
    batch_size=32,
    env=env,
    gamma=0.99,
    learning_rate=0.00005,
    buffer_size=50000,
    tensorboard_log="./runs",
    learning_starts=10000,
    policy_kwargs=policy_kwargs
    
)

print(model.policy)

