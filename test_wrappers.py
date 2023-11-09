# For env register
import gym_duckietown
import gymnasium

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers.resize_observation import ResizeObservation

from gymnasium.wrappers.frame_stack import FrameStack
from gym_duckietown.wrappers import DiscreteWrapper, CropObservation, SegmentMiddleLaneWrapper, SegmentRemoveExtraInfo, SegmentLaneWrapper

from PIL import Image

def save_obs(env, fname: str):
    obs, *_ = env.reset(seed=1337)
    img = Image.fromarray(obs)
    img.save(f"{fname}.jpg")

env = gymnasium.make("Duckietown-udem1-v0")
save_obs(env, "original")


env = CropObservation(env, 150)
env = SegmentLaneWrapper(env)
env = SegmentMiddleLaneWrapper(env)
env = SegmentRemoveExtraInfo(env)
env = ResizeObservation(env, 120)

save_obs(env, "processed")