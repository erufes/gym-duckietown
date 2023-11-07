# For env register
import gym_duckietown

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import gymnasium as gym
from env import launch_env
from gym.wrappers.resize_observation import ResizeObservation
import numpy as np

MODEL_PREFIX = "td3"
SEED = 123

ckpt_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix=f"{MODEL_PREFIX}_ckpt",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# env = VecFrameStack(make_vec_env(launch_env, 1), 4)
env = gym.make("Duckietown-udem1-v0")

model = TD3(
    policy="CnnPolicy",
    env=env,
    batch_size=32,
    seed=SEED,
    tau=0.005,
    target_policy_noise=0.2,
    buffer_size=10_000,
    tensorboard_log="./runs",
    target_noise_clip=0.5,
    verbose=2,
    train_freq=32 * 32
)

model.learn(
    total_timesteps=1_000_000,
    callback=ckpt_callback,
    progress_bar=True,
    tb_log_name=f"{MODEL_PREFIX}",
)
model.save(f"{MODEL_PREFIX}_duck")
