import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tqdm import trange
import gymnasium
import torch
import numpy as np
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env_cfg import HumanoidAmpWalkEnvCfg

from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.alg.ppo import PPO
from RLAlg.alg.discriminator_alg import DiscriminatorAlg

from model import Actor, Critic, Discriminator

def process_obs(obs):
    features = obs["policy"]
    return features

class Evaluator:
    def __init__(self):
        self.cfg = HumanoidAmpWalkEnvCfg()
        self.cfg.scene.num_envs = 1
        self.env = gymnasium.make("Isaac-Humanoid-AMP-Walk-Direct-v0", cfg=self.cfg)

        obs_dim = self.cfg.observation_space
        reference_motion_dim = self.cfg.amp_observation_space * self.cfg.num_amp_observations
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        _, actor_weights, _ = torch.load("model.pth")
        self.actor.load_state_dict(actor_weights)
        self.actor.eval()
        self.steps = 25
        self.obs = None

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        

        return action
    
    def rollout(self):

        obs = self.obs
        for i in range(self.steps):
            obs = process_obs(obs)
            action = self.get_action(obs, True)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)

            obs = next_obs

        self.obs = obs

    def eval(self):
        obs, info = self.env.reset()
        self.obs = obs
        for epoch in trange(20):
            self.rollout()

        self.env.close()

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.eval()
    simulation_app.close()