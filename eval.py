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

from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep

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
        motion_dim = self.cfg.amp_observation_space * self.cfg.num_amp_observations
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.discriminator = Discriminator(motion_dim).to(self.device)

        discriminator_weight, actor_weights, _ = torch.load("weight.pth")
        self.actor.load_state_dict(actor_weights)
        self.discriminator.load_state_dict(discriminator_weight)
        self.actor.eval()
        self.discriminator.eval()

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action
    
    @torch.no_grad()
    def get_discriminator_reward(self, motion_obs_batch: torch.Tensor) -> torch.Tensor:
        disc_step:ValueStep = self.discriminator(motion_obs_batch)
        print(disc_step.value)
        print(torch.sigmoid(disc_step.value))
        rewards = -torch.log(1 - 1 / (1 + torch.exp(-disc_step.value)) + 1e-5)
        return rewards
    
    def rollout(self, obs, info):

        for i in range(1000):
            obs = process_obs(obs)
            action = self.get_action(obs, True)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            motion_obs = info["amp_obs"]
            print("Fake")
            disc_reward = self.get_discriminator_reward(motion_obs)
            print(disc_reward)
            print("True")
            true_motion_obs = self.env.unwrapped.collect_reference_motions(16)
            disc_reward = self.get_discriminator_reward(true_motion_obs)
            print(disc_reward)
            print("---------------")
            obs = next_obs

        return obs, info

    def eval(self):
        obs, info = self.env.reset()
        obs, info = self.rollout(obs, info)

        self.env.close()

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.eval()
    simulation_app.close()