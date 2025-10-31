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
from RLAlg.alg.gan import GAN

from model import Actor, Critic, Discriminator

def process_obs(obs):
    features = obs["policy"]
    return features

class Trainer:
    def __init__(self):
        self.cfg = HumanoidAmpWalkEnvCfg()
        self.env_name = "Isaac-Humanoid-AMP-Walk-Direct-v0"
        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        obs_dim = self.cfg.observation_space
        motion_dim = self.cfg.amp_observation_space * self.cfg.num_amp_observations
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.discriminator = Discriminator(motion_dim).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(params, lr=3e-4)
        self.d_optimizer = torch.optim.Adam(
            [
                {'params': self.discriminator.encoder.parameters(), "weight_decay":1e-4},
                {'params': self.discriminator.head.parameters(), "weight_decay":1e-2},
            ],
            lr=1e-5, betas=(0.5, 0.999)
        )

        self.steps = 20

        self.rollout_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            self.steps
        )

        self.batch_keys = ["observations",
                           "actions",
                           "log_probs",
                           "rewards",
                           "values",
                           "returns",
                           "advantages"
                        ]

        self.rollout_buffer.create_storage_space("observations", (obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("motion_observations", (motion_dim,))
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("dones", (), torch.float32)

        self.reference_motion_buffer = ReplayBuffer(
            4000,
            50
        )
        

        self.reference_motion_buffer.create_storage_space("motion_observations", (motion_dim,))

        for _ in range(50):
            motion_obs = self.env.unwrapped.collect_reference_motions(4000)
            self.reference_motion_buffer.add_records({"motion_observations": motion_obs})

        self.hisotry_motion_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            100
        )

        self.hisotry_motion_buffer.create_storage_space("motion_observations", (motion_dim,))

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        log_prob = actor_step.log_prob
        if determine:
            action = actor_step.mean
        
        critic_step:ValueStep = self.critic(obs_batch)
        value = critic_step.value

        return action, log_prob, value
    
    @torch.no_grad()
    def get_discriminator_reward(self, motion_obs_batch: torch.Tensor) -> torch.Tensor:
        disc_step:ValueStep = self.discriminator(motion_obs_batch)
        rewards = -torch.log(1 - 1 / (1 + torch.exp(-disc_step.value)) + 1e-5)
        return rewards, disc_step.value
    
    def rollout(self, obs, info):
        rewards_sum = 0
        logit_sum = 0
        for _ in range(self.steps):
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            motion_obs = info["amp_obs"]
            disc_reward, logit = self.get_discriminator_reward(motion_obs)
            reward = task_reward * 0 + disc_reward * 2.0

            rewards_sum += reward.mean()
            logit_sum += logit.mean()

            done = terminate | timeout
            
            records = {
                "observations": obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "motion_observations": motion_obs,
                "values": value,
                "dones": done
            }

            motion_record = {
                "motion_observations": motion_obs
            }

            self.rollout_buffer.add_records(records)
            self.hisotry_motion_buffer.add_records(motion_record)

            obs = next_obs

        print(rewards_sum/self.steps)
        print(logit_sum/self.steps)
        print("------------------")

        last_obs = process_obs(obs)
        _, _, last_value = self.get_action(last_obs)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["dones"],
            last_value,
            0.99,
            0.95
        )
        
        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        motion_obs = self.env.unwrapped.collect_reference_motions(4000)
        self.reference_motion_buffer.add_records({"motion_observations": motion_obs})

        return obs, info
    
    def update(self):
        for _ in range(5):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, 4096*10):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                policy_loss, entropy, kl_divergence = PPO.compute_policy_loss(self.actor,
                                                                              log_prob_batch,
                                                                              obs_batch,
                                                                              action_batch,
                                                                              advantage_batch,
                                                                              0.2,
                                                                              1e-4)

                value_loss = PPO.compute_clipped_value_loss(self.critic,
                                                    obs_batch,
                                                    value_batch,
                                                    return_batch,
                                                    0.2)
                
                loss = policy_loss + value_loss * 2.5 - entropy * 1e-3

                self.ac_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.ac_optimizer.step()
                
            '''
            current_motion_batch = self.rollout_buffer.sample_tensor(
                    "motion_observations",
                    4096
                ).to(self.device)

            reference_motion_batch = self.reference_motion_buffer.sample_tensor(
                "motion_observations",
                4096
            ).to(self.device)

            history_motion_batch = self.hisotry_motion_buffer.sample_tensor(
                "motion_observations",
                4096
            ).to(self.device)

            agent_motion_batch = torch.cat([current_motion_batch, history_motion_batch])
            
            d_loss = GAN.compute_bce_loss(self.discriminator,
                                            reference_motion_batch,
                                            agent_motion_batch,
                                            r1_gamma=5.0) * 0.5
            
            self.d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            self.d_optimizer.step()
            '''

    def train(self):
        obs, info = self.env.reset()
        for epoch in trange(1000):
            obs, info = self.rollout(obs, info)
            self.update()
        self.env.close()

        torch.save(
            [self.actor.state_dict(), self.critic.state_dict()],
            "weight.pth"
        )

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    simulation_app.close()