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
from isaaclab_tasks.direct.humanoid.humanoid_env import HumanoidEnvCfg

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
        #self.cfg = HumanoidEnvCfg()
        #self.env_name = "Isaac-Humanoid-Direct-v0"
        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        obs_dim = self.cfg.observation_space
        motion_dim = self.cfg.amp_observation_space * self.cfg.num_amp_observations
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.discriminator = Discriminator(motion_dim).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(params, lr=1e-3)
        
        self.d_optimizer = torch.optim.Adam(
            [
                {'params': self.discriminator.encoder.parameters(), "weight_decay":1e-4},
                {'params': self.discriminator.head.parameters(), "weight_decay":1e-2},
            ],
            lr=5e-5, betas=(0.5, 0.999)
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

        self.expert_motion_buffer = ReplayBuffer(
            500,
            400
        )
        
        self.expert_motion_buffer.create_storage_space("motion_observations", (motion_dim,))

        for _ in range(400):
            motion_obs = self.env.unwrapped.collect_reference_motions(500)
            self.expert_motion_buffer.add_records({"motion_observations": motion_obs})

        self.agent_motion_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            100
        )

        self.agent_motion_buffer.create_storage_space("motion_observations", (motion_dim,))
        
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
        rewards = -torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-disc_step.value)),
                                            torch.tensor(0.0001, device=self.device)))
        return rewards, disc_step.value
    
    def rollout(self, obs):
        rewards_sum = 0
        logit_sum = 0
        for _ in range(self.steps):
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            motion_obs = info["amp_obs"]
            disc_reward, logit = self.get_discriminator_reward(motion_obs)
            reward = task_reward * 0. + disc_reward * 2.0
            #reward = task_reward

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
            self.agent_motion_buffer.add_records(motion_record)

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

        motion_obs = self.env.unwrapped.collect_reference_motions(500)
        self.expert_motion_buffer.add_records({"motion_observations": motion_obs})

        return obs
    
    def update(self):
        for _ in range(5):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, 4096*10):
                obs_batch = batch["observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                policy_loss_dict = PPO.compute_policy_loss(self.actor,
                                                                              log_prob_batch,
                                                                              obs_batch,
                                                                              action_batch,
                                                                              advantage_batch,
                                                                              0.2,
                                                                              0.0)
                
                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = PPO.compute_clipped_value_loss(self.critic,
                                                    obs_batch,
                                                    value_batch,
                                                    return_batch,
                                                    0.2)
                
                value_loss = value_loss_dict["loss"]
                
                loss = policy_loss + value_loss * 2.5 - entropy * 0.0

                self.ac_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.ac_optimizer.step()
                
                
                current_motion_batch = self.rollout_buffer.sample_tensor(
                        "motion_observations",
                        4096
                    ).to(self.device)

                expert_motion_batch = self.expert_motion_buffer.sample_tensor(
                    "motion_observations",
                    4096
                ).to(self.device)

                past_motion_batch = self.agent_motion_buffer.sample_tensor(
                    "motion_observations",
                    4096
                ).to(self.device)

                agent_motion_batch = torch.cat([current_motion_batch, past_motion_batch])
                
                d_loss_dict = GAN.compute_bce_loss(self.discriminator,
                                                expert_motion_batch,
                                                agent_motion_batch,
                                                detach_fake=False,
                                                r1_gamma=5.0)
                
                d_loss = d_loss_dict["loss"] * 5.0
                d_loss_real = d_loss_dict["loss_real"]
                d_loss_fake = d_loss_dict["loss_fake"]
                d_loss_gp = d_loss_dict["gradient_penalty"]
                
                self.d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                self.d_optimizer.step()

    def train(self):
        obs, _ = self.env.reset()
        for epoch in trange(500):
            obs = self.rollout(obs)
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