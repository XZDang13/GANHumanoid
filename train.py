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

from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_advantage
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.alg.ppo import PPO
from RLAlg.alg.gan import GAN
from RLAlg.logger import WandbLogger

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

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.actor.parameters(),
                 "name": "actor"},
                 {'params': self.critic.parameters(),
                 "name": "critic"},
                {'params': self.discriminator.encoder.parameters(),
                 "weight_decay":1e-3,
                 "name": "discriminator"},
                {'params': self.discriminator.head.parameters(),
                 "weight_decay":1e-1,
                 "name": "discriminator_head"},
            ],
            lr=5e-5)
        
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

        self.global_step = 0   
        WandbLogger.init_project("AMP", f"Humanoid")
        
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
        self.actor.eval()
        self.critic.eval()
        self.discriminator.eval()
        for _ in range(self.steps):
            self.global_step += 1
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            motion_obs = info["amp_obs"]
            disc_reward, logit = self.get_discriminator_reward(motion_obs)
            reward = task_reward * 0.0 + disc_reward * 2.0
            #reward = task_reward

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

            step_info = {}
            step_info['step/mean_reward'] = disc_reward.mean().item()
            step_info['step/mean_logit'] = logit.mean().item()
                
            WandbLogger.log_metrics(step_info, self.global_step)

        #last_obs = process_obs(obs)
        #_, _, last_value = self.get_action(last_obs)
        #returns, advantages = compute_gae(
        #    self.rollout_buffer.data["rewards"],
        #    self.rollout_buffer.data["values"],
        #    self.rollout_buffer.data["dones"],
        #    last_value,
        #    0.99,
        #    0.95
        #)

        returns, advantages = compute_advantage(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["dones"],
        )
        
        #print(returns)

        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        expert_motion_obs = self.env.unwrapped.collect_reference_motions(500)
        self.expert_motion_buffer.add_records({"motion_observations": expert_motion_obs})

        self.actor.train()
        self.critic.train()
        self.discriminator.train()

        return obs
    
    def update(self):
        policy_loss_buffer = []
        value_loss_buffer = []
        entropy_buffer = []
        kl_divergence_buffer = []
        discriminator_loss_buffer = []
        discriminator_real_loss_buffer = []
        discriminator_fake_loss_buffer = []
        discriminator_gradient_penalty_buffer = []

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

                agent_motion_batch.requires_grad_(True)
                
                d_loss_dict = GAN.compute_bce_loss(self.discriminator,
                                                expert_motion_batch,
                                                agent_motion_batch,
                                                detach_fake=True,
                                                r1_gamma=5.0)
                
                d_loss = d_loss_dict["loss"]
                d_loss_real = d_loss_dict["loss_real"]
                d_loss_fake = d_loss_dict["loss_fake"]
                d_loss_gp = d_loss_dict["gradient_penalty"]

                loss = policy_loss - entropy * 0.001 + value_loss * 0.5 + d_loss * 5.0

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.optimizer.step()
                

                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                entropy_buffer.append(entropy.item())
                kl_divergence_buffer.append(kl_divergence.item())
                discriminator_loss_buffer.append(d_loss.item())
                discriminator_real_loss_buffer.append(d_loss_real.item())
                discriminator_fake_loss_buffer.append(d_loss_fake.item())
                discriminator_gradient_penalty_buffer.append(d_loss_gp.item())

        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_value_loss = np.mean(value_loss_buffer)
        avg_entropy = np.mean(entropy_buffer)
        avg_kl_divergence = np.mean(kl_divergence_buffer)
        avg_discriminator_loss = np.mean(discriminator_loss_buffer)
        avg_discriminator_real_loss = np.mean(discriminator_real_loss_buffer)
        avg_discriminator_fake_loss = np.mean(discriminator_fake_loss_buffer)
        avg_discriminator_gp_loss = np.mean(discriminator_gradient_penalty_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_entropy": avg_entropy,
            "update/avg_kl_divergence": avg_kl_divergence,
            "update/avg_discriminator_loss": avg_discriminator_loss,
            "update/avg_discriminator_real_loss": avg_discriminator_real_loss,
            "update/avg_discriminator_fake_loss": avg_discriminator_fake_loss,
            "update/avg_discriminator_gp_loss": avg_discriminator_gp_loss
        }

        WandbLogger.log_metrics(train_info, self.global_step)

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