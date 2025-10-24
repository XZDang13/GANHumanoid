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
        self.env = gymnasium.make("Isaac-Humanoid-AMP-Walk-Direct-v0", cfg=self.cfg)

        obs_dim = self.cfg.observation_space
        reference_motion_dim = self.cfg.amp_observation_space * self.cfg.num_amp_observations
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.discriminator = Discriminator(reference_motion_dim).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.discriminator.parameters())
        self.optimizer = torch.optim.Adam(params, lr=5e-5)

        self.steps = 25

        self.rollout_buffer = ReplayBuffer(self.cfg.scene.num_envs,
                                           self.steps
                                           )
        
        self.rollout_buffer.create_storage_space("observations", (obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("reference_observations", (reference_motion_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("dones", (), torch.float32)

        self.batch_keys = ["observations", "reference_observations", "actions", "log_probs", "rewards", "values", "returns", "advantages"]

        self.reference_buffer = ReplayBuffer(200, 1000)
        self.reference_buffer.create_storage_space("reference_observations", (reference_motion_dim,), torch.float32)
        for i in range(1000):
            reference_motions = self.env.unwrapped.collect_reference_motions(200)
            self.reference_buffer.add_records({"reference_observations": reference_motions})

        self.replay_buffer = ReplayBuffer(self.cfg.scene.num_envs, 100)
        self.replay_buffer.create_storage_space("reference_observations", (reference_motion_dim,), torch.float32)

        self.obs = None

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
        return rewards
    
    def rollout(self):
        self.actor.eval()
        self.critic.eval()

        obs = self.obs
        for i in range(self.steps):
            obs = process_obs(obs)
            action, log_prob, value = self.get_action(obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)

            motion_obs = info["amp_obs"]
            disc_reward = self.get_discriminator_reward(motion_obs)
            
            reward = disc_reward * 1.0 + task_reward * 0.0
            
            done = terminate | timeout

            record = {
                "observations": obs,
                "reference_observations": motion_obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "dones": done
            }

            self.rollout_buffer.add_records(record)
            self.replay_buffer.add_records({"reference_observations": motion_obs})

            obs = next_obs

        self.obs = obs
        obs = process_obs(obs)
        _, _, value = self.get_action(obs)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["dones"],
            value,
            0.99,
            0.95
            )
        
        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        self.actor.train()
        self.critic.train()

    def update(self):
        policy_loss_buffer = []
        value_loss_buffer = []
        discriminator_loss_buffer = []
        for _ in range(10):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, 4096):
                obs_batch = batch["observations"].to(self.device)
                motion_obs_batch = batch["reference_observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                reference_motion_obs_batch = self.reference_buffer.sample_batch(["reference_observations"],
                                                                                2*4096)["reference_observations"].to(self.device)
                

                history_motion_obs_batch = self.replay_buffer.sample_batch(["reference_observations"],
                                                                                4096)["reference_observations"].to(self.device)
                
                motion_obs_batch = torch.cat([motion_obs_batch, history_motion_obs_batch], dim=0)

                policy_loss, entropy, kl_divergence = PPO.compute_policy_loss(self.actor, log_prob_batch, obs_batch,
                                                                              action_batch, advantage_batch,
                                                                              0.2,
                                                                              1e-4)
                value_loss = PPO.compute_clipped_value_loss(self.critic, obs_batch, value_batch,
                                                            return_batch, 0.2)
                
                discriminator_loss = GAN.compute_bce_loss(self.discriminator, reference_motion_obs_batch,
                                                            motion_obs_batch,
                                                            label_smoothing=0.2,
                                                            r1_gamma=5.0)


                loss = policy_loss + value_loss * 0.5 - entropy * 0.01 + discriminator_loss * 5.0

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                discriminator_loss_buffer.append(discriminator_loss.item())

        print(f"Policy Loss: {np.mean(policy_loss_buffer):.4f}, "
              f"Value Loss: {np.mean(value_loss_buffer):.4f}, "
              f"Discriminator Loss: {np.mean(discriminator_loss_buffer):.4f}")


    def train(self):
        obs, info = self.env.reset()
        self.obs = obs
        for epoch in trange(250):
            self.rollout()
            self.update()

        self.env.close()

        torch.save([self.discriminator.state_dict(), self.actor.state_dict(), self.critic.state_dict()], "model.pth")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    simulation_app.close()