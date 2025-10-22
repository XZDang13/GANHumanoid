import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import MLPLayer, GaussianHead, CriticHead
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep

class Actor(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super().__init__()

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
        )

        self.head = GaussianHead(256, action_dim)

    def forward(self, obs:torch.Tensor, action:torch.Tensor|None=None) -> StochasticContinuousPolicyStep:
        x = self.encoder(obs)
        step = self.head(x, action)

        return step
    
class Critic(nn.Module):
    def __init__(self, obs_dim:int):
        super().__init__()

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
        )

        self.head = CriticHead(256)

    def forward(self, obs:torch.Tensor) -> ValueStep:
        x = self.encoder(obs)
        step = self.head(x)

        return step
    
class Discriminator(nn.Module):
    def __init__(self, obs_dim:int):
        super().__init__()

        self.encoder = nn.Sequential(
            MLPLayer(obs_dim, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
            MLPLayer(256, 256, nn.SiLU(), True),
        )

        self.head = CriticHead(256)

    def forward(self, obs:torch.Tensor) -> ValueStep:
        x = self.encoder(obs)
        step = self.head(x)

        return step