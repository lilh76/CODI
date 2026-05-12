from torch import nn as nn


class BehaviorClone(nn.Module):
    def __init__(
        self,
        model,
        observation_dim,
        action_dim,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.model = model

    def loss(self, observations, actions):
        log_prob = self.model.get_log_prob(observations, actions).mean()
        loss = -1.0 * log_prob
        info = dict(bc_loss=loss, log_prob=log_prob)
        return loss, info

    def forward(self, observations, deterministic=False):
        return self.model(observations, deterministic=deterministic)[0]
