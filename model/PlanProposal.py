import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal


# The plan proposal network
# Input the perceptual embedding and the latent goal, output the proposed plan
class PlanMLP(nn.Module):
    def __init__(self,
                 perceptual_features,
                 latent_goal_features: int = 32,
                 plan_features: int = 256,
                 min_std: float = 0.0001):
        super(PlanMLP, self).__init__()
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.min_std = min_std
        self.in_features = self.perceptual_features + self.latent_goal_features
        self.act_fn = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=2048),
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
            nn.Linear(in_features=2048, out_features=2048),
            self.act_fn,
        )
        self.mean_fc = nn.Linear(in_features=2048, out_features=self.plan_features)
        self.variance_fc = nn.Linear(in_features=2048, out_features=self.plan_features)

    def forward(self, initial_state, latent_goal):
        x = torch.cat([initial_state, latent_goal], dim=-1)
        x = self.mlp(x)
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pp_dist = Independent(Normal(mean, std), 1)
        return pp_dist