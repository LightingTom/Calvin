import torch.nn as nn


# The visual goal encoder
# Input the perceptual embedding, encode it to the latent space
class VisualGoalEncoder(nn.Module):
    def __init__(self,
                 in_feature,
                 hidden_size: int = 2048,
                 latent_goal_features: int = 32
                 ):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            self.act_fn,
            nn.Linear(hidden_size, hidden_size),
            self.act_fn,
            nn.Linear(hidden_size, latent_goal_features),
        )

    def forward(self, x):
        return self.mlp(x)
