import torch.nn as nn


# The language goal encoder
# Encode the embedded natural language instruction to the latent space
class LanguageGoalEncoder(nn.Module):
    def __init__(self,
                 in_features: int = 384,
                 lantent_goal_feature: int = 32,
                 hidden_size: int = 2048):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            self.act_fn,
            nn.Linear(hidden_size, hidden_size),
            self.act_fn,
            nn.Linear(hidden_size, lantent_goal_feature)
        )

    def forward(self, x):
        return self.mlp(x)
