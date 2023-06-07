import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


# The plan recognition network
# input the perceptual embedding, output the encoded plan
class PlanRNN(nn.Module):
    def __init__(self,
                 in_features,
                 plan_features: int = 256,
                 action_space: int = 7,
                 min_std: float = 0.0001):
        super(PlanRNN, self).__init__()
        self.plan_features = plan_features
        self.action_space = action_space
        self.min_std = min_std
        self.in_features = in_features
        self.rnn = nn.RNN(
            input_size=self.in_features,
            hidden_size=2048,
            nonlinearity="relu",
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        self.mean_fc = nn.Linear(in_features=4096, out_features=self.plan_features)
        self.variance_fc = nn.Linear(in_features=4096, out_features=self.plan_features)

    def forward(self, perceptual_emb):
        x, hn = self.rnn(perceptual_emb)
        x = x[:, -1]
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pr_dist = Independent(Normal(mean, std), 1)
        return pr_dist
