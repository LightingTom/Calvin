from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


# The action decoder (use the original code)
# Input the latent goal, latent plan, perceptual embedding, output the decoded plan
class ActionDecoder(nn.Module):
    def __init__(self,
                 perceptual_features: int,
                 n_dist: int = 10,
                 latent_goal_features: int = 32,
                 plan_features: int = 256,
                 hidden_size: int = 2048,
                 out_features: int = 7,
                 log_scale_min: float = -7.0,
                 num_classes: int = 256):
        super(ActionDecoder, self).__init__()
        self.n_dist = n_dist
        self.log_scale_min = log_scale_min
        self.num_classes = num_classes
        self.plan_features = plan_features
        in_features = perceptual_features + latent_goal_features + plan_features
        self.out_features = out_features
        self.rnn = nn.RNN(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=2,
            nonlinearity="relu",
            bidirectional=False,
            batch_first=True,
            dropout=0,
        )
        self.mean_fc = nn.Linear(hidden_size, out_features * self.n_dist)
        self.log_scale_fc = nn.Linear(hidden_size, out_features * self.n_dist)
        self.prob_fc = nn.Linear(hidden_size, out_features * self.n_dist)
        self.register_buffer("one_hot_embedding_eye", torch.eye(self.n_dist))
        self.register_buffer("ones", torch.ones(1, 1, self.n_dist))
        act_max_bound = [0.42977 , 0.139396, 0.796262, 3.141592, 0.638583, 3.141551,        1.]
        act_min_bound = [-0.432188, -0.545456,  0.293439, -3.141593, -0.811348, -3.141573, -1.]
        action_max_bound = torch.Tensor(act_max_bound).float()
        action_min_bound = torch.Tensor(act_min_bound).float()
        assert action_max_bound.shape[0] == self.out_features
        assert action_min_bound.shape[0] == self.out_features
        action_max_bound = action_max_bound.unsqueeze(0).unsqueeze(0)  # [1, 1, action_space]
        action_min_bound = action_min_bound.unsqueeze(0).unsqueeze(0)  # [1, 1, action_space]
        action_max_bound = action_max_bound.unsqueeze(-1) * self.ones  # broadcast to [1, 1, action_space, N_DIST]
        action_min_bound = action_min_bound.unsqueeze(-1) * self.ones  # broadcast to [1, 1, action_space, N_DIST]
        self.register_buffer("action_max_bound", action_max_bound)
        self.register_buffer("action_min_bound", action_min_bound)
        self.one_hot_embedding_eye: torch.Tensor = self.one_hot_embedding_eye
        self.action_max_bound: torch.Tensor = self.action_max_bound
        self.action_min_bound: torch.Tensor = self.action_min_bound

        self.hidden_state = None

    def clean_hidden(self):
        self.hidden_state = None

    def loss_and_act(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ):
        logit_probs, log_scales, means, _ = self(latent_plan, perceptual_emb, latent_goal)
        # loss
        loss = self._loss(logit_probs, log_scales, means, actions)
        # act
        pred_actions = self._sample(logit_probs, log_scales, means)
        return loss, pred_actions

    def act(self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor) -> torch.Tensor:
        logit_probs, log_scales, means, self.hidden_state = self(
            latent_plan, perceptual_emb, latent_goal, self.hidden_state
        )
        return self._sample(logit_probs, log_scales, means)

    def loss(
        self, latent_plan: torch.Tensor, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ):
        logit_probs, log_scales, means, _ = self(latent_plan, perceptual_emb, latent_goal)
        return self._loss(logit_probs, log_scales, means, actions)

    def _loss(
        self,
        logit_probs: torch.Tensor,
        log_scales: torch.Tensor,
        means: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Appropriate scale
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)
        # Broadcast actions (B, A, N_DIST)
        actions = actions.unsqueeze(-1) * self.ones
        # Approximation of CDF derivative (PDF)
        centered_actions = actions - means
        inv_stdv = torch.exp(-log_scales)
        assert torch.is_tensor(self.action_max_bound)
        assert torch.is_tensor(self.action_min_bound)
        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            actions < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                actions > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.num_classes - 1) / 2),
                ),
            ),
        )
        log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)
        loss = -torch.sum(log_sum_exp(log_probs), dim=-1).mean()
        return loss

    # Sampling from logistic distribution
    def _sample(self, logit_probs: torch.Tensor, log_scales: torch.Tensor, means: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Selecting Logistic distribution (Gumbel Sample)
        r1, r2 = 1e-5, 1.0 - 1e-5
        temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        temp = logit_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)
        # TODO: find out why mypy complains about type
        dist = self.one_hot_embedding_eye[argmax]

        # Select scales and means
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales)  # Make positive
        u = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        actions = means + scales * (torch.log(u) - torch.log(1.0 - u))

        return actions

    def forward(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        latent_plan = latent_plan.unsqueeze(1).expand(-1, seq_len, -1) if latent_plan.nelement() > 0 else latent_plan
        latent_goal = latent_goal.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([latent_plan, perceptual_emb, latent_goal], dim=-1)  # b, s, (plan + visuo-propio + goal)
        self.rnn.flatten_parameters()
        x, h_n = self.rnn(x, h_0)
        probs = self.prob_fc(x)
        means = self.mean_fc(x)
        log_scales = self.log_scale_fc(x)
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)
        # Appropriate dimensions
        logit_probs = probs.view(batch_size, seq_len, self.out_features, self.n_dist)
        means = means.view(batch_size, seq_len, self.out_features, self.n_dist)
        log_scales = log_scales.view(batch_size, seq_len, self.out_features, self.n_dist)
        return logit_probs, log_scales, means, h_n
