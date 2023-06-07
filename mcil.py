from model.PerceptualEncoder import PerceptualEncoder
from model.VisualGoalEncoder import VisualGoalEncoder
from model.ActionDecoder import ActionDecoder
from model.LanguageGoalEncoder import LanguageGoalEncoder
from model.PlanRecognition import PlanRNN
from model.PlanProposal import PlanMLP
import torch.nn as nn
import torch.distributions as D


# The MCIL model
class MCILModel(nn.Module):
    def __init__(self):
        super(MCILModel, self).__init__()
        self.perceptual_encoder = PerceptualEncoder()
        latent_size = self.perceptual_encoder.latent_size
        self.vis_encoder = VisualGoalEncoder(latent_size)
        self.lang_encoder = LanguageGoalEncoder()
        self.plan_prior = PlanMLP(latent_size)
        self.plan_posterior = PlanRNN(latent_size)
        self.action_decoder = ActionDecoder(latent_size)
        self.kl_beta = 0.001

    def compute_kl_loss(self, pr_dist, pp_dist):
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def forward(self, img_static, robot_obs, lang, actions):
        p_embedding = self.perceptual_encoder(img_static, robot_obs)
        if lang is not None:
            latent_goal = self.lang_encoder(lang)
        else:
            latent_goal = self.vis_encoder(p_embedding[:, -1])

        pp_dist = self.plan_prior(p_embedding[:, 0], latent_goal)
        pr_dist = self.plan_posterior(p_embedding)
        sampled_plan = pr_dist.rsample()
        action_loss = self.action_decoder.loss(sampled_plan, p_embedding, latent_goal, actions)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        total_loss = action_loss + kl_loss
        return kl_loss, action_loss, total_loss, pp_dist, pr_dist
