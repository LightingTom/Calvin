import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F


# Use Spatial Softmax as in the original code
class SpatialSoftmax(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, temperature):
        """
        Computes the spatial softmax of a convolutional feature map.
        Read more here:
        "Learning visual feature spaces for robotic manipulation with
        deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
        :param num_rows:  size related to original image width
        :param num_cols:  size related to original image height
        :param temperature: Softmax temperature (optional). If None, a learnable temperature is created.
        """
        super(SpatialSoftmax, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, num_cols), torch.linspace(-1.0, 1.0, num_rows), indexing="ij"
        )
        x_map = grid_x.reshape(-1)
        y_map = grid_y.reshape(-1)
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        if temperature:
            self.register_buffer("temperature", torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.view(-1, h * w)  # batch, C, W*H
        softmax_attention = F.softmax(x / self.temperature, dim=1)  # batch, C, W*H
        expected_x = torch.sum(self.x_map * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.y_map * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat((expected_x, expected_y), 1)
        self.coords = expected_xy.view(-1, c * 2)
        return self.coords  # batch, C*2


# Encoder the image from observation space
class VisualEncoder(nn.Module):
    def __init__(self,
                 visual_features,
                 num_c):
        super(VisualEncoder, self).__init__()
        self.act_fn = nn.ReLU()
        self.spatial_softmax = SpatialSoftmax(num_rows=21, num_cols=21, temperature=1.0)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_c, out_channels=32, kernel_size=8, stride=4),  # shape: [N, 32, 49, 49]
            self.act_fn,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # shape: [N, 64, 23, 23]
            self.act_fn,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # shape: [N, 64, 21, 21]
            self.act_fn
        )
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            self.act_fn,
            nn.Linear(in_features=512, out_features=visual_features)
        ) #output: (N, 64)

    def forward(self, x):
        x = self.cnn(x)
        x = self.spatial_softmax(x)
        return self.mlp_layer(x)


# Output the original proprioception state
class ProprioceptionEncoder(nn.Module):
    def __init__(self):
        super(ProprioceptionEncoder, self).__init__()
        self.n_state_obs = int(np.sum(np.diff([list(x) for x in [list(y) for y in [[0,15]]]])))
        self.layer = nn.Identity()

    @property
    def out_features(self):
        return self.n_state_obs

    def forward(self, x):
        return self.layer(x)


# The perceptual encoder, calculate the perceptual embedding
class PerceptualEncoder(nn.Module):
    def __init__(self):
        super(PerceptualEncoder, self).__init__()
        self._latent_size = 64
        self.vis_encoder = VisualEncoder(64, 3)
        self.pro_encoder = ProprioceptionEncoder()
        self._latent_size += self.pro_encoder.out_features

    @property
    def latent_size(self):
        return self._latent_size

    def forward(self, imgs_static, state_obs):
        batch_size, seq_len, n_channel, height, width = imgs_static.shape
        imgs_static = imgs_static.reshape(-1, n_channel, height, width).float()
        encoded_imgs = self.vis_encoder(imgs_static)
        encoded_imgs = encoded_imgs.reshape(batch_size, seq_len, -1)
        state_obs_out = self.pro_encoder(state_obs)
        perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
        return perceptual_emb
