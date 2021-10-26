import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEncoder(nn.Module):
    def __init__(
            self,
            input_dim=2,
            feat_dim=64,
        ):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, feat_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class LinearDecoder(nn.Module):
    def __init__(
            self,
            input_dim=2,
            feat_dim=64,
        ):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Linear(feat_dim, input_dim)
        # self.xy_decoder = nn.Linear(feat_size, 5)
        # self.flag_decoder = nn.Linear(feat_size, 3)

    def forward(self, x):
        x = self.linear(x)
        return x


class VideoEncoderCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        channels_mult=1,
    ):
        super(VideoEncoderCNN, self).__init__()

        self._channels_mult = channels_mult

        filters = 32
        self._h1_conv = nn.Conv2d(
            in_channels, 
            self._channels_mult*filters, 
            kernel_size=4, 
            stride=2,
        )
        self._h2_conv = nn.Conv2d(
            self._channels_mult*filters,   
            self._channels_mult*filters*2, 
            kernel_size=4, 
            stride=2,
        )
        self._h3_conv = nn.Conv2d(
            self._channels_mult*filters*2, 
            self._channels_mult*filters*4, 
            kernel_size=4, 
            stride=2,
        )
        self._h4_conv = nn.Conv2d(
            self._channels_mult*filters*4, 
            self._channels_mult*filters*8, 
            kernel_size=4, 
            stride=2,
        )


    def forward(self, obs):
        # obs: (batch, time, c, w, h)
        batch_size, n_timesteps, c, w, h = list(obs.shape)
        hidden = obs.view(-1, *obs.shape[2:])
        hidden = F.leaky_relu(self._h1_conv(hidden))
        hidden = F.leaky_relu(self._h2_conv(hidden))
        hidden = F.leaky_relu(self._h3_conv(hidden))
        hidden = F.leaky_relu(self._h4_conv(hidden))
        hidden = hidden.view(batch_size, n_timesteps, -1)

        return hidden



class VideoDecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels, channels_mult=1):
        super(VideoDecoderCNN, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channels_mult = channels_mult

        self._h1 = nn.Linear(in_channels, self._channels_mult*1024)

        filters = 32
        self._h2 = nn.ConvTranspose2d(
            in_channels=self._channels_mult*1024,
            out_channels=self._channels_mult*filters*4,
            kernel_size=5,
            stride=2,
        )
        self._h3 = nn.ConvTranspose2d(
            in_channels=self._channels_mult*filters*4,
            out_channels=self._channels_mult*filters*2,
            kernel_size=5,
            stride=2,
        )
        self._h4 = nn.ConvTranspose2d(
            in_channels=self._channels_mult*filters*2,
            out_channels=self._channels_mult*filters,
            kernel_size=6,
            stride=2,
        )
        self._out = nn.ConvTranspose2d(
            in_channels=self._channels_mult*filters,
            out_channels=self._out_channels,
            kernel_size=6,
            stride=2,
        )

    def forward(self, states):
        batch_size, n_timesteps, _ = list(states.shape)
        hidden = self._h1(states)
        hidden = hidden.view(-1, hidden.shape[-1], 1, 1)
        hidden = F.leaky_relu(self._h2(hidden))
        hidden = F.leaky_relu(self._h3(hidden))
        hidden = F.leaky_relu(self._h4(hidden))
        out = F.leaky_relu(self._out(hidden))
        out = out.view(batch_size, n_timesteps, -1, out.shape[2], out.shape[3])
        return out