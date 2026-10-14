import math
from copy import deepcopy

import torch
import torch.nn as nn


class RMSNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.view(1, -1, 1)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def MLP(channels: list, do_bn=True, dropout=0.0):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(RMSNorm1d(channels[i]))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers, use_positional_encoding=True):
        super().__init__()
        self.use_pe = use_positional_encoding

        if use_positional_encoding:
            self.pe_dim = 32
            input_dim = 2 + self.pe_dim * 2
        else:
            input_dim = 2

        self.encoder = MLP([input_dim] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
        nn.init.normal_(self.encoder[-1].weight, std=0.01)

        self.output_scale = nn.Parameter(torch.ones(1))

    def _positional_encoding(self, coords):
        freq_bands = torch.exp(
            torch.linspace(0, math.log(1000.0), self.pe_dim // 2, device=coords.device)
        )
        coords_expanded = coords.unsqueeze(-1)
        freq_bands = freq_bands.view(1, 1, 1, -1)
        angles = coords_expanded * freq_bands
        pe_sin = torch.sin(angles)
        pe_cos = torch.cos(angles)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        batch, _, N = coords.shape
        pe = pe.permute(0, 1, 3, 2).reshape(batch, 2 * self.pe_dim, N)
        return torch.cat([coords, pe], dim=1)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        if self.use_pe:
            inputs = self._positional_encoding(inputs)
        output = self.encoder(inputs)
        output = output * self.output_scale
        return output


def normalize_keypoints(kpts, image_shape):
    height, width = image_shape[0], image_shape[1]
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.prob = None

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob = prob
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        hidden_dim = feature_dim * 4
        self.mlp = nn.Sequential(
            nn.Conv1d(feature_dim * 2, hidden_dim, kernel_size=1),
            RMSNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=1)
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        feat = torch.cat([x, message], dim=1)
        feat = self.mlp(feat)
        return feat


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 1)])

    def forward(self, desc):
        for layer in self.layers:
            layer.attn.prob = []
            delta = layer(desc, desc)
            desc = desc + delta
        return desc


class Matchsc_st(nn.Module):
    def __init__(self, descriptor_dim, keypoint_encoder):
        super().__init__()
        self.kenc = KeypointEncoder(descriptor_dim, keypoint_encoder)
        self.gnn = AttentionalGNN(descriptor_dim)
        self.final_proj = nn.Conv1d(descriptor_dim, descriptor_dim, kernel_size=1, bias=True)

    def forward(self, data):
        descriptors = data['descripScale'].transpose(0, 1)
        keypoints = torch.reshape(data['keypoints'], (1, -1, 2))
        keypoints = normalize_keypoints(keypoints, data['location'])

        descriptors = descriptors + self.kenc(keypoints)
        descriptors = self.gnn(descriptors)
        descriptors = self.final_proj(descriptors)
        return descriptors
