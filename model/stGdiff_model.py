import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
import einops

from model import stGdiff_match


class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)
        qkv = einops.rearrange(qkv, 'b l n h d -> n b h l d')
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einops.rearrange(attn @ v, 'b h l d -> b l (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    dtype = x.dtype
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x.to(dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention2(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_stGdiff(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        depth,
        dit_type,
        num_heads,
        classes,
        mlp_ratio=4.0,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.time_emb = TimestepEmbedder(hidden_size)
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.cond_layer = nn.Linear(input_size, hidden_size)

        self.spatial_encoder = stGdiff_match.Matchsc_st(
            descriptor_dim=input_size,
            keypoint_encoder=[128],
        )

        self.blks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.out_layer = FinalLayer(hidden_size, input_size)
        self.condi_emb = nn.Embedding(classes, hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        for block in self.blks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def _spatial_encode(self, expr, coords):
        coord_min = coords.min(dim=0).values
        coord_max = coords.max(dim=0).values
        coord_range = coord_max - coord_min

        keypoints = coords.unsqueeze(0).unsqueeze(0)
        descriptors = expr.T.unsqueeze(1)

        encoder_input = {
            'keypoints': keypoints,
            'descripScale': descriptors,
            'location': coord_range,
        }

        encoded = self.spatial_encoder(encoder_input)
        return encoded.squeeze(0).T

    def forward(self, x, t, y, LC, **kwargs):
        t = self.time_emb(t)
        x = x.float()
        y = y.float()

        x = self._spatial_encode(x, LC)
        y = self._spatial_encode(y, LC)

        x = self.in_layer(x)
        y = self.cond_layer(y)

        c = t + y

        x = x.unsqueeze(1)

        for blk in self.blks:
            x = blk(x, c)

        x = self.out_layer(x, c)
        x = x.squeeze(1)

        return x
