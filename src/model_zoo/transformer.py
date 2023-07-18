# TODO: Remove or Document

import math
import torch
import torch.nn as nn
from timm.models.layers import drop_path


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# BEiTv2 block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, xq, xk, xv, attn_mask=None, key_padding_mask=None):
        if self.gamma_1 is None:
            x = xq + self.drop_path(
                self.attn(
                    self.norm1(xq),
                    self.norm1(xk),
                    self.norm1(xv),
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = xq + self.drop_path(
                self.gamma_1
                * self.attn(
                    self.norm1(xq),
                    self.norm1(xk),
                    self.norm1(xv),
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Tmixer(nn.Module):
    def __init__(self, n, head_size=32, num_layers=2, **kwargs):
        super().__init__()
        self.seq_enc = SinusoidalPosEmb(n)
        self.blocks = nn.ModuleList([Block(n, n // 64) for i in range(num_layers)])

    def forward(self, x, frame_idx=-1):
        B, N, C, H, W = x.shape
        x = x.flatten(-2, -1).permute(0, 1, 3, 2)  # bs x n x hw x c

        enc = self.seq_enc(torch.arange(N, device=x.device)).view(1, N, 1, C)
        xq = x[:, frame_idx] + enc[:, frame_idx]  # frame 4
        xk = (x + enc).flatten(1, 2)  # bs x nhw x c
        xv = x.flatten(1, 2)  # bs x nhw x c

        for m in self.blocks:
            xq = m(xq, xk, xv)

        x = xq.view(B, H, W, C).permute(0, 3, 1, 2)  # bs x c x h x w
        return x
