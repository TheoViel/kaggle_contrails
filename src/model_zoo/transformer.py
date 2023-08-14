import math
import torch
import torch.nn as nn
from timm.models.layers import drop_path


class DropPath(nn.Module):
    """
    DropPath module for stochastic depth regularization.

    This module implements drop path regularization for stochastic depth training.
    During training, each element of the input tensor is randomly set to zero with a certain probability.
    During evaluation, this module has no effect and simply passes the input tensor through.

    Methods:
        forward(x):
            Applies drop path regularization during training.

    Attributes:
        drop_prob (float): Probability of dropping an element to zero.
    """
    def __init__(self, drop_prob=None):
        """
        Constructor.

        Args:
            drop_prob (float, optional): Probability of dropping an element to zero. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Applies drop path regularization during training.

        Args:
            x (torch tensor): Input tensor.

        Returns:
            torch tensor: Regularized output tensor.
        """
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Returns an additional string representation containing the drop probability.

        Returns:
            str: String representation of the drop probability.
        """
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module with activation and dropout.

    This module implements a simple MLP architecture with an activation function and dropout.

    Methods:
        forward(x):
            Forward pass through the MLP.

    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        act (torch.nn.Module): Activation function layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        drop (torch.nn.Dropout): Dropout layer.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Constructor.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden fts. Defaults to the same as input features.
            out_features (int, optional): Number of output features. Defaults to the same as input features.
            act_layer (torch.nn.Module, optional): Activation function layer. Defaults to nn.GELU.
            drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch tensor): Input tensor.

        Returns:
            torch tensor: Output tensor from the MLP.
        """
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding for sequence modeling.

    This module implements the Sinusoidal Positional Embedding, which is commonly used in Transformer-based
    models to add positional information to input sequences.

    Methods:
        forward(x):
            Forward pass through the Sinusoidal Positional Embedding.

    Attributes:
        dim (int): Dimension of the positional embeddings.
        M (int): Scaling factor for the sinusoidal embedding.
    """

    def __init__(self, dim=16, M=10000):
        """
        Constructor.

        Args:
            dim (int, optional): Dimension of the positional embeddings. Defaults to 16.
            M (int, optional): Scaling factor for the sinusoidal embedding. Defaults to 10000.
        """
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        """
        Forward pass through the Sinusoidal Positional Embedding.

        Args:
            x (torch tensor): Input tensor containing sequence positions.

        Returns:
            torch tensor: Sinusoidal positional embeddings concatenated with their cosine values.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# BEiTv2 block
class Block(nn.Module):
    """
    Transformer Block for sequence modeling.

    This module implements a Transformer Block, a fundamental building block of the Transformer model
    architecture. It consists of a multi-head self-attention mechanism followed by a multi-layer perceptron.

    Methods:
        forward(xq, xk, xv, attn_mask=None, key_padding_mask=None):
            Forward pass through the Transformer Block.

    Attributes:
        norm1 (nn.Module): Normalization layer for the first attention block.
        attn (nn.MultiheadAttention): Multi-head self-attention mechanism.
        drop_path (DropPath): DropPath module for the residual connection.
        norm2 (nn.Module): Normalization layer for the second attention block.
        mlp (Mlp): Multi-layer perceptron module.
        gamma_1 (nn.Parameter or None): Learnable parameter for gamma_1 if init_values is provided, else None.
        gamma_2 (nn.Parameter or None): Learnable parameter for gamma_2 if init_values is provided, else None.
    """
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
        """
        Constructor.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of hidden dim to input dim for the MLP. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to include bias in the projections. Defaults to False.
            qk_scale (float, optional): Scaling factor for query and key. Defaults to None.
            drop (float, optional): Dropout rate for all layers except self-attention. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate for self-attention. Defaults to 0.0.
            drop_path (float, optional): Dropout rate for the residual connection. Defaults to 0.0.
            init_values (float, optional): Initial values for gamma_1 and gamma_2. Defaults to None.
            act_layer (nn.Module, optional): Activation function layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            **kwargs: Additional keyword arguments.
        """
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
        """
        Forward pass through the Transformer Block.

        Args:
            xq (torch tensor): Query tensor.
            xk (torch tensor): Key tensor.
            xv (torch tensor): Value tensor.
            attn_mask (torch tensor, optional): Mask for attention mechanism. Defaults to None.
            key_padding_mask (torch tensor, optional): Mask for padding in key. Defaults to None.

        Returns:
            torch tensor: Output tensor after applying the Transformer Block.
        """
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
    """
    Temporal Mixer module for sequence modeling.
    It uses a transformer architecture.

    Methods:
        forward(x, frame_idx=-1):
            Forward pass through the Temporal Mixer module.

    Attributes:
        seq_enc (SinusoidalPosEmb): Sinusoidal positional encoding module.
        blocks (nn.ModuleList): List of mixer blocks.
    """
    def __init__(self, n, head_size=32, num_layers=2, **kwargs):
        """
        Constructor.

        Args:
            n (int): Number of elements in the sequence.
            head_size (int, optional): Size of the attention head. Defaults to 32.
            num_layers (int, optional): Number of layers in the mixer. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.seq_enc = SinusoidalPosEmb(n)
        self.blocks = nn.ModuleList([Block(n, n // 64) for i in range(num_layers)])

    def forward(self, x, frame_idx=-1):
        """
        Forward pass through the Temporal Mixer module.

        Args:
            x (torch tensor): Input tensor.
            frame_idx (int, optional): Index of the frame to process. Defaults to -1.

        Returns:
            torch tensor: Output tensor after applying Temporal Mixer.
        """
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
