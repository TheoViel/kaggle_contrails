# Adapted from:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/decoder.py

import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from fastai.layers import PixelShuffle_ICNR

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from model_zoo.aspp import ASPP


class DecoderBlock(nn.Module):
    """
    Decoder block module for U-Net style architectures.

    This module represents a decoder block used in U-Net style architectures for semantic segmentation.
    It consists of convolutional layers and optional attention mechanisms for feature refinement.
    The block can perform upscaling of the input tensor.

    Methods:
        forward(x, skip=None, upscale=True):
            Applies the decoder block to the input tensor.

    Attributes:
        upscale (bool): Whether to perform upscaling of the input tensor.
        use_pixel_shuffle (bool): Whether to use pixel shuffle for upscaling.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        upscale=True,
        use_pixel_shuffle=False,
    ):
        """
        Constructor.

        Args:
            in_channels (int): Number of input channels.
            skip_channels (int): Number of skip connection channels.
            out_channels (int): Number of output channels.
            use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
            attention_type (str, optional): Type of attention mechanism to use. Defaults to None.
            upscale (bool, optional): Whether to perform upscaling of the input tensor. Defaults to True.
            use_pixel_shuffle (bool, optional): Whether to use pixel shuffle for upscaling. Defaults to False.
        """
        super().__init__()
        self.upscale = upscale
        self.use_pixel_shuffle = use_pixel_shuffle

        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

        if self.use_pixel_shuffle:
            self.pixel_shuffle = PixelShuffle_ICNR(in_channels, scale=2)

    def forward(self, x, skip=None, upscale=True):
        """
        Applies the decoder block to the input tensor.

        Args:
            x (torch tensor): Input tensor.
            skip (torch tensor, optional): Skip connection tensor. Defaults to None.
            upscale (bool, optional): Whether to perform upscaling of the input tensor. Defaults to True.

        Returns:
            torch tensor: Output tensor.
        """
        if self.upscale and upscale:
            if self.use_pixel_shuffle:
                x = self.pixel_shuffle(x)
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    """
    Center block module for U-Net style architectures.

    This module represents a center block used in U-Net style architectures for semantic segmentation.
    It consists of consecutive convolutional layers.

    Attributes:
        Sequential: Inherits from nn.Sequential and contains convolutional layers.
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        """
        Constructor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """
    Decoder module for U-Net style architectures.

    This module represents the decoder part of a U-Net style architecture used for semantic segmentation.
    It combines skip connections from the encoder with the main pathway
    to produce the final segmentation output.

    Attributes:
        center (nn.Module): The center block module.
        blocks (nn.ModuleList): List of decoder blocks.
        hypercolumns_conv (nn.Sequential): Convolutional layers for hypercolumns if enabled.
        use_hypercolumns (bool): Whether hypercolumns are used.
    """
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        use_pixel_shuffle=False,
        use_hypercolumns=False,
    ):
        """
        Constructor.

        Args:
            encoder_channels (list): List of channel dimensions of the encoder's output feature maps.
            decoder_channels (list): List of channel dimensions for each block in the decoder.
            n_blocks (int): Number of blocks in the decoder. Defaults to 5.
            use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
            attention_type (str, optional): Type of attention mechanism to use. Defaults to None.
            center (str, optional): Center block. Value can be 'std', 'aspp', or None. Defaults to None.
            use_pixel_shuffle (bool, optional): Whether to use pixel shuffle for upscaling. Defaults to False.
            use_hypercolumns (bool, optional): Whether to use hypercolumns. Defaults to False.
        """
        super().__init__()

        self.use_hypercolumns = use_hypercolumns

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center == "std":
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        elif center == "aspp":
            self.center = ASPP(head_channels, head_channels)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_batchnorm=use_batchnorm, attention_type=attention_type, use_pixel_shuffle=use_pixel_shuffle
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if use_hypercolumns:
            hc_conv_size = np.sum(decoder_channels[-4:])
            self.hypercolumns_conv = nn.Sequential(
                nn.Conv2d(hc_conv_size, hc_conv_size, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(hc_conv_size),
                nn.Conv2d(hc_conv_size, hc_conv_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(hc_conv_size),
            )

    def forward(self, *features):
        """
        Forward pass of the decoder module.

        Args:
            *features (list of torch.Tensor): Encoder's output feature maps, where each tensor corresponds
                to a different resolution level.

        Returns:
            torch.Tensor: Segmentation output tensor.
        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        xs = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

            x = decoder_block(x, skip)
            xs.append(x)

        if self.use_hypercolumns:
            h, w = x.size()[2:]
            x = torch.cat([
                F.upsample_bilinear(xs[-4], size=(h, w)),
                F.upsample_bilinear(xs[-3], size=(h, w)),
                F.upsample_bilinear(xs[-2], size=(h, w)),
                x
            ], 1)

            x = self.hypercolumns_conv(x)

        return x


def convnext_forward_features(self, x):
    """
    Forward pass through the ConvNeXt backbone to extract features.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        list of torch.Tensor: List of feature maps from different stages of the ConvNeXt backbone.
    """
    fts = [x]
    x = self.stem(x)

    if (x.size(-1) % 8) != 0:
        x = F.pad(x, [0, 1, 0, 1], value=0.0)

    for stage in self.stages:
        x = stage(x)
        fts.append(x)

    return fts


class Unet(SegmentationModel):
    """
    UNet-like architecture for image segmentation.

    Attributes:
        encoder: The encoder architecture.
        decoder: The decoder architecture.
        segmentation_head: Head for segmentation output.
        classification_head: Head for auxiliary classification output.
        name (str): Model name.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        use_pixel_shuffle=False,
        use_hypercolumns=False,
        use_lstm=False,
        center="none",
    ):
        """
        Constructor.

        Args:
            encoder_name (str): Name of the encoder architecture.
            encoder_depth (int): Depth of the encoder.
            encoder_weights (str, optional): Pretrained weights for the encoder. Default is "imagenet".
            decoder_use_batchnorm (bool): Whether to use batch normalization in the decoder. Default is True.
            decoder_channels (List[int]): Number of channels for each decoder block.
            decoder_attention_type (str, optional): Attention mechanism type in the decoder. Default is None.
            in_channels (int): Number of input channels. Default is 3.
            classes (int): Number of output classes. Default is 1.
            activation (str or callable, optional): Activation function to use. Default is None.
            aux_params (dict, optional): Auxiliary classification head parameters. Default is None.
            use_pixel_shuffle (bool): Whether to use pixel shuffle in the decoder blocks. Default is False.
            use_hypercolumns (bool): Whether to use hypercolumns for feature fusion. Default is False.
            use_lstm (bool): Whether to use LSTM for temporal feature aggregation. Default is False.
            center (str): Type of center block. Default is "none".
        """
        super().__init__()

        if "convnext" in encoder_name:
            self.encoder = timm.create_model(encoder_name[3:] + ".fcmae", pretrained=encoder_weights)
            self.encoder.head = nn.Identity()
            self.encoder.forward = lambda x: convnext_forward_features(self.encoder, x)
            if "tiny" in encoder_name:
                self.encoder.out_channels = [3, 96, 192, 384, 768]
            elif "base" in encoder_name:
                self.encoder.out_channels = [3, 128, 256, 512, 1024]
            elif "nano" in encoder_name:
                self.encoder.out_channels = [3, 80, 160, 320, 640]
            else:
                print('Check encoder channels !!')
                self.encoder.out_channels = [3, 128, 256, 512, 1024]
            self.encoder.output_stride = 32
            encoder_depth = 4
            decoder_channels = decoder_channels[:encoder_depth]
        else:
            self.encoder = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=decoder_attention_type,
            use_pixel_shuffle=use_pixel_shuffle,
            use_hypercolumns=use_hypercolumns,
        )

        if use_hypercolumns:
            decoder_out_channels = np.sum(decoder_channels[-4:])
        else:
            decoder_out_channels = decoder_channels[-1]

        self.decoder_out_channels = decoder_out_channels

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_out_channels,  # * (1 + use_lstm),
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
