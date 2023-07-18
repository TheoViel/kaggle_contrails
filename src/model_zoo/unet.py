# Adapted from:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/decoder.py
# TODO : Clean, update doc

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
import model_zoo.nextvit as nextvit


class DecoderBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
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
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

#         for ft in features:
#             print(ft.size())
#         print()

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        xs = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

#             print(i, x.size(), skip.size() if skip is not None else None)
            x = decoder_block(x, skip)
            xs.append(x)
#             print(x.size())

#         print(len(xs))
        if self.use_hypercolumns:
            h, w = x.size()[2:]
            x = torch.cat([
                F.upsample_bilinear(xs[-4], size=(h, w)),
                F.upsample_bilinear(xs[-3], size=(h, w)),
                F.upsample_bilinear(xs[-2], size=(h, w)),
                x
            ], 1)

            x = self.hypercolumns_conv(x)
#             print(x.size())

        return x


def convnext_forward_features(self, x):
    fts = [x]
    x = self.stem(x)

    if (x.size(-1) % 8) != 0:
        x = F.pad(x, [0, 1, 0, 1], value=0.0)

    for stage in self.stages:
        x = stage(x)
        fts.append(x)

    return fts


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask.
    Use *concatenation* for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet)
            and other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for decoder convolutions.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head).
            Auxiliary output is build on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

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
        super().__init__()

        if "nextvit" in encoder_name:
            assert in_channels == 3
            self.encoder = getattr(nextvit, encoder_name[3:])(pretrained=encoder_weights)
            encoder_depth = 4
            decoder_channels = decoder_channels[:encoder_depth]
        elif "convnext" in encoder_name:
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
