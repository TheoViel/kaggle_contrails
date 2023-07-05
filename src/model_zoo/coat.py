import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from src.coat import CoaT, coat_lite_mini, coat_lite_small, coat_lite_medium


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf=None, scale=2, blur=True):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [
            nn.Conv2d(ni, nf * (scale**2), 1),
            LayerNorm2d(nf * (scale**2)),
            nn.GELU(),
            nn.PixelShuffle(scale),
        ]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    LayerNorm2d(out_ch * 2),
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                )
                for in_ch, out_ch in zip(input_channels, output_channels)
            ]
        )

    def forward(self, xs: list, last_layer):
        hcs = [
            F.interpolate(
                c(x), scale_factor=2 ** (len(self.convs) - i), mode="bilinear"
            )
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(nn.Module):
    def __init__(
        self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False, **kwargs
    ):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = LayerNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = nn.Sequential(nn.Conv2d(ni, nf, 3, padding=1), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.GELU())
        self.relu = nn.GELU()

    def forward(self, up_in: torch.Tensor, left_in: torch.Tensor) -> torch.Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class UpBlock(nn.Module):
    def __init__(self, up_in_c: int, nf: int = None, blur: bool = True, **kwargs):
        super().__init__()
        ni = up_in_c // 4
        self.shuf = PixelShuffle_ICNR(up_in_c, ni, blur=blur, **kwargs)
        nf = nf if nf is not None else max(up_in_c // 4, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(ni, ni, 3, padding=1),
            LayerNorm2d(ni) if ni >= 16 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(ni, nf, 1),
        )

    def forward(self, up_in: torch.Tensor) -> torch.Tensor:
        return self.conv(self.shuf(up_in))


def TTA(x, i=0):
    if i % 2 == 1:
        x = x.flip(-1)
    if (i // 2) % 2 == 1:
        x = x.flip(-2)
    if (i // 4) % 2 == 1:
        x = torch.transpose(x, -1, -2)
    return x


def TTAi(x, i=0):
    if (i // 4) % 2 == 1:
        x = torch.transpose(x, -1, -2)
    if i % 2 == 1:
        x = x.flip(-1)
    if (i // 2) % 2 == 1:
        x = x.flip(-2)
    return x


class CoaTFormer(nn.Module):
    def __init__(
        self,
        pre="coat_lite_medium_a750cd63.pth",
        arch="medium",
        num_classes=1,
        ps=0,
        tta=1,
        **kwargs
    ):
        super().__init__()
        in_chans = 3
        if arch == "mini":
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == "small":
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == "medium":
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128, 256, 320, 512]
        else:
            raise Exception("Unknown model")

        if pre is not None:
            sd = torch.load(pre)["model"]
            print(self.enc.load_state_dict(sd, strict=False))

        self.dec4 = UnetBlock(nc[-1], nc[-2], 384)
        self.dec3 = UnetBlock(384, nc[-3], 192)
        self.dec2 = UnetBlock(192, nc[-4], 96)
        self.fpn = FPN([nc[-1], 384, 192], [32] * 3)
        self.drop = nn.Dropout2d(ps)
        # self.final_conv = nn.Conv2d(96+32*3, num_classes, 3, padding=1)
        # self.final_conv = nn.Sequential(UpBlock(96+32*3, 32),
        #                                UpBlock(32, num_classes, blur=True))
        self.final_conv = nn.Sequential(UpBlock(96 + 32 * 3, num_classes, blur=True))
        self.up_result = 1
        self.tta = tta

    def pred(self, x):
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        dec4 = encs[-1]
        dec3 = self.dec4(dec4, encs[-2])
        dec2 = self.dec3(dec3, encs[-3])
        dec1 = self.dec2(dec2, encs[-4])
        x = self.fpn([dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))
        if self.up_result != 0:
            x = F.interpolate(x, scale_factor=self.up_result, mode="bilinear")
        return x

    def forward(self, x):
        if self.training:
            return self.pred(x)
        else:
            return torch.stack(
                [TTAi(self.pred(TTA(x, i)), i) for i in range(self.tta)], 0
            ).mean(0)

    split_layers = lambda model: (
        lambda m: [
            list(m.enc.parameters()),
            list(m.dec4.parameters())
            + list(m.dec3.parameters())
            + list(m.dec2.parameters())
            + list(m.fpn.parameters())
            + list(m.final_conv.parameters()),
        ]
    )(model if not isinstance(model, nn.DataParallel) else model.module)
