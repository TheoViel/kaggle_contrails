import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    This module performs atrous convolution with different dilation rates
    to capture multi-scale contextual information.

    Methods:
        forward(x):
            Forward pass through the ASPP module.

        _init_weight():
            Initialize the weights of the module's layers.
    """
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        """
        Constructor.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding applied to the input.
            dilation (int): Dilation rate for the atrous convolution.
        """
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        """
        Forward pass through the ASPP module.

        Args:
            x (torch tensor): Input tensor.

        Returns:
            torch tensor: Output tensor after atrous convolution and batch normalization.
        """
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        """
        Initialize the weights of the module's layers.
        This method is used to initialize the weights of the convolutional and batch normalization layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    This module implements Atrous Spatial Pyramid Pooling,
    which uses atrous convolutions with multiple dilation rates to capture multi-scale contextual information.

    Args:
        inplanes (int, optional): Number of input channels. Defaults to 512.
        mid_c (int, optional): Number of intermediate channels. Defaults to 256.
        dilations (list, optional): Dilation rates for atrous convolutions. Defaults to [1, 6, 12, 18].

    Methods:
        forward(x):
            Forward pass through the ASPP module.

        _init_weight():
            Initialize the weights of the module's layers.
    """
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        """
        Constructor.

        Args:
            inplanes (int, optional): Number of input channels. Defaults to 512.
            mid_c (int, optional): Number of intermediate channels. Defaults to 256.
            dilations (list, optional): Dilation rates for atrous convolutions. Defaults to [1, 6, 12, 18].
        """
        super().__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Forward pass through the ASPP module.

        Args:
            x (torch tensor): Input tensor.

        Returns:
            torch tensor: Output tensor after ASPP operations.
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        """
        Initialize the weights of the module's layers.
        This method is used to initialize the weights of the convolutional and batch normalization layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
