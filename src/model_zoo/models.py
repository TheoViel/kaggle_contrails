# TODO : DOC

import torch
import torch.nn as nn
import segmentation_models_pytorch

from model_zoo.unet import Unet
from model_zoo.transformer import Tmixer


DECODERS = [
    "Unet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
    "UnetPlusPlus",
]


def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    encoder_weights="imagenet",
    pretrained=True,
    n_channels=3,
    pretrained_weights=None,
    reduce_stride=False,
    use_pixel_shuffle=False,
    use_hypercolumns=False,
    center="none",
    use_cls=False,
    frames=4,
    use_lstm=False,
    bidirectional=False,
    use_cnn=False,
    kernel_size=5,
    use_transfo=False,
    verbose=0,
):
    """
    Loads a segmentation architecture.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
        encoder_weights (str, optional): Pretrained weights. Defaults to "imagenet".

    Returns:
        torch model: Segmentation model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"

    if decoder_name == "Unet":
        model = Unet(
            encoder_name="tu-" + encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=n_channels,
            classes=num_classes,
            use_pixel_shuffle=use_pixel_shuffle,
            use_hypercolumns=use_hypercolumns,
            center=center,
            use_lstm=use_lstm,
            aux_params={"dropout": 0.2, "classes": num_classes} if use_cls else None,
        )
    elif decoder_name == "FPN" and "nextvit" in encoder_name:
        from mmseg.models import build_segmentor
        from mmcv.utils import Config
        import sys
        sys.path.append('../nextvit/segmentation/')
        # from nextvit import nextvit_small

        cfg = Config.fromfile(f"../nextvit/segmentation/configs/fpn_512_{encoder_name}_80k.py")
        model = build_segmentor(cfg.model)
        if pretrained:
            state_dict = torch.load(f'../input/fpn_80k_{encoder_name}_1n1k6m_pretrained.pth')["state_dict"]
            del state_dict['decode_head.conv_seg.weight'], state_dict['decode_head.conv_seg.bias']
            model.load_state_dict(state_dict, strict=False)
        model.backbone.stem[0].conv.stride = (1, 1)
        model.backbone.stem[3].conv.stride = (1, 1)
        model = nn.Sequential(model.backbone, model.neck, model.decode_head)

    else:
        decoder = getattr(segmentation_models_pytorch, decoder_name)
        model = decoder(
            encoder_name="tu-" + encoder_name,
            encoder_weights=encoder_weights if pretrained else None,
            in_channels=n_channels,
            classes=num_classes,
            aux_params={"dropout": 0.2, "classes": 1} if use_cls else None,
            upsampling=int(4 // 2 ** reduce_stride),
        )

    model.num_classes = num_classes

    model = SegWrapper(
        model,
        use_cls,
        frames=frames,
        use_lstm=use_lstm,
        bidirectional=bidirectional,
        use_cnn=use_cnn,
        kernel_size=kernel_size,
        use_transfo=use_transfo,
    )
    model.reduce_stride(encoder_name, decoder_name, reduce_stride)

    if pretrained_weights is not None:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
        del (
            state_dict['model.segmentation_head.0.weight'],
            state_dict['model.segmentation_head.0.bias'],
        )
        model.load_state_dict(state_dict, strict=False)

    return model


class SegWrapper(nn.Module):
    def __init__(
        self,
        model,
        use_cls=False,
        frames=4,
        use_lstm=False,
        bidirectional=False,
        use_cnn=False,
        kernel_size=3,
        use_transfo=False,
    ):
        """
        Constructor.
        TODO

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.model = model
        self.num_classes = model.num_classes
        self.use_cls = use_cls
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.use_transfo = use_transfo
        self.frames = frames

        if use_lstm or use_cnn:
            assert isinstance(frames, (tuple, list)), "frames must be tuple or int"
            assert (len(frames) > 1) and (4 in frames), "several frames expected, 4 has to be included"
        if use_transfo:
            assert not use_lstm and not use_cnn, "Cannot use transformer and lstm/cnn"

        if self.use_lstm:
            self.lstm = nn.LSTM(
                model.encoder.out_channels[-1],
                model.encoder.out_channels[-1] // (1 + bidirectional),
                batch_first=True,
                bidirectional=bidirectional
            )
        if self.use_cnn:
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)

            self.cnn = nn.Sequential(
                nn.Conv3d(
                    model.encoder.out_channels[-1],
                    model.encoder.out_channels[-1],
                    kernel_size=kernel_size,
                    padding=(kernel_size[0] // 2 if use_lstm else 0, kernel_size[1] // 2, kernel_size[2] // 2)
                ),
                nn.BatchNorm3d(model.encoder.out_channels[-1]),
                nn.ReLU(),
            )

        if self.use_transfo:
            self.transfo = Tmixer(model.encoder.out_channels[-1],)

    def reduce_stride(self, encoder_name, decoder_name="Unet", reduce_stride=0):
        if "nextvit" in encoder_name:
            return

        if reduce_stride == 0:
            return

        if "nfnet" in encoder_name:
            self.model.encoder.model.stem_conv1.stride = (1, 1)
        elif "efficientnet" in encoder_name:
            self.model.encoder.model.conv_stem.stride = (1, 1)
        elif "resnet" in encoder_name or "resnext" in encoder_name:
            self.model.encoder.model.conv1.stride = (1, 1)
        elif "convnext" in encoder_name:
            self.model.encoder.stem[0].stride = (2, 2)
            self.model.encoder.stem[0].padding = (1, 1)
        else:
            raise NotImplementedError

        if decoder_name == "Unet":
            if len(self.model.decoder.blocks) >= 5:
                self.model.decoder.blocks[4].upscale = False
                self.model.decoder.blocks[4].pixel_shuffle = nn.Identity()

        if reduce_stride >= 2:
            if "efficientnetv2" in encoder_name:
                self.model.encoder.model.blocks[1][0].conv_exp.stride = (1, 1)
            elif "efficientnet" in encoder_name:
                self.model.encoder.model.blocks[1][0].conv_dw.stride = (1, 1)
            elif "convnext" in encoder_name:
                self.model.encoder.stem[0].stride = (1, 1)
            if decoder_name == "Unet":
                if len(self.model.decoder.blocks) >= 4:
                    self.model.decoder.blocks[3].upscale = False
                    self.model.decoder.blocks[3].pixel_shuffle = nn.Identity()

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        bs = x.size(0)
        if len(x.size()) == 5:
            bs, n_frames, c, h, w = x.size()
            x = x.view(bs * n_frames, c, h, w)
        else:
            assert len(x.size()) == 4, "Length of input size not supported"
            bs, c, h, w = x.size()
            n_frames = 1

        features = self.model.encoder(x)

        if self.use_lstm or self.use_cnn or self.use_transfo:
            assert n_frames > 1, "Only one frame, cannot use LSTM / CNN"
            features_ = []
            frame_idx = self.frames.index(4)

            for i, ft in enumerate(features):
                # print(ft.size())

                if i != len(features) - 1:  # not last layer
                    ft = ft.view(bs, n_frames, ft.size(1), ft.size(2), ft.size(3))[:, frame_idx]
                    features_.append(ft)
                    continue

                _, n_fts, h, w = ft.size()
                ft = ft.view(bs, n_frames, n_fts, h, w)

                if self.use_transfo:
                    ft = self.transfo(ft, frame_idx=frame_idx)

                if self.use_cnn:
                    ft = ft.permute(0, 2, 1, 3, 4).contiguous()  # bs x n_fts x n_frames h x w
                    ft = self.cnn(ft)  # bs x n_fts x h x w

                if self.use_lstm:
                    ft = ft.permute(0, 3, 4, 2, 1).contiguous()  # bs x h x w x n_frames x n_fts
                    ft = ft.view(bs * h * w, n_frames, n_fts)

                    ft = self.lstm(ft)[0][:, frame_idx]  # bs x h x w x n_fts

                    ft = ft.view(bs, h, w, n_fts).permute(0, 3, 1, 2)  # bs x n_fts x h x w

                features_.append(ft.view(bs, n_fts, h, w))

            features = features_

        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
        else:
            labels = torch.zeros(bs, 1).to(x.device)

        return masks, labels
