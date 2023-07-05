import torch
import torch.nn as nn
import segmentation_models_pytorch

from model_zoo.unet import Unet

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
            aux_params={"dropout": 0.2, "classes": num_classes} if use_cls else None,
        )
    elif decoder_name == "FPN" and "nextvit" in encoder_name:
        from mmseg.models import build_segmentor
        from mmcv.utils import Config
        import sys
        sys.path.append('../nextvit/segmentation/')
        from nextvit import nextvit_small
        
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
            aux_params={"dropout": 0.2, "classes": num_classes} if use_cls else None,
            upsampling=int(4 // 2 ** reduce_stride),
        )

    if pretrained_weights is not None:
        raise NotImplementedError

    model.num_classes = num_classes
        
    model = SegWrapper(model, use_cls)
    model.reduce_stride(encoder_name, decoder_name, reduce_stride)

    return model


class SegWrapper(nn.Module):
    def __init__(
        self,
        model,
        use_cls=False,
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
        if self.use_cls:
            return self.model(x)
        else:
            return self.model(x), torch.zeros(x.size(0), self.num_classes).to(x.device)
