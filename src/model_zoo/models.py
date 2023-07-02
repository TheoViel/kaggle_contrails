import torch
import torch.nn as nn
import segmentation_models_pytorch


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

    decoder = getattr(segmentation_models_pytorch, decoder_name)
    
    if pretrained_weights is not None:
        raise NotImplementedError

    if reduce_stride:
        raise NotImplementedError
        
    model = decoder(
        encoder_name="tu-" + encoder_name,
        encoder_weights=encoder_weights if pretrained else None,
        in_channels=n_channels,
        classes=num_classes,
        aux_params={"dropout": 0.2, "classes": num_classes} if use_cls else None
    )
    model.num_classes = num_classes

    return SegWrapper(model, use_cls)


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

    def reduce_stride(self):
        raise NotImplementedError

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
