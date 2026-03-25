from .resnet_gn import resnet8, resnet18, resnet50
from .cnn import CIFARNet, EMNISTNet, ImageNet, CIFARNetTaylor
from .uncertaintycnn import CIFARNet_EDL, ImageNet_EDL, EMNISTNet_EDL

model_dict = {
    "cifaredl": CIFARNet_EDL,
    "imagenetedl": ImageNet_EDL,
    "emnistnetedl": EMNISTNet_EDL,
    "cifarnettaylor": CIFARNetTaylor,
    "cifarnet": CIFARNet,
    "emnistnet": EMNISTNet,
    "resnet8": resnet8,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "imagenet": ImageNet
}