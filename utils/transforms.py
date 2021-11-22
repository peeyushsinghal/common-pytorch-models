import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def apply_basic_transforms(mean,std_dev):
    train_transforms = A.Compose([
        A.Normalize(mean=mean, std=std_dev, always_apply=True),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Normalize(mean=mean, std=std_dev, always_apply=True),
        ToTensorV2(),
    ])

    return lambda img: train_transforms(image=np.array(img))["image"], lambda img: test_transforms(image=np.array(img))["image"]

def apply_transforms_resnet(mean,std_dev):
    train_transforms = A.Compose([

        A.Sequential(
                [
                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                    A.RandomCrop(width=32, height=32, p=1),# Random Crop
                ],
                p=0.5,
            ),

        A.HorizontalFlip(p=0.2),
        A.Rotate(limit=5, p=0.2),# Rotate

        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.2),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=tuple((x * 255.0 for x in mean)), p=0.2,
        ),# Cutout
        # A.ToGray(p=0.15),
        A.Normalize(mean=mean, std=std_dev, always_apply=True),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Normalize(mean=mean, std=std_dev, always_apply=True),
        ToTensorV2(),
    ])

    return lambda img: train_transforms(image=np.array(img))["image"], lambda img: test_transforms(image=np.array(img))["image"]
