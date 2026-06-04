"""Albumentations: light (default) and hard in-car augmentation."""
import numpy as np
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_light_train_augmentation():
    if A is None:
        raise ImportError('albumentations is required: pip install albumentations')
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=8,
            border_mode=0, p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(8, 24),
                        hole_width_range=(8, 24), fill=0, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_hard_train_augmentation():
    """Simulate in-car motion, lighting, blur, occlusion."""
    if A is None:
        raise ImportError('albumentations is required: pip install albumentations')
    return A.Compose([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.25),
        A.MotionBlur(blur_limit=7, p=0.25),
        A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.08, rotate_limit=12,
            border_mode=0, p=0.5,
        ),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(8, 24),
                        hole_width_range=(8, 24), fill=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_augmentation():
    if A is None:
        raise ImportError('albumentations is required: pip install albumentations')
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class AlbuTransform:
    """PIL image -> augmented tensor."""
    def __init__(self, size, train=True, hard=False):
        self.size = size if isinstance(size, tuple) else (size, size)
        if train:
            self.pipeline = build_hard_train_augmentation() if hard else build_light_train_augmentation()
        else:
            self.pipeline = build_val_augmentation()

    def __call__(self, pil_img):
        img = np.array(pil_img.convert('RGB').resize(self.size, Image.BILINEAR))
        return self.pipeline(image=img)['image']


class AlbuReplayTransform:
    """Same random aug replayed on every frame in a clip (temporal consistency)."""
    def __init__(self, size, train=True, hard=False):
        self.size = size if isinstance(size, tuple) else (size, size)
        if train:
            aug = build_hard_train_augmentation() if hard else build_light_train_augmentation()
        else:
            aug = build_val_augmentation()
        transforms = [t for t in aug.transforms if not isinstance(t, (A.Normalize, ToTensorV2))]
        tail = [t for t in aug.transforms if isinstance(t, (A.Normalize, ToTensorV2))]
        self.replay = A.ReplayCompose(transforms + tail)

    def __call__(self, pil_img):
        img = np.array(pil_img.convert('RGB').resize(self.size, Image.BILINEAR))
        return self.replay(image=img)['image']

    def apply_clip(self, pil_images):
        """Apply one sampled aug pipeline to all frames."""
        tensors = []
        replay_data = None
        for i, pil_img in enumerate(pil_images):
            img = np.array(pil_img.convert('RGB').resize(self.size, Image.BILINEAR))
            if i == 0:
                out = self.replay(image=img)
                replay_data = out['replay']
                tensors.append(out['image'])
            else:
                out = A.ReplayCompose.replay(replay_data, image=img)
                tensors.append(out['image'])
        return tensors
