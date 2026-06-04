import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import csv

from roi_config import (
    FACE_ROI,
    FLIP_SWAP,
    LEFT_HAND_ROI,
    MIRROR_ROI,
    RIGHT_HAND_ROI,
    WHEEL_ROI,
    MAKEUP_CLASS,
    PHONE_CLASSES,
    active_console_roi,
    active_mirror_roi,
)


def crop_roi_fraction(img, roi_fractions):
    """Crop image using fractional coordinates (x1, y1, x2, y2) relative to W/H."""
    width, height = img.size
    x1, y1, x2, y2 = roi_fractions
    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))

    left = max(0, min(left, width - 2))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 2))
    bottom = max(top + 1, min(bottom, height))

    return img.crop((left, top, right, bottom))


class DistractedDataset(Dataset):
    """
    Triple-ROI loader: person, hand, steering wheel crops only.
    """
    def __init__(self, data_dir, transform_person=None, transform_hand=None, transform_wheel=None,
                 transform=None, transform_face=None, transform_crop=None,
                 split='train', val_split=0.2, seed=42,
                 subject_map_csv=None, subjects=None,
                 person_roi=(0.06, 0.02, 0.56, 0.70),
                 hand_roi=(0.38, 0.42, 0.78, 0.88),
                 wheel_roi=(0.45, 0.28, 0.92, 0.92),
                 face_roi=None, return_path=False):
        self.data_dir = data_dir
        self.transform_person = transform_person or transform or transform_face
        self.transform_hand = transform_hand or transform_crop
        self.transform_wheel = transform_wheel or transform_hand or transform_crop
        self.person_roi = face_roi or person_roi
        self.hand_roi = hand_roi
        self.wheel_roi = wheel_roi
        self.return_path = return_path
        self.samples = []
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        img_to_subject = {}
        if subject_map_csv is not None and os.path.exists(subject_map_csv):
            try:
                with open(subject_map_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_to_subject[row['img']] = row.get('subject')
            except Exception:
                img_to_subject = {}

        all_samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    if not os.path.isfile(img_path):
                        continue
                    label = self.class_to_idx[class_name]
                    if subjects is not None and len(subjects) > 0:
                        subject = img_to_subject.get(img_file)
                        if subject is None or subject not in subjects:
                            continue
                    all_samples.append((img_path, label))

        if subjects is None or len(subjects) == 0:
            random.seed(seed)
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1 - val_split))
            self.samples = all_samples[:split_idx] if split == 'train' else all_samples[split_idx:]
        else:
            self.samples = all_samples

        print(f"[OK] Dataset loaded: {len(self.samples)} images ({split})")
        print(f"[OK] Classes: {len(self.class_names)}, List: {self.class_names}")

    def get_class_counts(self):
        counts = [0] * len(self.class_names)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def _apply_transform(self, img, transform, default_size):
        def pil_to_tensor_and_normalize(pil_img):
            import numpy as _np
            import torch as _torch
            arr = _np.array(pil_img).astype('float32') / 255.0
            if arr.ndim == 2:
                arr = _np.stack([arr, arr, arr], axis=-1)
            arr = _torch.tensor(arr, dtype=_torch.float32).permute(2, 0, 1)
            mean = _torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = _torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (arr - mean) / std

        try:
            return transform(img)
        except TypeError:
            try:
                size = None
                if hasattr(transform, 'transforms'):
                    for t in reversed(transform.transforms):
                        if hasattr(t, 'size'):
                            size = t.size
                            break
                if size is None:
                    size = default_size
                pil_img = img.resize(size, resample=Image.BILINEAR)
                return pil_to_tensor_and_normalize(pil_img)
            except Exception:
                return pil_to_tensor_and_normalize(img)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if (self.transform_person is not None and self.transform_hand is not None
                and self.transform_wheel is not None):
            person = self._apply_transform(
                crop_roi_fraction(img, self.person_roi), self.transform_person, default_size=(256, 256)
            )
            hand = self._apply_transform(
                crop_roi_fraction(img, self.hand_roi), self.transform_hand, default_size=(224, 224)
            )
            wheel = self._apply_transform(
                crop_roi_fraction(img, self.wheel_roi), self.transform_wheel, default_size=(288, 288)
            )
            if self.return_path:
                return (person, hand, wheel), label, img_path
            return (person, hand, wheel), label

        if self.transform_person is not None:
            img = self.transform_person(img)
        return img, label


def load_five_roi_pils(img, face_roi=FACE_ROI, left_roi=LEFT_HAND_ROI, right_roi=RIGHT_HAND_ROI,
                       wheel_roi=WHEEL_ROI, mirror_roi=MIRROR_ROI, opt_v3=False):
    """Return PIL crops for fixed five-ROI layout."""
    if opt_v3:
        mirror_roi = active_mirror_roi(True)
    return (
        crop_roi_fraction(img, face_roi),
        crop_roi_fraction(img, left_roi),
        crop_roi_fraction(img, right_roi),
        crop_roi_fraction(img, wheel_roi),
        crop_roi_fraction(img, mirror_roi),
    )


def load_heuristic_roi_pils(img, opt_v3=False):
    """Extra ROI crops for v3 console/reach heuristics (inference only)."""
    face, left, right, wheel, mirror = load_five_roi_pils(img, opt_v3=opt_v3)
    console = crop_roi_fraction(img, active_console_roi(opt_v3))
    return {
        'face': face,
        'left': left,
        'right': right,
        'wheel': wheel,
        'mirror': mirror,
        'console': console,
    }


def mirror_aux_label(class_name):
    return 1.0 if class_name == MAKEUP_CLASS else 0.0


def phone_aux_label(class_name):
    return 1.0 if class_name in PHONE_CLASSES else 0.0


class FiveROIDataset(DistractedDataset):
    """
    Fixed-camera five-ROI: face, left_hand, right_hand, wheel, mirror (upper-right).
    Supports horizontal flip with left/right class swap (c1<->c3, c2<->c4).
    """
    def __init__(
        self,
        data_dir,
        transform_face=None,
        transform_left=None,
        transform_right=None,
        transform_wheel=None,
        transform_mirror=None,
        split='train',
        val_split=0.2,
        seed=42,
        subject_map_csv=None,
        subjects=None,
        face_roi=FACE_ROI,
        left_hand_roi=LEFT_HAND_ROI,
        right_hand_roi=RIGHT_HAND_ROI,
        wheel_roi=WHEEL_ROI,
        mirror_roi=MIRROR_ROI,
        return_path=False,
        flip_swap_train=True,
    ):
        self.data_dir = data_dir
        self.transform_face = transform_face
        self.transform_left = transform_left
        self.transform_right = transform_right
        self.transform_wheel = transform_wheel
        self.transform_mirror = transform_mirror
        self.face_roi = face_roi
        self.left_hand_roi = left_hand_roi
        self.right_hand_roi = right_hand_roi
        self.wheel_roi = wheel_roi
        self.mirror_roi = mirror_roi
        self.return_path = return_path
        self.flip_swap_train = flip_swap_train and split == 'train'
        # Skip DistractedDataset.__init__ body — build five-ROI sample list
        self.data_dir = data_dir
        self.samples = []
        self.class_names = sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        img_to_subject = {}
        if subject_map_csv is not None and os.path.exists(subject_map_csv):
            try:
                with open(subject_map_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_to_subject[row['img']] = row.get('subject')
            except Exception:
                img_to_subject = {}

        all_samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    if not os.path.isfile(img_path):
                        continue
                    label = self.class_to_idx[class_name]
                    if subjects is not None and len(subjects) > 0:
                        subject = img_to_subject.get(img_file)
                        if subject is None or subject not in subjects:
                            continue
                    all_samples.append((img_path, label, class_name))

        if subjects is None or len(subjects) == 0:
            random.seed(seed)
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1 - val_split))
            self.samples = all_samples[:split_idx] if split == 'train' else all_samples[split_idx:]
        else:
            self.samples = all_samples

        print(f"[OK] Five-ROI dataset: {len(self.samples)} images ({split})")
        print(f"[OK] ROIs: face | L-hand | R-hand | wheel | mirror(UR)")
        print(f"[OK] Classes: {self.class_names}")

    def get_class_counts(self):
        counts = [0] * len(self.class_names)
        for _, label, _ in self.samples:
            counts[label] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.flip_swap_train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            swapped = FLIP_SWAP.get(class_name)
            if swapped is not None and swapped in self.class_to_idx:
                class_name = swapped
                label = self.class_to_idx[class_name]

        face, left, right, wheel, mirror = load_five_roi_pils(
            img, self.face_roi, self.left_hand_roi, self.right_hand_roi,
            self.wheel_roi, self.mirror_roi,
        )

        face_t = self._apply_transform(face, self.transform_face, (256, 256))
        left_t = self._apply_transform(left, self.transform_left, (224, 224))
        right_t = self._apply_transform(right, self.transform_right, (224, 224))
        wheel_t = self._apply_transform(wheel, self.transform_wheel, (288, 288))
        mirror_t = self._apply_transform(mirror, self.transform_mirror, (160, 160))

        aux = {
            'mirror': mirror_aux_label(class_name),
            'phone': phone_aux_label(class_name),
        }

        if self.return_path:
            return (face_t, left_t, right_t, wheel_t, mirror_t), label, img_path, aux
        return (face_t, left_t, right_t, wheel_t, mirror_t), label, aux
