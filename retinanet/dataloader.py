from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate



import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image



def collater(data):

    imgs = [s["img"] for s in data]
    annots = [s["annot"] for s in data]
    # scales = [s["scale"] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, : annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {"img": padded_imgs, "annot": annot_padded}


def letterbox(image, expected_size, fill_value=0):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # print(image)
    new_img = np.full((eh, ew, 3), fill_value, dtype=np.float32)
    # fill new image with the resized image and centered it

    offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2

    new_img[offset_y : offset_y + nh, offset_x : offset_x + nw, :] = image.copy()
    return new_img, scale, offset_x, offset_y


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, annots = sample["img"], sample["annot"]
        rsz_img, scale, offset_x, offset_y = letterbox(image, self.size)

        annots[:, :4] *= scale
        annots[:, 0] += offset_x
        annots[:, 1] += offset_y
        annots[:, 2] += offset_x
        annots[:, 3] += offset_y

        return {
            "img": torch.from_numpy(rsz_img),
            "annot": torch.from_numpy(annots),
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

        # rows, cols, cns = image.shape

        # smallest_side = min(rows, cols)

        # # rescale the image so the smallest side is min_side
        # scale = min_side / smallest_side

        # # check if the largest side is now greater than max_side, which can happen
        # # when images have a large aspect ratio
        # largest_side = max(rows, cols)

        # if largest_side * scale > max_side:
        #     scale = max_side / largest_side

        # # resize the image with the computed scale
        # image = skimage.transform.resize(
        #     image, (int(round(rows * scale)), int(round((cols * scale))))
        # )
        # rows, cols, cns = image.shape

        # pad_w = (32 - rows % 32) % 32
        # pad_h = (32 - cols % 32) % 32

        # new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_image[:rows, :cols, :] = image.astype(np.float32)

        # annots[:, :4] *= scale

        # return {
        #     "img": torch.from_numpy(new_image),
        #     "annot": torch.from_numpy(annots),
        #     "scale": scale,
        # }


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample["img"], sample["annot"]
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {"img": image, "annot": annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample["img"], sample["annot"]

        return {
            "img": ((image.astype(np.float32) - self.mean) / self.std),
            "annot": annots,
        }


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]


class ImageDirectory(Dataset):
    def __init__(self, image_dir, ext="jpg"):
        self.images = glob.glob(os.path.join(image_dir, f"*.{ext}"))
        self.transforms = torchvision.transforms.Compose(
            [
                #                   torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        return self.transforms(img), os.path.basename(self.images[idx])

    def get_image(self, idx):
        return np.array(Image.open(self.images[idx]))


def custom_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids


def eval_collate(batch):
    image_ids, images, labels, scales, offset_x, offset_y = [], [], [], [], [], []
    for b in batch:
        instance, img_id = b
        images.append(instance["img"])
        labels.append(instance["annot"])
        scales.append(instance["scale"])
        offset_x.append(instance["offset_x"])
        offset_y.append(instance["offset_y"])
        image_ids.append(img_id)
    return (
        torch.stack(images).permute(0, 3, 1, 2).contiguous(),
        labels,
        scales,
        offset_x,
        offset_y,
        image_ids,
    )
