# type: ignore
import glob
import os

import cv2
import numpy as np
import torch.utils.data as data
from tqdm import tqdm


class ImageNetREC(data.Dataset):
    def __init__(
        self,
        root,
        image_size=128,
        transform=None,
        split="train",
    ):
        self.image_list = []
        self.image_ids = []
        self.transform = transform
        self.image_size = image_size

        if split == "train":
            image_folder = os.path.join(root, "ILSVRC2012_img_train")
        elif split == "val":
            if os.path.basename(root) == "":
                root = os.path.dirname(root)
            image_folder = os.path.join(root, "VAL_SET", "imgs")

        else:
            raise ValueError(f"unknown split type {split}")

        all_synsets_in_imagenet = [
            os.path.basename(p) for p in glob.glob(os.path.join(image_folder, "*"))
        ]
        interest_synsets = all_synsets_in_imagenet
        self.syns2ind = {}
        for index, syns in enumerate(tqdm(interest_synsets)):
            images = glob.glob(os.path.join(image_folder, syns, "*"))
            self.image_list += images
            self.image_ids += [syns for i in range(len(images))]
            self.syns2ind[syns] = index

        # return image_list, image_ids

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # print({index})
        im_path = self.image_list[index]
        id = self.image_ids[index]
        label = self.syns2ind[id]
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.image_size > 1:
            im_small = (
                cv2.resize(im, (self.image_size, self.image_size)) / 255.0
            ).astype(np.float32)
        else:
            im_small = (im / 255.0).astype(np.float32)

        if self.transform is not None:
            im_small = self.transform(im_small)

        # return {"image": im_small, "path": im_path, "class_id": id}
        return im_small, label
