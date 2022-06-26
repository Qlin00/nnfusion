# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2022-04-07 06:31:55
# @Last Modified by:   gunjianpan
# @Last Modified time: 2022-04-19 07:05:31

import os
import io
import csv
import zipfile
from zipfile import ZipFile
from contextlib import contextmanager

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from data.zip_reader import ZipReader


class _ImageZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_file, samples, transform=None, target_transform=None):
        self.zip_file = zip_file
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(
            io.BytesIO(ZipReader.read(f"{self.zip_file}@{path}"))
        ).convert("RGB")

        # with self.zip_file.open(path) as f:
        #     sample = Image.open(io.BytesIO(f.read())).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"pixel_values": sample, "labels": target}

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class ImageZipDatasetWrapper(torch.utils.data.Dataset):
    """
    A dataset where images are stored in a zip file:
    <filename>.zip@/<img_1>.JPEG
    <filename>.zip@/<img_2>.JPEG
    <filename>.zip@/<img_3>.JPEG
    ...
    And the class assignments are stored in a TSV file:
    .../<filename>.zip@<img_1>.JPEG    <class_a>
    .../<filename>.zip@<img_2>.JPEG    <class_b>
    .../<filename>.zip@<img_3>.JPEG    <class_c>
    Args:
        zip_path (string): path to zip file
        info_path (string): filename of TSV file with class assignments
        transform (callable, optional): transforms to apply to each image
        target_transform (callable, optional): transforms to apply to each target
    """

    def __init__(
        self,
        zip_path,
        info_path,
        transform=None,
        target_transform=None,
        info_encoding="utf8",
    ):
        if not os.path.exists(zip_path):
            raise RuntimeError("%s does not exist" % zip_path)

        if not os.path.exists(info_path):
            raise RuntimeError("%s does not exist" % info_path)

        self.zip_path = zip_path
        self.info_path = info_path
        self.transform = transform
        self.target_transform = target_transform

        with open(self.info_path, "r", encoding=info_encoding) as f:
            reader = csv.reader(f, delimiter="\t")
            self.samples = sorted(
                ((os.path.basename(row[0]), int(row[1])) for row in reader),
                key=lambda x: (x[1], x[0]),
            )

    def dataset(self):
        res = _ImageZipDataset(
            zip_file=self.zip_path,
            samples=self.samples,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        return res

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Zip Location: {}\n".format(self.zip_path)
        fmt_str += "    Info Location: {}\n".format(self.info_path)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str
