import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import utils


# import accimage
# try:
#     return accimage.Image(path)
# except IOError:
def open_image(img_path):
    return Image.open(img_path).convert("RGB")


class ImageDataset(Dataset):

    def __init__(self, img_dir, label_path, n_classes, img_format="jpg", transform=None):
        self.img_dir = img_dir
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.n_classes = n_classes
        self.transform = transform
        self.img_format = img_format

    def __getitem__(self, i):
        sample = self.labels[i]
        img_path = os.path.join(self.img_dir, sample['image_id'] + "." + self.img_format)
        img = open_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        y = utils.normalize_labels(sample['label'])
        return img, y

    def __len__(self):
        return len(self.labels)
