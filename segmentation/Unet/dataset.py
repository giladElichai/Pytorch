import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import torch
import torchvision.transforms as tt

def form_2D_label(mask,class_map):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2])
    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i    
    return label

def rgb_to_mask(img, color_map):
    num_classes = len(color_map)
    shape = img.shape[:2]+(num_classes,)
    out = np.zeros(shape, dtype=np.float64)-1
    for i, cls in enumerate(color_map):
        out[:,:,i] = np.all(np.array(img).reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
    return out#.transpose(2,0,1)

class segmentationDataset(Dataset):
    def __init__(self, image_pairs, class_map, image_size, transform=None):
        self.image_pairs = image_pairs
        self.class_map = class_map
        self.transform = transform
        self.image_size = image_size
        self.resize_transform = tt.Compose([tt.Resize(image_size,0)])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.image_pairs[index]
        image = np.array(self.resize_transform(Image.open(img_path).convert("RGB")))
        mask = np.array(self.resize_transform(Image.open(mask_path).convert("RGB")))

        #mask = form_2D_label(mask, self.class_map)
        mask = rgb_to_mask(mask, self.class_map)
        mask = np.argmax(mask, 2)
        if self.transform is not None:
            out_image, mask = self.transform(image), torch.from_numpy(mask)

        return out_image, mask