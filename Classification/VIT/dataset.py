

import torch.utils.data as data
from PIL import Image

class ButterfliesDataset(data.Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        im = Image.open(self.image_files[index]) 
        label = self.labels[index]

        if self.transforms:
            im = self.transforms(im)

        return im, label




