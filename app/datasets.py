import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, 'frames/A') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'frames/B') + '/*.*'))

    def __getitem__(self, index):
        item_a = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        if self.unaligned:
            item_b = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_b = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
