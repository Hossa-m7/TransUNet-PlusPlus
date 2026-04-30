
import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# =========================
# Data Augmentation
# =========================
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# =========================
# Train Transform
# =========================
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label.long()}

# =========================
# Dataset Loader
# =========================
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size=224, transform=None):
        self.transform = transform
        self.split = split
        self.img_size = img_size
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir
        self.files = os.listdir(base_dir)  # list all npz files

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip()

        # find matching file automatically
        found_file = None
        for f in self.files:
            if name in f:
                found_file = f
                break

        if found_file is None:
            print("File not found for:", name)
            raise FileNotFoundError(name)

        data_path = os.path.join(self.data_dir, found_file)
        data = np.load(data_path)

        image, label = data['image'], data['label']

        # Resize to model input size
        x, y = image.shape
        if x != self.img_size or y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        sample['case_name'] = found_file.replace('.npz', '')

        return sample
 
