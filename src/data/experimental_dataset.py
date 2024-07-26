import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from const import experimental_set
from src.utils.json_utils import read_json


class ExperimentalDatasetSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


class ExperimentalDataset(Dataset):
    def __init__(self, get_images=False, get_image_descriptors=False, get_classification=False,
                 split='test'):
        self.images_dir = os.path.join(experimental_set, split, 'images')
        self.data_dir = os.path.join(experimental_set, split, 'data')
        self.split = split
        self.get_images = get_images
        self.get_image_descriptors = get_image_descriptors
        self.get_classification = get_classification
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = None
        image = None
        image_descriptors = None
        classification = None
        data = read_json(os.path.join(self.data_dir, self.data_files[idx]))
        if self.get_images:
            image = Image.open(data['image_path'])

        sample = {
            'data': data,
            'image': image,
            'image_descriptors': image_descriptors,
            'classification': classification
        }
        return sample
