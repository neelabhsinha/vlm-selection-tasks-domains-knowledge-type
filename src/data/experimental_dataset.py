import os
from PIL import Image
from torch.utils.data import Dataset, Sampler
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
        self.image_descriptors_dir = os.path.join(experimental_set, split, 'image_descriptors')
        self.instance_classifications_dir = os.path.join(experimental_set, split, 'task_instance_classifications')
        self.split = split
        self.get_images = get_images
        self.get_image_descriptors = get_image_descriptors
        self.get_classification = get_classification
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image = None
        image_descriptors = None
        classification = None
        data = read_json(os.path.join(self.data_dir, self.data_files[idx]))
        if self.get_images:
            image = Image.open(data['image_path'])
            if image.mode != 'RGB':
                image = image.convert('RGB')
        if self.get_image_descriptors:
            image_descriptors = read_json(os.path.join(self.image_descriptors_dir, self.data_files[idx]))
        if self.get_classification:
            classification = read_json(os.path.join(self.instance_classifications_dir, self.data_files[idx]))
        sample = {
            'data': data,
            'image': image,
            'image_descriptors': image_descriptors,
            'classification': classification
        }
        return sample


def collate_function(batch):
    questions, images, labels, task_types, domains, knowledge_types, image_paths, datasets, keys = [], [], [], [], [], [], [], [], []
    for instance in batch:
        questions.append(instance['data']['question'])
        images.append(instance['image'])
        labels.append(instance['data']['label'])
        task_types.append(instance['classification']['task_type'])
        domains.append(instance['classification']['application_domain'])
        knowledge_types.append(instance['classification']['knowledge_type'])
        image_paths.append(instance['data']['image_path'])
        datasets.append(instance['data']['dataset'])
        keys.append(instance['data']['key'])
    return questions, images, labels, task_types, domains, knowledge_types, image_paths, datasets, keys

