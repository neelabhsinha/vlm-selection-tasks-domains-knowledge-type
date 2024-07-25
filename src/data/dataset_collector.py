import os
import json
import random
from tqdm import tqdm

from const import datasets, experimental_set
from src.data.raw_datasets import get_dataset


def save_dict_to_json(file_path, dictionary):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the dictionary to JSON: {e}")


class DatasetCollector:
    def __init__(self, split='test', random_seed=42, instance_per_dataset=1145):
        self._datasets = datasets
        self.image_path = f'{experimental_set}/{split}/images'
        self.other_data_path = f'{experimental_set}/{split}/data'
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.other_data_path, exist_ok=True)
        self.split = split
        self.random_seed = random_seed
        self.instance_per_dataset = instance_per_dataset
        random.seed(random_seed)

    def collate_chartqa(self, dataset):
        dataset = dataset[self.split]
        dataset_name = 'ChartQA'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = dataset[key]['label']
            question = dataset[key]['query']
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def collate_documentvqa(self, dataset):
        if self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'validation'
        else:
            split = 'test'
        dataset = dataset[split]
        dataset_name = 'DocumentVQA'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = dataset[key]['answers']
            question = dataset[key]['question']
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def collate_okvqa(self, dataset):
        if self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'validation'
        else:
            split = 'test'
        dataset = dataset[split]
        dataset_name = 'OK-VQA'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = [item['answer'] for item in dataset[key]['answers']]
            question = dataset[key]['question']
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def collate_aokvqa(self, dataset):
        if self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'validation'
        else:
            split = 'test'
        dataset = dataset[split]
        dataset_name = 'A-OKVQA'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = dataset[key]['direct_answers']
            question = dataset[key]['question']
            choices = dataset[key]['choices']
            correct_choice = dataset[key]['choices'][dataset[key]['correct_choice_idx']] \
                if dataset[key]['correct_choice_idx'] is not None else None
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label,
                'choices': choices,
                'correct_choice': correct_choice
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def collate_coco(self, dataset):
        dataset = dataset[self.split]
        dataset_name = 'COCO'
        question = 'Describe this image.'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = [dataset[key]['sentences']['raw']] if isinstance(dataset[key]['sentences'], dict) \
                else [item['raw'] for item in dataset[key]['sentences']]
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def collate_vqav2(self, dataset):
        if self.split == 'train':
            split = 'train'
        elif self.split == 'test':
            split = 'validation'
        else:
            split = 'test'
        dataset = dataset[split]
        dataset_name = 'VQAv2'
        indices = self.get_indices(dataset=dataset)
        for key in (tqdm(indices)):
            image_path = os.path.join(self.image_path, f'{dataset_name}_{key}.png')
            image = dataset[key]['image']
            label = [item['answer'] for item in dataset[key]['answers']]
            question = dataset[key]['question']
            data_json = {
                'dataset': dataset_name,
                'key': key,
                'image_path': image_path,
                'question': question,
                'label': label
            }
            image.save(image_path)
            save_dict_to_json(os.path.join(self.other_data_path, f'{dataset_name}_{key}.json'), data_json)

    def get_indices(self, dataset):
        if len(dataset) <= self.instance_per_dataset:
            return list(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), self.instance_per_dataset)
            indices = sorted(indices)
            return indices

    def process_datasets(self):
        for dataset_name in self._datasets:
            dataset = get_dataset(dataset_name)
            if dataset_name == 'ChartQA':
                self.collate_chartqa(dataset)
            elif dataset_name == 'DocumentVQA':
                self.collate_documentvqa(dataset)
            elif dataset_name == 'OK-VQA':
                self.collate_okvqa(dataset)
            elif dataset_name == 'A-OKVQA':
                self.collate_aokvqa(dataset)
            elif dataset_name == 'COCO':
                self.collate_coco(dataset)
            elif dataset_name == 'VQAv2':
                self.collate_vqav2(dataset)
            else:
                print(f'Dataset {dataset_name} not found.')
