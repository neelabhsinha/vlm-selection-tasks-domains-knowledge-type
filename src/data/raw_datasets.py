from datasets import load_dataset
from const import cache_dir


def get_dataset(dataset_name):
    dataset_full_name = f'HuggingFaceM4/{dataset_name}'
    print('Loading dataset', dataset_name, 'from HuggingFace source', dataset_full_name)
    dataset = load_dataset(dataset_full_name, cache_dir=cache_dir)
    return dataset