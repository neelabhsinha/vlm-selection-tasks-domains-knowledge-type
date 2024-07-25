from tqdm import tqdm

from src.data.experimental_dataset import ExperimentalDataset


def generate_image_descriptors(split='test'):
    dataset = ExperimentalDataset(get_images=True, split=split)
    for idx in tqdm(range(len(dataset)), total=len(dataset), desc='Generating image descriptors'):
        image = dataset[idx]['image']

