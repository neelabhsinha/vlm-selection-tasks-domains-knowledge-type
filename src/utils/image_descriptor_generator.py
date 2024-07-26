import os

import torch.backends.mps
from tqdm import tqdm

from const import experimental_set
from src.data.experimental_dataset import ExperimentalDataset
from src.model.object_tags_generator import ObjectTagsGenerator
from src.model.vit_gpt2_caption_generator import ImageCaptionGenerator
from src.utils.json_utils import save_dict_to_json


def generate_image_descriptors(split='test', batch_size=1):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    caption_generator = ImageCaptionGenerator(device=device)
    object_tags_generator = ObjectTagsGenerator()
    dataset = ExperimentalDataset(get_images=True, split=split)
    num_samples = len(dataset)
    for start_idx in tqdm(range(0, num_samples, batch_size), total=num_samples // batch_size,
                          desc='Generating image descriptors'):
        end_idx = min(start_idx + batch_size, num_samples)
        images = [dataset[idx]['image'] for idx in range(start_idx, end_idx)]
        captions = caption_generator.get_caption(images)
        object_tags = [object_tags_generator.get_tags(image) for image in images]
        for idx in range(start_idx, end_idx):
            image_descriptors = {
                'dataset': dataset[idx]['data']['dataset'],
                'key': dataset[idx]['data']['key'],
                'caption': captions[idx - start_idx],
                'object_tags': object_tags[idx - start_idx]
            }
            file_name = dataset[idx]['data']['dataset'] + '_' + str(dataset[idx]['data']['key']) + '.json'
            dir_name = f'{experimental_set}/{split}/image_descriptors'
            os.makedirs(dir_name, exist_ok=True)
            save_dict_to_json(os.path.join(dir_name, file_name), image_descriptors)
            break
        break





