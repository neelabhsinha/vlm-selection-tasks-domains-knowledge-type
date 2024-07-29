from src.data.experimental_dataset import ExperimentalDataset
from src.model.gpt_3_5_turbo import GPT35Turbo
from const import dataset_to_text_type_mapping, experimental_set, valid_knowledge_types, valid_application_domains
from src.utils.json_utils import save_dict_to_json

import os
from tqdm import tqdm


def generate_task_instance_classifications(split='test'):
    classification_generator = GPT35Turbo()
    dataset = ExperimentalDataset(get_image_descriptors=True, split=split)
    num_samples = len(dataset)
    for idx in tqdm(range(0, num_samples), total=num_samples, desc='Generating task instance classifications'):
        image_descriptors = dataset[idx]['image_descriptors']
        data = dataset[idx]['data']
        question = data['question']
        caption = image_descriptors['caption']
        object_tags = image_descriptors['object_tags']
        object_tags = ", ".join(object_tags)
        classification_type = 'Other'
        if data['dataset'] in dataset_to_text_type_mapping:
            classification_type = dataset_to_text_type_mapping[data['dataset']]
        domains = classification_generator.get_image_classification(question, caption, object_tags,
                                                                    'application_domain')
        domains = domains['application_domain'] if 'application_domain' in domains else ['Other']
        domains = filter_elements(domains, valid_application_domains)
        knowledge_types = classification_generator.get_image_classification(question, caption, object_tags,
                                                                            'knowledge_type')
        knowledge_types = knowledge_types['knowledge_type'] if 'knowledge_type' in knowledge_types else ['Other']
        knowledge_types = filter_elements(knowledge_types, valid_knowledge_types)
        classifications = {
            'dataset': data['dataset'],
            'key': data['key'],
            'task_type': classification_type,
            'application_domain': domains,
            'knowledge_type': knowledge_types
        }
        file_name = dataset[idx]['data']['dataset'] + '_' + str(dataset[idx]['data']['key']) + '.json'
        dir_name = f'{experimental_set}/{split}/task_instance_classifications'
        os.makedirs(dir_name, exist_ok=True)
        save_dict_to_json(os.path.join(dir_name, file_name), classifications)


def filter_elements(input_list, categories):
    result = [element for element in input_list if element in categories]
    if len(result) == 0:
        result.append('Other')
    return result
