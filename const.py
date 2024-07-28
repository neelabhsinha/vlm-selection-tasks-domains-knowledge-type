project_name = 'vlm-evaluation-for-task-types-and-domains'
project_root = '.'

cache_dir = f'{project_root}/cache'
experimental_set = f'{project_root}/experimental_dataset'

datasets = ['ChartQA', 'DocumentVQA', 'OK-VQA', 'A-OKVQA', 'COCO', 'VQAv2']
tasks = ['collect_experimental_data', 'generate_image_descriptors', 'generate_instance_classifications', 'eval',
         'collect_results']

dataset_to_text_type_mapping = {
    'ChartQA': 'Chart Understanding',
    'DocumentVQA': 'Document Understanding',
    'OK-VQA': 'Knowledge-based Visual Question-Answering',
    'A-OKVQA': 'Knowledge-based Visual Question-Answering',
    'COCO': 'Image Captioning',
    'VQAv2': 'Visual Question-Answering'
}
