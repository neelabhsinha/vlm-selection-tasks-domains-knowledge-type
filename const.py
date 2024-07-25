project_name = 'vlm-evaluation-for-task-types-and-domains'
project_root = '.'

cache_dir = f'{project_root}/cache'
experimental_set = f'{project_root}/experimental_dataset'

datasets = ['ChartQA', 'DocumentVQA', 'OK-VQA', 'A-OKVQA', 'COCO', 'VQAv2']
tasks = ['collect_experimental_data', 'generate_image_descriptors', 'eval', 'collect_results']
