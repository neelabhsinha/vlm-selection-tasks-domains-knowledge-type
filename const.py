project_name = 'vlm-evaluation-for-task-types-and-domains'
project_root = '.'

cache_dir = f'{project_root}/cache'
experimental_set = f'{project_root}/experimental_dataset'
results_dir = f'{project_root}/results'  # results directory

datasets = ['ChartQA', 'DocumentVQA', 'OK-VQA', 'A-OKVQA', 'COCO', 'VQAv2']
tasks = ['collect_experimental_data', 'generate_image_descriptors', 'generate_instance_classifications', 'execute',
         'generate_quantitative_results']
supported_models = ['paligemma-3b-mix-224', 'paligemma-3b-pt-224', 'paligemma-3b-pt-448', 'paligemma-3b-mix-448', 'paligemma-3b-pt-896', 
                    'llava-v1.6-34b-hf', 'llava-v1.6-mistral-7b-hf', 'llava-v1.6-vicuna-7b-hf', 'llava-v1.6-vicuna-13b-hf',
                    'gpt-4o', 'gpt-4o-mini', 
                    'gemini-1.5-flash', 'gemini-1.5-pro', 
                    'cogvlm2-llama3-chat-19B',
                    'InternVL2-1B','InternVL2-2B', 'InternVL2-4B', 'InternVL2-8B', 'InternVL2-26B', 'InternVL2-40B']

dataset_to_text_type_mapping = {
    'ChartQA': 'Chart Understanding',
    'DocumentVQA': 'Document Understanding',
    'OK-VQA': 'Knowledge-based Visual Question-Answering',
    'A-OKVQA': 'Knowledge-based Visual Question-Answering',
    'COCO': 'Image Captioning',
    'VQAv2': 'Visual Question-Answering'
}

valid_application_domains = categories = [
    "Anthropology",
    "Books",
    "Computer Science",
    "Economics",
    "Fiction",
    "Formal logic",
    "Government and Politics",
    "History",
    "Justice",
    "Knowledge Base",
    "Law",
    "Linguistics",
    "Movies",
    "Mathematics",
    "Nature",
    "News",
    "Nutrition and Food",
    "Professions",
    "Public Places",
    "Reviews",
    "Science",
    "Social Media",
    "Sports"
]

valid_knowledge_types = [
    "Commonsense Knowledge",
    "Visual Knowledge",
    "Cultural Knowledge",
    "Temporal Knowledge",
    "Geographical Knowledge",
    "Social Knowledge",
    "Scientific Knowledge",
    "Technical Knowledge",
    "Mathematical Knowledge",
    "Literary Knowledge",
    "Other"
]
