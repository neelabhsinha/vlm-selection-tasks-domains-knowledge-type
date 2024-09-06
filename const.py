project_name = 'vlm-evaluation-for-task-types-and-domains'
project_root = '.'

cache_dir = f'{project_root}/cache'
experimental_set = f'{project_root}/experimental_dataset'
results_dir = f'{project_root}/results'  # results directory
aggregated_results_dir = f'{project_root}/aggregated_results'  # aggregated results directory

datasets = ['ChartQA', 'DocumentVQA', 'OK-VQA', 'A-OKVQA', 'COCO', 'VQAv2']
tasks = ['collect_experimental_data', 'generate_image_descriptors', 'generate_instance_classifications', 'analyze_dataset', 'execute',
         'compute_metrics', 'collect_results']
supported_models = ['paligemma-3b-mix-224', 'paligemma-3b-pt-224', 'paligemma-3b-pt-448', 'paligemma-3b-mix-448', 'paligemma-3b-pt-896', 
                    'llava-v1.6-34b-hf', 'llava-v1.6-mistral-7b-hf', 'llava-v1.6-vicuna-7b-hf', 'llava-v1.6-vicuna-13b-hf',
                    'gpt-4o', 'gpt-4o-mini', 
                    'gemini-1.5-flash', 'gemini-1.5-pro', 
                    'cogvlm2-llama3-chat-19B',
                    'InternVL2-1B','InternVL2-2B', 'InternVL2-4B', 'InternVL2-8B', 'InternVL2-26B', 'InternVL2-40B',
                    'Qwen2-VL-2B-Instruct', 'Qwen2-VL-7B-Instruct']

dataset_to_text_type_mapping = {
    'ChartQA': 'Chart Understanding',
    'DocumentVQA': 'Document Understanding',
    'OK-VQA': 'Knowledge-based Visual Question-Answering',
    'A-OKVQA': 'Knowledge-based Visual Question-Answering',
    'COCO': 'Image Captioning',
    'VQAv2': 'Visual Question-Answering'
}

valid_application_domains = [
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

beautified_model_names = {
    "InternVL2-1B": "InternVL-2-1B",
    "Qwen2-VL-2B-Instruct": "Qwen-2-VL-2B",
    "paligemma-3b-pt-224": "PaliGemma-3B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-v1.6-Mistral-7B",
    "Qwen2-VL-7B-Instruct": "Qwen-2-VL-7B",
    "InternVL2-8B": "InternVL-2-8B",
    "cogvlm2-llama3-chat-19B": "CogVLM-2-Llama-3-19B",
    "gemini-1.5-flash": "Gemini-1.5-Flash",
    "gemini-1.5-pro": "Gemini-1.5-Pro",
    "gpt-4o-mini": "GPT-4o-Mini",
}

distinctive_colors = [
    '#e6194b', '#3cb44b', '#ffc43a', '#4e4d6d', '#4363d8', '#c19d6d',
    '#911eb4', '#a64d79', '#000075', '#ea780c', '#614051', '#808000',
    '#008080', '#9a6324', '#800000', '#808080'
]

