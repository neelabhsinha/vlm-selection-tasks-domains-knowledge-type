import pandas as pd
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from const import results_dir
from src.data.experimental_dataset import ExperimentalDataset, ExperimentalDatasetSampler, collate_function
from src.model.vlm import InternVL2, PaliGemma, LlavaNext, Gemini, GPT4o, CogVLM2
from src.utils.results_io_util import write_results
from src.utils.gpu_stats import get_gpu_memory


def execute_vlm(model_name, batch_size, do_sample=False, top_k=None, top_p=None, checkpoint=None):
    parameters_dict = {'model_name': model_name, 'top_k': top_k, 'top_p': top_p, 'checkpoint': checkpoint}
    print('Parameters -')
    print(str(parameters_dict) + '\n\n')
    dataset = ExperimentalDataset(get_classification=True, get_images=True)
    sampler = ExperimentalDatasetSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_function)
    if 'paligemma' in model_name:
        model = PaliGemma(model_name, do_sample, top_k, top_p, checkpoint)
    elif 'llava' in model_name:
        model = LlavaNext(model_name, do_sample, top_k, top_p, checkpoint)
    elif 'gpt' in model_name:
        model = GPT4o(model_name)
    elif 'gemini' in model_name:
        model = Gemini(model_name)
    elif 'cogvlm2' in model_name:
        model = CogVLM2(model_name, do_sample, top_k, top_p, checkpoint)
    elif 'InternVL2' in model_name:
        model = InternVL2(model_name, do_sample, top_k, top_p, checkpoint)
    else:
        raise ValueError(f'Specified model {model_name} not currently supported.')
    name = checkpoint if checkpoint is not None else ('pretrained--' + model_name.replace('/', '--'))
    name += f'--do_sample-{do_sample}--top_k-{top_k}--top_p-{top_p}'
    results_path = f'{results_dir}/{name}'
    results_df = evaluation_loop(dataloader, model, model_name)
    write_results(results_df, results_path, parameters_dict)


def evaluation_loop(dataloader, model, model_name):
    results = {'dataset': [], 'key': [], 'question': [], 'image_path': [], 'response': [], 'label': [],
               'task_type': [], 'domain': [], 'knowledge_type': []}
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Evaluating Model {model_name}')
    count = 0
    for questions, images, labels, task_types, domains, knowledge_types, image_paths, datasets, keys in pbar:
        try:
            with torch.no_grad():
                response = model(questions, images)
            results['dataset'].extend(datasets)
            results['key'].extend(keys)
            results['question'].extend(questions)
            results['image_path'].extend(image_paths)
            results['response'].extend(response)
            results['label'].extend(labels)
            results['task_type'].extend(task_types)
            results['domain'].extend(domains)
            results['knowledge_type'].extend(knowledge_types)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('\nCuda Out of Memory Error: Clearing Cache', file=sys.stderr)
        if count % 10 == 0:
            pbar.set_postfix(get_gpu_memory())
        count += 1
    results_df = pd.DataFrame(results)
    return results_df
