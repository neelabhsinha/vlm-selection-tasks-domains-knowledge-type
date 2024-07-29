import pandas as pd
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from const import cache_dir, results_dir
from src.data.experimental_dataset import ExperimentalDataset, ExperimentalDatasetSampler, raw_collate_function
from src.model.vlm import VisionLanguageModel
from src.utils.results_io_util import write_results
from src.utils.gpu_stats import get_gpu_memory


def evaluate(model_name, batch_size, do_sample=False, top_k=None, top_p=None, checkpoint=None):
    parameters_dict = {'model_name': model_name, 'top_k': top_k, 'top_p': top_p, 'checkpoint': checkpoint}
    print('Parameters -')
    print(str(parameters_dict) + '\n\n')
    dataset = ExperimentalDataset(get_classification=True, get_images=True)
    sampler = ExperimentalDatasetSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=raw_collate_function)
    model_builder = VisionLanguageModel(f'{cache_dir}/{checkpoint}' if checkpoint is not None else model_name)
    model = model_builder.get_model()
    processor = model_builder.get_processor()
    prompt_template = model_builder.get_prompt_template()
    name = checkpoint if checkpoint is not None else ('pretrained--' + model_name.replace('/', '--'))
    name += f'--do_sample-{do_sample}--top_k-{top_k}--top_p-{top_p}'
    results_path = f'{results_dir}/{name}'
    results_df = evaluation_loop(dataloader, processor, prompt_template, model, do_sample, top_k, top_p, model_name)
    write_results(results_df, results_path, parameters_dict)


def evaluation_loop(dataloader, processor, prompt_template, model, do_sample, top_k, top_p, model_name):
    results = {'dataset': [], 'key': [], 'question': [], 'image_path': [], 'response': [], 'label': [],
               'task_type': [], 'domain': [], 'knowledge_type': []}
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'Evaluating Model {model_name}')
    count = 0
    for batch in pbar:  # list of dictionaries
        try:
            questions, images, labels, task_types, domains, knowledge_types, image_paths, datasets, keys = (
                (
                    instance['data']['question'],
                    instance['image'],
                    instance['data']['label'],
                    ', '.join(instance['classification']['task_type']),
                    ', '.join(instance['classification']['application_domain']),
                    ', '.join(instance['classification']['knowledge_type']),
                    instance['data']['image_path'],
                    instance['data']['dataset'],
                    instance['data']['key']
                 )
                for instance in batch
            )
            prompts = [prompt_template.format(question=question) for question in questions]
            device = next(model.parameters()).device
            inputs = processor(prompts, images, padding=True, return_tensors="pt").to(device=device)
            inputs = {key: value.to(dtype=torch.int32) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=processor.tokenizer.eos_token_id)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
            candidate_batch = processor.batch_decode(outputs, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                     skip_special_tokens=True)
            results['dataset'].extend(datasets)
            results['key'].extend(keys)
            results['question'].extend(questions)
            results['image_path'].extend(image_paths)
            results['response'].extend(candidate_batch)
            results['label'].extend(labels)
            results['task_type'].extend(task_types)
            results['domain'].extend(domains)
            results['knowledge_type'].extend(knowledge_types)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('\nCuda Out of Memory Error: Clearing Cache', file=sys.stderr)
        if count % 10 == 0:
            pbar.set_postfix(get_gpu_memory())
        count += 1
    results_df = pd.DataFrame(results)
    return results_df
