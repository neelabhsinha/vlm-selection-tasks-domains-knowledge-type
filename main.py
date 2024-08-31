import argparse
import os
from const import tasks, supported_models
import torch

from src.data.dataset_collector import DatasetCollector
from src.utils.image_descriptor_generator import generate_image_descriptors
from src.utils.task_instance_classifications_generator import generate_task_instance_classifications
from src.utils.execute import execute_vlm
from src.utils.evaluate_results import compute_metric


def configure_huggingface():
    try:
        hf_token = os.getenv('HF_API_KEY')  # Make sure to add HF_API_KEY to environment variables
        # Add it in .bashrc or .zshrc file to access it globally
        os.environ['HF_TOKEN'] = hf_token
    except (TypeError, KeyError):
        print('Not able to set HF token. Please set HF_API_KEY in environment variables.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Project for evaluating different vision language models (LLMs) on various tasks across multiple'
                    ' domains, categories, and types of reasoning.')

    # General execution parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Define the batch size for model training or evaluation. Default is 1.")
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force the re-computation of all evaluation metrics, even if they have been computed'
                             ' previously.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Specify the split of the dataset to use for evaluation. Default is test split.')

    # Task-related parameters
    parser.add_argument("--task", type=str, default='eval', choices=tasks,
                        help="Specify the task to perform. Options are based on predefined tasks in the 'tasks'"
                             " module.")
    parser.add_argument("--model_name", nargs='+', default=None, choices=supported_models,
                        help="List of model names to be used for the evaluation or any other specified task.")

    # Sampling configuration (if none is selected, decoder defaults to greedy sampling)
    parser.add_argument('--do_sample', action='store_true',
                        help='Activate sampling mode during model output generation to introduce variability.')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Limit the sample to the top k most likely tokens. Used to control the randomness of'
                             ' output predictions.')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Set the cumulative probability cutoff for sampling. Tokens with cumulative probabilities'
                             ' up to this threshold will be considered.')

    # Filtering and selection options (filtering entities are taken from const.py file)
    parser.add_argument('--instance_per_task', type=int, default=50000,
                        help='Set the maximum number of instances per task to process. Default is 50000.')
    parser.add_argument('--filter_domains', action='store_true',
                        help='Enable this option to apply a domain-specific filter during result aggregation, using'
                             ' predefined lists in constants.')
    parser.add_argument('--filter_categories', action='store_true',
                        help='Apply category-specific filters during result aggregation, based on predefined'
                             ' lists in constants.')
    parser.add_argument('--filter_reasoning', action='store_true',
                        help='Activate this to apply reasoning-specific filters during the results aggregation process,'
                             ' according to predefined lists.')
    
    parser.add_argument('--metric', type=str, default='bert_score', help='Specify the metric to compute. Default is bert_score.')
    
    parser.add_argument('--results_folder', type=str, default=None, help='Specify the results folder to compute metrics for.')

    return parser.parse_args()

def execution_flow():
    model_name = args.model_name[0] if (args.model_name is not None and len(args.model_name) == 1) else args.model_name
    if args.task == 'collect_experimental_data':
        dataset_collector = DatasetCollector(split=args.split)
        dataset_collector.process_datasets()
    if args.task == 'generate_image_descriptors':
        generate_image_descriptors(split=args.split, batch_size=args.batch_size)
    if args.task == 'generate_instance_classifications':
        generate_task_instance_classifications(split=args.split)
    if args.task == 'execute':
        execute_vlm(model_name, args.batch_size, args.do_sample, args.top_k, args.top_p)
    if args.task == 'compute_metrics':
        compute_metric(args.metric, args.force_recompute, args.results_folder)
        

if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    current_device = torch.cuda.current_device()
    if current_device==0:
        execution_flow()
    else:
        print('Terminating process on device', current_device)
    

