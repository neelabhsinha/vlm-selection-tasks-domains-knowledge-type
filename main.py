import argparse
import os
from const import tasks

from src.data.dataset_collector import DatasetCollector
from src.utils.image_descriptor_generator import generate_image_descriptors


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
    parser.add_argument("--model_name", nargs='+', default=None,
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

    return parser.parse_args()


if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    if args.task == 'collect_experimental_data':
        dataset_collector = DatasetCollector(split=args.split)
        dataset_collector.process_datasets()
    if args.task == 'generate_image_descriptors':
        generate_image_descriptors(split=args.split, batch_size=args.batch_size)
