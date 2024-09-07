# Guiding Vision-Language Model Selection for Visual Question-Answering Across Tasks, Domains, and Knowledge Types

## Overview

This repository contains the code for evaluating various Vision-Language Models (VLMs) on multiple tasks across different domains, categories, and types of reasoning. The evaluation process supports multiple models and metrics, with pre-built tasks to handle data collection, image descriptor generation, model execution, and performance evaluation.

## Table of Contents

1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Usage](#usage)
   - [Available Tasks](#available-tasks)
   - [Supported Models](#supported-models)
   - [Metrics](#metrics)
4. [Configurations](#configurations)
5. [How to Run](#how-to-run)
6. [File Structure](#file-structure)

## Installation

To use this repository, clone it and install the required dependencies.

### Step 1: Install Required Python Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Keys
You'll need a Hugging Face API, Gemini API and OpenAI APO token to download and execute models.

```bash
export HF_API_KEY='your-huggingface-token'
export OPENAI_API_KEY='your-openai-token'
export GOOGLE_API_KEY='your-google-token'
```

To run Azure Computer Vision API for Object tags generation, you need to set up the following environment variables:

```bash
export AZURE_COMPUTER_VISION_API_KEY='your-azure-computer-vision-token'
export AZURE_COMPUTER_VISION_ENDPOINT='your-azure-computer-vision-endpoint'
```

We provide our generated object tags, so this is not necessary.

## Usage

The script provides a versatile interface to run evaluations on different vision-language models using pre-defined tasks and metrics.

### Available Tasks
The following tasks are supported:

- `collect_experimental_data` - Collect experimental datasets for model evaluation.
- `generate_image_descriptors` - Generate image descriptors for the dataset.
- `generate_instance_classifications` - Generate task instance classifications.
- `analyze_dataset` - Perform a detailed analysis of the dataset.
- `execute` - Execute a Vision-Language Model (VLM) on the dataset.
- `compute_metrics` - Compute evaluation metrics such as GOEval and BERTScore.
- `collect_results` - Collect and aggregate evaluation results.

### Supported Models

The following models are supported in this project:

- `paligemma-3b-mix-224`
- `paligemma-3b-pt-224`
- `paligemma-3b-pt-448`
- `paligemma-3b-mix-448`
- `paligemma-3b-pt-896`
- `llava-v1.6-34b-hf`
- `llava-v1.6-mistral-7b-hf`
- `llava-v1.6-vicuna-7b-hf`
- `llava-v1.6-vicuna-13b-hf`
- `gpt-4o`
- `gpt-4o-mini`
- `gemini-1.5-flash`
- `gemini-1.5-pro`
- `cogvlm2-llama3-chat-19B`
- `InternVL2-1B`
- `InternVL2-2B`
- `InternVL2-4B`
- `InternVL2-8B`
- `InternVL2-26B`
- `InternVL2-40B`
- `Qwen2-VL-2B-Instruct`
- `Qwen2-VL-7B-Instruct`

### Metrics

By default, the script uses the GOEval_referenced_mini metric. Other options are available as well, depending on the task and dataset.

### Configurations

- Batch Size: You can set the batch size using `--batch_size`. The default is 1.
- Dataset Split: Choose between train and test splits using `--split`.
- Task: Specify the task using `--task`. The default task is `eval`.
- Sampling: Enable sampling using `--do_sample`, and configure `--top_k` and `--top_p` to control randomness.
  - Results Folder: Specify a custom results folder with `--results_folder`.

### How to Run

Collect Experimental Data:

```bash
python main.py --task collect_experimental_data --split test
```
Execute Model:

```bash
python main.py --task execute --model_name gpt-4o --batch_size 8
```

Compute Metrics:

```bash
python main.py --task compute_metrics --metric GOEval_referenced_mini --results_folder ./results
```

Collect Results:

```bash
python main.py --task collect_results --results_folder ./results
```

Additional Arguments:
- `force_recompute`: Forces re-computation of all metrics.
- `do_sample`: Enables sampling during model execution.
- `top_k`: Sets the top-K sampling parameter.
- `top_p`: Sets the nucleus sampling threshold.

## File Structure

Following are some utility files and their descriptions:
- `main.py`: Entry point for running tasks.
- `const.py`: Contains task lists, supported models, and directory paths.
- `dataset_collector.py`: Handles the collection of datasets.
- `analyze_dataset.py`: Analyzes datasets.
- `execute.py`: Executes VLM models.
- `evaluate_results.py`: Computes metrics for evaluation.
- `image_descriptor_generator.py`: Generates image descriptors.
- `task_instance_classifications_generator.py`: Classifies task instances.