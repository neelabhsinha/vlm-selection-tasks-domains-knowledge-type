# Guiding Vision-Language Model Selection for Visual Question-Answering with VQA360 Dataset and GoEval Metric

## Overview

This repository contains the code for evaluating various Vision-Language Models (VLMs) on multiple tasks across different domains, categories, and types of reasoning. The evaluation process supports multiple models and metrics, with pre-built tasks to handle data collection, image descriptor generation, model execution, and performance evaluation.

## Table of Contents

- [Dataset](#1-dataset)
  - [Overview](#overview)
  - [Dataset Composition](#dataset-composition)
  - [Features](#features)
  - [Usage](#usage)
  - [Statistics](#statistics)
  - [Files](#files)
- [Repository Installation](#2-repository-installation)
- [Usage](#3-usage)
  - [Available Tasks](#available-tasks)
  - [Supported Models](#supported-models)
  - [Metrics](#metrics)
  - [Configurations](#configurations)
- [How to Run](#4-how-to-run)
- [File Structure](#5-file-structure)
- [Paper Results](#6-paper-results)
- [Citation](#7-citation)

## 1. Dataset

### Overview
This repository contains a dataset designed for evaluating Vision-Language Models (VLMs) on a variety of Visual Question Answering (VQA) tasks. The dataset is compiled from five well-known VQA benchmarks, providing a diverse set of task instances across different application domains and knowledge types.

### Dataset Composition
The dataset includes 5725 task instances, with equal contributions from the following source datasets:
- VQAv2
- OK-VQA
- A-OKVQA
- ChartQA
- DocumentVQA

### Features
- **Multi-Dimensional Tags**: Each instance is annotated not only with the task type but also with tags for application domains and knowledge types, allowing for nuanced analyses of model performance across various dimensions.
- **Balanced Contributions**: Each source dataset contributes an equal number of task instances to ensure diversity and balance in the dataset.
- **Extensive Annotations**: The dataset includes detailed annotations for application domains and knowledge types, crafted to reflect a broad spectrum of real-world applications.

### Usage
This dataset is intended for use in research on Vision-Language Models, particularly in the context of Visual Question Answering. Researchers can use this dataset to evaluate model performance, understand model biases, and develop new VLM approaches.

Using the code of this repository, you can also generate training set or a more exhaustive test set. Refer to code usage guidelines below.

### Statistics

Dataset Statistics: [View PDF](aggregated_results/dataset_statistics/tag_statistics.pdf)

### Files

The dataset proposed with this work is a collection of tasks across different task types, application domains and knowledge types. It is located in the `experimental_data` directory and contains the following subdirectories:
- `data`: A JSON Object for each task instance containing the following fields:
  - `dataset` (String): Name of the dataset from which it is taken
  - `key` (Integer): Unique identifier for the task instance in that dataset (Dataset + key uniquely identifies a task instance)
  - `image_path` (String): Path to the image file in corresponding `images` directory
  - `question` (String): Question text
  - `label` (List): Ground truth answer
  - `choices` (List or None): List of answer choices (if present. Not used in this work.)
- `images`: Contains images for each task instance
- `image_descriptors`: A JSON object for each task instance with image descriptors generated for each image
  - `dataset` (String): Name of the dataset from which it is taken
  - `key` (Integer): Unique identifier for the task instance in that dataset (Dataset + key uniquely identifies a task instance)
  - `caption` (String): Image caption generated using VIT-GPT2
  - `object_tags` (List): Object tags generated using Azure Computer Vision API
- `task_instance_classifications`: A JSON object for each task instance with model predictions
  - `dataset (String)`: Name of the dataset from which it is taken
  - `key` (Integer): Unique identifier for the task instance in that dataset (Dataset + key uniquely identifies a task instance)
  - `task_type` (String): Type of task (e.g., Chart Understanding, Knowledge-based Visual Question Answering, etc.)
  - `application_domain` (List): Application domains of the task (e.g., Mathematics, Social Media, etc.)
  - `knowledge_type` (List): Knowledge type required to solve the task (e.g., Visual, Commonsense, etc.)

A corresponding PyTorch Dataset class can be found in `src/data/experimental_dataset.py` to load the dataset for model evaluation.

## 2. Repository Installation

To use this repository, first clone it and install the required dependencies.

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

## 3. Usage

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

## 4.  How to Run

Collect Experimental Data:

```bash
python main.py --task collect_experimental_data --split test
```

Generate Image Descriptors:

```bash
python main.py --task generate_image_descriptors --batch_size 8
```

Generate Instance Classifications:

```bash
python main.py --task generate_instance_classifications --model_name gpt-4o --batch_size 8
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

## 5. File Structure

Following are the code files inside the `src` directory:
- `main.py`: Entry point for running tasks.
- `const.py`: Contains task lists, supported models, and directory paths.
- `prompts.py`: Contains prompts for different tasks.
- `data`: Contains classes for loading and processing data.
  - `dataset_collector.py`: Class for collecting experimental datasets.
  - `experimental_dataset.py`: PyTorch Dataset class for loading the experimental dataset.
  - `raw_dataset.py`: PyTorch Dataset class for loading raw datasets.
- `metrics`: Contains classes for computing evaluation metrics.
  - `bert_score.py`: Class for computing BERTScore.
  - `go_eval.py`: Class for computing GOEval.
  - `meteor.py`: Class for computing METEOR.
  - `rouge.py`: Class for computing ROUGE.
- `models`: Contains classes for executing Vision-Language Models.
  - `gpt_3_5_turbo.py`: Class for executing GPT-3.5 Turbo for collecting application domain and knowledge type tags.
  - `object_tags_generator.py`: Class for generating object tags using Azure Computer Vision API.
  - `vit_gpt2_caption_generator.py`: Class for generating image captions using VIT-GPT2.
  - `vlm.py`: Classes having implementation of all supported Vision-Language Models.
- `utils`: Contains utility functions.
  - `aggregate_results.py`: Functions for aggregating evaluation results.
  - `analyze_dataset.py`: Functions for analyzing the dataset.
  - `evaluate_results.py`: Functions for evaluating model performance.
  - `execute.py`: Functions for executing Vision-Language Models.
  - `gpu_status.py`: Functions for checking GPU Memory.
  - `image_descriptor_generator.py`: Functions for generating image descriptors (caption and object tags).
  - `json_utils.py`: Functions for reading and writing JSON files.
  - `radar_chart_generator.py`: Functions for generating radar charts.
  - `results_io_util.py`: Functions for reading and writing evaluation results.
  - `task_instance_classifications_generator.py`: Functions for generating tags of task type, application domain, and knowledge type (assumes that the image descriptors are already generated).

## 6. Paper Results

The results of the paper are available in the `aggregated_results` directory. The results are aggregated across all tasks and models, providing a comprehensive view of model performance.

## 7. Citation

If you use this dataset or code in your research, please cite the following paper:

```bibtex
NOT ADDED AS THE PAPER IS CURRENTLY UNDER REVIEW
```
