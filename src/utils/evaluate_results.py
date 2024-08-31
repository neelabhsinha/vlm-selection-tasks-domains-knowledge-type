from const import results_dir
from src.model.go_eval import GOEval
from src.utils.results_io_util import write_results
import numpy as np
from tqdm import tqdm

from src.metrics.bert_score import BertScore
from src.metrics.meteor import Meteor
from src.metrics.rouge import RougeScore

import json
import pandas as pd
import os

def compute_metric(metric, force_recompute=False, results_folder=None):
    skip_existing = False if force_recompute else True
    if results_folder is not None:
        files = [results_folder]
    else:
        files = os.listdir(results_dir)
    if 'bert_score' in metric:
        bert_score_calculator = BertScore()
    if metric == 'meteor':
        meteor_calculator = Meteor()
    if 'rouge' in metric:
        rouge_calculator = RougeScore()
    for file in tqdm(files, desc=f'Calculating {metric} for results'):
        path = os.path.join(results_dir, file, 'predictions.csv')
        try:
            df = pd.read_csv(path)
            predictions = df['candidate'].fillna('').tolist()
            references = df['reference'].fillna('').tolist()
            if 'bert_score' in metric and (
                    not skip_existing or ('bert_score_recall' not in df.columns or 'bert_score_f1' not in df.columns
                                          or 'bert_score_precision' not in df.columns)):
                scores = bert_score_calculator.get_score(predictions, references)
                f1 = np.array(scores['f1']) * 100
                recall = np.array(scores['recall']) * 100
                precision = np.array(scores['precision']) * 100
                df['bert_score_precision'] = precision
                df['bert_score_recall'] = recall
                df['bert_score_f1'] = f1
            if metric == 'meteor' and (not skip_existing or 'meteor' not in df.columns):
                scores = meteor_calculator.get_score(predictions, references)
                scores = np.array(scores) * 100
                df['meteor'] = scores
            if 'rouge' in metric and (not skip_existing or (
                    'rouge1' not in df.columns or 'rouge2' not in df.columns or 'rougeL' not in df.columns)):
                scores = rouge_calculator.get_score(predictions, references)
                df['rouge1'] = np.array(scores['rouge1']) * 100
                df['rouge2'] = np.array(scores['rouge2']) * 100
                df['rougeL'] = np.array(scores['rougeL']) * 100
            if 'GOEval_referenced' in metric and (not skip_existing or 'GOEval_referenced' not in df.columns):
                out = evaluate_go_eval_results(file, 'gpt-4o', 'referenced')
                if out is not None:
                    df['GOEval_referenced'] = np.array(out) * 100
            if 'GOEval_referenceless' in metric and (not skip_existing or 'GOEval_referenceless' not in df.columns):
                out = evaluate_go_eval_results(file, 'gpt-4o', 'referenceless')
                if out is not None:
                    df['GOEval_referenceless'] = np.array(out) * 100
            if 'GOEval_referenced_mini' in metric and (not skip_existing or 'GOEval_referenced_mini' not in df.columns):
                out = evaluate_go_eval_results(file, 'gpt-4o-mini', 'referenced')
                if out is not None:
                    df['GOEval_referenced_mini'] = np.array(out) * 100
            if 'GOEval_referenceless_mini' in metric and (not skip_existing or 'GOEval_referenceless_mini' not in df.columns):
                out = evaluate_go_eval_results(file, 'gpt-4o-mini', 'referenceless')
                if out is not None:
                    df['GOEval_referenceless_mini'] = np.array(out) * 100
            write_results(df, os.path.join(results_dir, file))
        except FileNotFoundError:
            print(f'Prediction file {file} not found in the given folder')

def evaluate_go_eval_results(results_folder, model_name, mode):
    model_name = model_name if model_name else 'gpt-4o-mini'
    evaluator = GOEval(model_name=model_name, mode=mode)
    if os.path.exists(os.path.join(results_dir, results_folder, f'{evaluator.mode}_eval_batch_object.json')):
        evaluator.get_batch_results(results_folder)
        results_path = os.path.join(results_dir, results_folder, f'{evaluator.mode}_eval_batch_results.jsonl')
        if os.path.exists(results_path):
            results = extract_content_from_jsonl_to_dataframe(results_path)
        return results 
    else:
        evaluator.submit_batch(results_folder)
        
            
def extract_content_from_jsonl_to_dataframe(jsonl_file_path):
    mode = jsonl_file_path.split("/")[-1].split("_")[0]
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            custom_id = record.get("custom_id", "")
            if "_" in custom_id:
                dataset_name, key = custom_id.split("_", 1)
                key = int(key)
            else:
                dataset_name, key = custom_id, -1
            if "response" in record and "body" in record["response"]:
                choices = record["response"]["body"].get("choices", [])
                if choices:
                    content = choices[0]["message"].get("content", "")
                else:
                    content = ""
            else:
                content = ""
            data.append(1 if 'yes' in content.lower() else 0)
    return data

