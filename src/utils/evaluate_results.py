from const import results_dir
from src.metrics.go_eval import GOEval
from src.utils.results_io_util import write_results, process_list_field
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
    if 'GOEval' in metric:
        reference_type = 'referenced' if 'referenced' in metric else 'referenceless'
        model_name = 'gpt-4o' if 'mini' not in metric else 'gpt-4o-mini'
        go_eval_calculator = GOEval(mode=reference_type, model_name=model_name)
    for file in tqdm(files, desc=f'Calculating {metric} for results'):
        path = os.path.join(results_dir, file, 'predictions.csv')
        try:
            df = pd.read_csv(path)
            predictions = df['response'].fillna('').tolist()
            references = df['label'].fillna('').tolist()
            if 'bert_score' in metric and (
                    not skip_existing or ('bert_score_recall' not in df.columns or 'bert_score_f1' not in df.columns
                                            or 'bert_score_precision' not in df.columns)):
                references = [process_list_field(reference) for reference in references]
                scores = bert_score_calculator.get_score(predictions, references)
                f1 = np.array(scores['f1'])
                recall = np.array(scores['recall'])
                precision = np.array(scores['precision'])
                df['bert_score_precision'] = precision
                df['bert_score_recall'] = recall
                df['bert_score_f1'] = f1
            if metric == 'meteor' and (not skip_existing or 'meteor' not in df.columns):
                references = [process_list_field(reference) for reference in references]
                scores = meteor_calculator.get_score(predictions, references)
                scores = np.array(scores)
                df['meteor'] = scores
            if 'rouge' in metric and (not skip_existing or (
                    'rouge1' not in df.columns or 'rouge2' not in df.columns or 'rougeL' not in df.columns)):
                scores = rouge_calculator.get_score(predictions, references)
                references = [process_list_field(reference) for reference in references]
                df['rouge1'] = np.array(scores['rouge1'])
                df['rouge2'] = np.array(scores['rouge2'])
                df['rougeL'] = np.array(scores['rougeL'])
            if 'GOEval' in metric and (not skip_existing or metric not in df.columns):
                out = go_eval_calculator.evaluate(df['question'].fillna('').tolist(), df['image_path'].fillna('').tolist(), references, predictions)
                if out is not None:
                    df[metric] = np.array(out)
            write_results(df, os.path.join(results_dir, file))
        except FileNotFoundError:
            print(f'Prediction file {file} not found in the given folder')

