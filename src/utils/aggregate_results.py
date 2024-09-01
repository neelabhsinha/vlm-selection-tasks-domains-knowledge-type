
import os

import pandas as pd

from const import results_dir, aggregated_results_dir


def calculate_correlation_between_matrix(results_folder):
    eval_results = pd.read_csv(os.path.join(results_dir, results_folder, 'predictions.csv'))
    eval_results = eval_results.dropna()
    columns_to_include = [
    'bert_score_precision', 'bert_score_recall', 'bert_score_f1', 
    'rouge1', 'rouge2', 'rougeL', 'user_input', 
    'GOEval_referenced_mini', 'GOEval_referenced', 
    'GOEval_referenceless', 'GOEval_referenceless_mini'
]
    eval_results = eval_results[columns_to_include]
    pearson_corr = eval_results.corr(method='pearson')
    kendall_corr = eval_results.corr(method='kendall')
    spearman_corr = eval_results.corr(method='spearman')
    dir_path = os.path.join(aggregated_results_dir, 'experiment_correlation_between_metrics', results_folder)
    os.makedirs(dir_path, exist_ok=True)
    pearson_corr.to_csv(os.path.join(dir_path, 'pearson_correlation.csv'))
    kendall_corr.to_csv(os.path.join(dir_path, 'kendall_correlation.csv'))
    spearman_corr.to_csv(os.path.join(dir_path, 'spearman_correlation.csv'))