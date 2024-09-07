import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from const import results_dir, aggregated_results_dir, beautified_model_names, valid_application_domains, valid_knowledge_types
from src.utils.radar_chart_generator import RadarChartPlotter

from src.utils.results_io_util import process_list_field


def calculate_correlation_between_eval_metrics(results_folder):
    results_folder = 'correlation_analysis'
    eval_results = pd.read_csv(os.path.join(results_dir, results_folder, 'predictions.csv'))
    eval_results = eval_results.dropna()
    columns_to_include = [
        'bert_score_precision', 'bert_score_recall', 'bert_score_f1',
        'rouge1', 'rouge2', 'rougeL', 'user_input',
        'GOEval_referenced_mini', 'GOEval_referenced',
        'GOEval_referenceless', 'GOEval_referenceless_mini',
        'GOEval_referenced_imageless_mini', 'GOEval_referenced_imageless',
        'GOEval_referenceless_imageless', 'GOEval_referenceless_imageless_mini'
    ]
    accuracy_matrix = pd.DataFrame(index=columns_to_include, columns=['Accuracy'])
    for col in columns_to_include:
        accuracy = (eval_results['user_input'] == eval_results[col]).sum() / eval_results.shape[0]
        accuracy_matrix.loc[col, 'Accuracy'] = accuracy
    accuracy_matrix = accuracy_matrix.T
    dir_path = os.path.join(aggregated_results_dir, 'experiment_correlation_between_metrics')
    os.makedirs(dir_path, exist_ok=True)
    accuracy_matrix.to_csv(os.path.join(dir_path, 'accuracy_to_human.csv'), index=True)

    eval_results = eval_results[columns_to_include]
    pearson_corr = eval_results.corr(method='pearson')
    kendall_corr = eval_results.corr(method='kendall')
    spearman_corr = eval_results.corr(method='spearman')
    pearson_corr.to_csv(os.path.join(dir_path, 'pearson_correlation.csv'))
    kendall_corr.to_csv(os.path.join(dir_path, 'kendall_correlation.csv'))
    spearman_corr.to_csv(os.path.join(dir_path, 'spearman_correlation.csv'))


def collect_model_performance_correlation_results(metric):
    models = os.listdir(results_dir)
    all_model_results = pd.DataFrame()
    try:
        models.remove('correlation_analysis')
        models.remove('.DS_Store')
    except ValueError:
        pass
    for folder in models:
        try:
            model_name = folder.split('--')[1]
            if model_name in beautified_model_names:
                model_name = beautified_model_names[model_name]
        except IndexError:
            print('Incorrect folder name:', folder)
        predictions_file = os.path.join(results_dir, folder, 'predictions.csv')
        predictions_df = pd.read_csv(predictions_file)
        all_model_results[model_name] = predictions_df[metric]
    order_of_models = beautified_model_names.values()
    for model in order_of_models:
        if model not in all_model_results.columns:
            order_of_models.remove(model)
    all_model_results = all_model_results[order_of_models].apply(pd.to_numeric, errors='coerce')
    corr = all_model_results.corr(method='pearson')
    save_correlation_matrix(corr, os.path.join(aggregated_results_dir, 'model_performance_correlation',
                                               'model_performance_correlation.pdf'))


def collect_model_comparison_results(metric):
    merged_task_type_df = None
    merged_domain_df = None
    merged_knowledge_type_df = None
    models = os.listdir(results_dir)
    try:
        models.remove('correlation_analysis')
        models.remove('.DS_Store')
    except ValueError:
        pass
    task_type_instance_count = None
    domain_instance_count = None
    knowledge_type_instance_count = None
    for folder in models:
        try:
            model_name = folder.split('--')[1]
            if model_name in beautified_model_names:
                model_name = beautified_model_names[model_name]
        except IndexError:
            print('Incorrect folder name:', folder)
        predictions_file = os.path.join(results_dir, folder, 'predictions.csv')
        predictions_df = pd.read_csv(predictions_file)
        list_fields = ['domain', 'knowledge_type']
        for list_field in list_fields:
            predictions_df[list_field] = predictions_df[list_field].apply(process_list_field)
            predictions_df[list_field] = predictions_df[list_field].apply(lambda x: x.split(';') if x else [])
        task_type_df, domain_df, knowledge_type_df = aggregate_aspect_level_results(predictions_df)
        merged_task_type_df = merge_dataframe(merged_task_type_df, task_type_df, model_name, 'task_type', metric)
        merged_domain_df = merge_dataframe(merged_domain_df, domain_df, model_name, 'domain', metric)
        merged_knowledge_type_df = merge_dataframe(merged_knowledge_type_df, knowledge_type_df, model_name,
                                                   'knowledge_type', metric)
        task_type_instance_count = task_type_df['num_instances']
        domain_instance_count = domain_df['num_instances']
        knowledge_type_instance_count = knowledge_type_df['num_instances']
    merged_task_type_df['num_instances'] = task_type_instance_count
    merged_domain_df['num_instances'] = domain_instance_count
    merged_knowledge_type_df['num_instances'] = knowledge_type_instance_count
    folder_path = os.path.join(aggregated_results_dir, 'comparative_model_performances')
    os.makedirs(folder_path, exist_ok=True)
    merged_task_type_df.set_index('task_type', inplace=True)
    merged_domain_df.set_index('domain', inplace=True)
    merged_knowledge_type_df.set_index('knowledge_type', inplace=True)
    # add a column to name the model (other column names except num_instances) which has max value
    merged_task_type_df['best_model'] = merged_task_type_df.drop(columns=['num_instances']).idxmax(axis=1)
    merged_domain_df['best_model'] = merged_domain_df.drop(columns=['num_instances']).idxmax(axis=1)
    merged_knowledge_type_df['best_model'] = merged_knowledge_type_df.drop(columns=['num_instances']).idxmax(axis=1)
    merged_task_type_df.to_csv(os.path.join(folder_path, 'task_type_variation.csv'))
    merged_domain_df.to_csv(os.path.join(folder_path, 'domain_variation.csv'))
    merged_knowledge_type_df.to_csv(os.path.join(folder_path, 'knowledge_type_variation.csv'))
    radar_chart_generator = RadarChartPlotter()
    included_models = beautified_model_names.values()
    for model in included_models:
        if model not in merged_task_type_df.columns:
            included_models.remove(model)
    merged_domain_df = merged_domain_df.reindex(valid_application_domains)
    merged_knowledge_type_df = merged_knowledge_type_df.reindex(valid_knowledge_types)
    radar_chart_generator.plot_radar_chart(merged_domain_df, 300, included_models,
                                           os.path.join(folder_path, 'domain_variation.pdf'))
    radar_chart_generator.plot_radar_chart(merged_knowledge_type_df, 300, included_models,
                                           os.path.join(folder_path, 'knowledge_type_variation.pdf'))


def aggregate_aspect_level_results(df):
    domains_df = df.explode('domain').groupby('domain').apply(get_stats)
    task_type_df = df.groupby('task_type').apply(get_stats)
    knowledge_type_df = df.explode('knowledge_type').groupby('knowledge_type').apply(get_stats)
    return task_type_df.reset_index(), domains_df.reset_index(), knowledge_type_df.reset_index()


def get_stats(group_df):
    metrics = ['GOEval_referenced_mini', 'rouge1', 'rouge2', 'rougeL', 'meteor', 'bert_score_precision',
               'bert_score_recall', 'bert_score_f1']
    existing_metrics = [metric for metric in metrics if metric in group_df.columns]
    if existing_metrics:
        stats = group_df[existing_metrics].mean() * 100
        stats['num_instances'] = len(group_df)
        return stats
    else:
        return pd.Series([float('nan')] * len(metrics), index=metrics)


def merge_dataframe(result_df, df, column_key, aspect, metric):
    if aspect is not None and metric is not None:
        df = df[[aspect, metric]]
    elif metric is not None:
        df = df[[metric]]
    df = df.rename(columns={metric: column_key})
    if result_df is None:
        result_df = df
    else:
        result_df = result_df.merge(df, on=aspect, how='outer')
    return result_df


def save_correlation_matrix(corr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Set larger font size for axis labels and ticks
    ax = sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm',
                     xticklabels=corr.columns, yticklabels=corr.columns,
                     cbar_kws={"shrink": .75})

    # Adjusting the font sizes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right', fontsize=15)

    plt.subplots_adjust(left=0.2, right=1, top=0.95, bottom=0.2)
    plt.savefig(path)
    plt.close()

