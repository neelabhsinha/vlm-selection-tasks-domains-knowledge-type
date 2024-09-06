import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from const import aggregated_results_dir
from src.data.experimental_dataset import ExperimentalDataset
from matplotlib import rcParams


def analyze_dataset(split='test'):
    # Initialize dataset
    dataset = ExperimentalDataset(get_image_descriptors=True, get_classification=True, split=split)
    num_samples = len(dataset)

    # Initialize lists to collect statistics
    caption_lengths = []
    num_tags_list = []
    application_domains_list = []
    knowledge_types_list = []
    num_application_domains = []
    num_knowledge_types = []
    task_type_count = Counter()

    # Process each sample
    for idx in tqdm(range(0, num_samples), total=num_samples, desc='Generating dataset statistics'):
        image_descriptors = dataset[idx]['image_descriptors']
        classification = dataset[idx]['classification']

        # Extract relevant information
        caption_length = len(image_descriptors['caption'])
        num_tags = len(image_descriptors['object_tags'])
        application_domains = classification['application_domain']
        knowledge_types = classification['knowledge_type']

        # Append to lists
        caption_lengths.append(caption_length)
        num_tags_list.append(num_tags)
        application_domains_list.extend(application_domains)
        num_application_domains.append(len(application_domains))
        knowledge_types_list.extend(knowledge_types)
        num_knowledge_types.append(len(knowledge_types))
        task_type_count[classification['task_type']] += 1

    # Convert lists to numpy arrays for statistical analysis
    caption_lengths = np.array(caption_lengths)
    num_tags_list = np.array(num_tags_list)

    # Calculate mean and std for caption lengths and number of tags
    avg_caption_length = np.mean(caption_lengths)
    std_caption_length = np.std(caption_lengths)
    max_caption_length = max(caption_lengths)
    avg_num_tags = np.mean(num_tags_list)
    std_num_tags = np.std(num_tags_list)
    max_num_tags = max(num_tags_list)

    # Calculate number of unique application domains and knowledge types per task
    avg_num_app_domains = np.mean(num_application_domains)
    std_num_app_domains = np.std(num_application_domains)
    avg_num_knowledge_types = np.mean(num_knowledge_types)
    std_num_knowledge_types = np.std(num_knowledge_types)
    max_num_app_domains = max(num_application_domains)
    max_num_knowledge_types = max(num_knowledge_types)

    # Create directory to save results
    output_dir = os.path.join(aggregated_results_dir, 'dataset_statistics')
    os.makedirs(output_dir, exist_ok=True)

    # Save statistics to a text file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write(f"Average Caption Length: {avg_caption_length:.2f}\n")
        f.write(f"Standard Deviation of Caption Length: {std_caption_length:.2f}\n")
        f.write(f"Maximum Caption Length: {max_caption_length}\n")
        f.write(f"Average Number of Object Tags: {avg_num_tags:.2f}\n")
        f.write(f"Standard Deviation of Number of Object Tags: {std_num_tags:.2f}\n")
        f.write(f"Maximum Number of Object Tags: {max_num_tags}\n")
        f.write(f"Mean Number of Application Domains per Task Instance: {avg_num_app_domains:.2f}\n")
        f.write(f"Standard Deviation of Application Domains per Task Instance: {std_num_app_domains:.2f}\n")
        f.write(f"Max Number of Application Domains per Task Instance: {max_num_app_domains:.2f}\n")
        f.write(f"Mean Number of Knowledge Types per Task Instance: {avg_num_knowledge_types:.2f}\n")
        f.write(f"Standard Deviation of Knowledge Types per Task Instance: {std_num_knowledge_types:.2f}\n")
        f.write(f"Max Number of Knowledge Types per Task Instance: {max_num_knowledge_types:.2f}\n")

        app_domain_counter = Counter(application_domains_list)
        knowledge_type_counter = Counter(knowledge_types_list)
        # Filter out counts less than 300
        app_domain_counter = {k: v for k, v in app_domain_counter.items() if v > 300}
        knowledge_type_counter = {k: v for k, v in knowledge_type_counter.items() if v > 300}

        # Set font to Montserrat
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 16

        # Plot bar charts and save them as PDFs
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 2, 1)
        plt.bar(app_domain_counter.keys(), app_domain_counter.values(), color='orange')
        plt.xlabel('Application Domain')
        plt.ylabel('Number of Instances')
        plt.title('Number of Instances per Application Domain')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.bar(knowledge_type_counter.keys(), knowledge_type_counter.values(), color='darkgreen')
        plt.xlabel('Knowledge Type')
        plt.ylabel('Number of Instances')
        plt.title('Number of Instances per Knowledge Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tag_statistics.pdf'), format='pdf')

        plt.close()
