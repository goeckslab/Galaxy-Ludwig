import argparse
import pandas as pd
import numpy as np
import random
import os

def load_csv(file_path):
    return pd.read_csv(file_path)

def str_array_split(split_proportions):
    split_array = [float(p) for p in split_proportions.split(",")]
    if len(split_array) == 2:
        split_array.insert(1,0.0)
    return split_array

def split_sizes(num_samples, proportions):
    train_size = int(proportions[0] * num_samples)
    val_size = int(proportions[1] * num_samples) if proportions[1] > 0 else 0
    test_size = int(proportions[2] * num_samples)
    return train_size, val_size, test_size

def split_data(metadata, split_proportions, dataleak=False):
    proportions = str_array_split(split_proportions)

    if dataleak:
        list_samples = metadata["sample_name"].unique()
    else:
        list_samples = metadata["sample_name"]

    num_samples = len(list_samples)

    train_size, val_size, test_size = split_sizes(num_samples, proportions)

    shuffled_samples = np.random.permutation(list_samples)

    train_samples = shuffled_samples[:train_size]
    if val_size > 0:
        val_samples = shuffled_samples[train_size:train_size + val_size]
        test_samples = shuffled_samples[train_size + val_size:]
    else:
        val_samples = []  # No validation set
        test_samples = shuffled_samples[train_size:]

    split_column = {sample: 0 for sample in train_samples}
    if val_size > 0:
        split_column.update({sample: 1 for sample in val_samples})
    split_column.update({sample: 2 for sample in test_samples})

    metadata["split"] = metadata["sample_name"].map(split_column)

    return metadata

def aggregate_max_pooling(embeddings):
    return np.max(embeddings, axis=0)

def aggregate_mean_pooling(embeddings):
    return np.mean(embeddings, axis=0)

def aggregate_sum_pooling(embeddings):
    return np.sum(embeddings, axis=0)

def convert_embedding_to_string(embedding):
    return ",".join(map(str, embedding))

def create_bags_from_split(split_data, split, repeats, bag_size, balance_enforce, pooling_method):
    bags = []
    bag_set = set()

    pooling_functions = {
        'maxpool': aggregate_max_pooling,
        'meanpool': aggregate_mean_pooling,
        'sum': aggregate_sum_pooling
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to create bags of embeddings")

    parser.add_argument("--embeddings_csv", type=str, required=True, help="The embeddings in CSV file (Must have 'sample_name' column)")
    parser.add_argument("--metadata_csv", type=str, required=True, help="The metadata in CSV file (Must contain 'sample_name' column and 'label' column.")
    parser.add_argument("--split_proportions", type=str, default='0.7,0.1,0.2', help="Proportions for train, validation, and test splits.")
    parser.add_argument("--dataleak", action="store_true", help="Prevents dataleak when splitting the data.")

    args = parser.parse_args()

    embeddings_data = load_csv(args.embeddings_csv)
    metadata_csv = load_csv(args.metadata_csv)

    if "split" not in metadata_csv:
        metadata_csv = split_data(
            metadata_csv, 
            split_proportions = args.split_proportions, 
            dataleak = args.dataleak
        )
    print(metadata_csv)
