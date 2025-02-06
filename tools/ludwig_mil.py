import argparse
import pandas as pd
import numpy as np
import random
import os

def parse_bag_size(value):
    """Parses bag_size argument to handle both single integers and ranges."""
    if "-" in value:
        min_val, max_val = map(int, value.split("-"))
        if min_val > max_val:
            raise argparse.ArgumentTypeError("Invalid range: min value cannot be greater than max value.")
        return [min_val, max_val]
    return [int(value), int(value)]

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

def aggregate_embeddings(embeddings, pooling_method):
    if pooling_method == "max_pooling":
        return np.max(embeddings, axis=0)

    if pooling_method == "mean_pooling":
        return np.mean(embeddings, axis=0)

    if pooling_method == "sum_pooling":
        return np.sum(embeddings, axis=0)

def convert_embedding_to_string(embedding):
    return ",".join(map(str, embedding))

def bag_turns(df_embeddings, bag_sizes, pooling_method):
    embeddings_0 = df_embeddings.loc[df_embeddings["label"] == 0].values.tolist()
    embeddings_1 = df_embeddings.loc[df_embeddings["label"] == 1].values.tolist()

    np.random.shuffle(embeddings_0)
    np.random.shuffle(embeddings_1)
    bag_size_range = parse_bag_size(bag_sizes)

    make_bag_1 = True  # Alternate between making bags starting with class 1

    while embeddings_0 or embeddings_1:
        bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)  # Inclusive upper bound

        if make_bag_1 and embeddings_1:
            num_1_samples = min(np.random.randint(1, bag_size + 1), len(embeddings_1))
        else:
            num_1_samples = 0  # If not making bag 1, no samples from embeddings_1

        selected_embeddings_1 = embeddings_1[:num_1_samples]
        embeddings_1 = embeddings_1[num_1_samples:]

        num_0_samples = min(bag_size - num_1_samples, len(embeddings_0))
        selected_embeddings_0 = embeddings_0[:num_0_samples]
        embeddings_0 = embeddings_0[num_0_samples:]

        # Combine to form the final bag
        bag_embeddings = selected_embeddings_0 + selected_embeddings_1

        # If bag is still not full, fill with remaining class 1 embeddings
        if len(bag_embeddings) < bag_size and embeddings_1:
            num_extra = min(bag_size - len(bag_embeddings), len(embeddings_1))
            extra_embeddings = embeddings_1[:num_extra]
            embeddings_1 = embeddings_1[num_extra:]
            bag_embeddings += extra_embeddings


        # Toggle bag type for next iteration
        make_bag_1 = not make_bag_1#        if len(bag_embeddings) > 0:
        sample_names = [x[0] for x in bag_embeddings]
        sample_labels = [x[1] for x in bag_embeddings]
        sample_split = [x[2] for x in bag_embeddings]

        only_embeddings = [row[3:] for row in bag_embeddings]

        aggregated_embedding = aggregate_embeddings(only_embeddings, pooling_method)

        print(aggregated_embedding)

def bag_random():
    pass

def bag_processing(embeddings, metadata, pooling_method, balance_enforced=False, bag_sizes=[3,5], seed=42):
    bags = []
    bag_set = set()
    bags_sizes = parse_bag_size(bag_sizes)

    for split in metadata['split'].unique():
        split_metadata = metadata[metadata['split'] == split]
        split_embeddings = pd.merge(split_metadata, embeddings, on='sample_name')
        if balance_enforced:
            bag_turns(split_embeddings, bag_sizes, pooling_method)
        #else:
            #bag_ramdom()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to create bags of embeddings")

    parser.add_argument("--embeddings_csv", type=str, required=True, help="The embeddings in CSV file (Must have 'sample_name' column)")
    parser.add_argument("--metadata_csv", type=str, required=True, help="The metadata in CSV file (Must contain 'sample_name' column and 'label' column.")
    parser.add_argument("--split_proportions", type=str, default='0.7,0.1,0.2', help="Proportions for train, validation, and test splits.")
    parser.add_argument("--dataleak", action="store_true", help="Prevents dataleak when splitting the data.")
    parser.add_argument("--balance_enforced", action="store_true", help="Create bags in turns, reducing probability of large imbalacend bags")
    parser.add_argument("--bag_size", type=str, required=True, help="Bag size as a single number (e.g., 4) or a range (e.g., 3-5).")    
    parser.add_argument("--seed", type=int, help="seed number")
    parser.add_argument("--pooling_method", type=str, required=True, help="The method for pooling the embeddings")
    #parser.add_argument("--output_csv", required=True, help="Path to the output CSV file")

    args = parser.parse_args()

    embeddings_data = load_csv(args.embeddings_csv)
    metadata_csv = load_csv(args.metadata_csv)

    if "split" not in metadata_csv:
        metadata_csv = split_data(
            metadata_csv,
            split_proportions = args.split_proportions, 
            dataleak = args.dataleak
        )
    bag_processing(embeddings_data, metadata_csv, args.pooling_method, args.balance_enforced, args.bag_size, args.seed)

