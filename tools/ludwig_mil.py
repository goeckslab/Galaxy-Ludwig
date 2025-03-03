"""
Module for processing embeddings and metadata to generate bags of \
embeddings, with options for splitting, pooling, and balancing.

This module provides functions for:
1. **Loading and parsing CSV files** containing embeddings and metadata.
2. **Splitting the dataset** into training, validation, and test sets \
based on user-specified proportions or avoiding data leakage.
3. **Generating bags of embeddings** using various methods, including random \
sampling and balancing classes, and applying different pooling techniques.
4. **Transforming the embeddings** to the format required for Ludwig \
(a machine learning framework) or saving them to a CSV file.

Key Functions:
- `parse_bag_size`: Parses the bag size argument to handle \
both single integers and ranges.
- `load_csv`: Loads a CSV file into a pandas DataFrame.
- `split_data`: Splits the data based on the specified proportions \
for train, validation, and test sets.
- `bag_turns`: Generates bags of embeddings in turns, \
alternating between classes for balanced bags.
- `bag_random`: Generates random bags of embeddings, \
optionally balancing the dataset.
- `aggregate_embeddings`: Aggregates embeddings \
using various pooling methods (max, mean, sum, etc.).
- `transform_bags_for_ludwig`: Transforms bags into the format \
suitable for Ludwig.
- `write_csv`: Writes the processed bags of embeddings to a CSV file.

Command-Line Interface:
This module includes a CLI that can be run with the following arguments:
- `--embeddings_csv`: Path to the CSV file containing embeddings.
- `--metadata_csv`: Path to the CSV file containing metadata \
(must include 'sample_name' and 'label' columns).
- `--split_proportions`: Proportions for splitting the dataset \
into train, validation, and test sets (default is '0.7,0.1,0.2').
- `--dataleak`: Flag to prevent data leakage when splitting.
- `--balance_enforced`: Flag to create balanced \
bags by alternating between classes.
- `--bag_size`: Specifies the bag size (either a single number or a range).
- `--seed`: Seed for random number generation.
- `--pooling_method`: The method to aggregate embeddings \
(e.g., 'max_pooling', 'mean_pooling').
- `--repeats`: The number of times to repeat the process of generating bags.
- `--ludwig_format`: Flag to prepare data for Ludwig input format \
(embedding as a string).
- `--output_csv`: Path to save the resulting CSV file.
"""
import argparse
import csv
import logging

import numpy as np

import pandas as pd

import torch
import torch.nn as nn


# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    # so write to a file
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)


def parse_bag_size(value):
    """Parses bag_size argument to handle both single integers and ranges."""
    if "-" in value:
        min_val, max_val = map(int, value.split("-"))
        if min_val > max_val:
            raise argparse.ArgumentTypeError(
                "Invalid range: min value cannot be greater than max value."
            )
        return [min_val, max_val]
    return [int(value), int(value)]


def load_csv(file_path):
    return pd.read_csv(file_path)


def str_array_split(split_proportions):
    split_array = [float(p) for p in split_proportions.split(",")]
    if len(split_array) == 2:
        split_array.insert(1, 0.0)
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

    train_size, val_size, _ = split_sizes(num_samples, proportions)

    shuffled_samples = np.random.permutation(list_samples)

    train_samples = shuffled_samples[:train_size]
    logging.info(train_samples)
    if val_size > 0:
        val_samples = shuffled_samples[train_size:train_size + val_size]
        test_samples = shuffled_samples[train_size + val_size:]
    else:
        val_samples = []  # No validation set
        test_samples = shuffled_samples[train_size:]
    logging.info(val_samples)
    logging.info(test_samples)
    split_column = {sample: 0 for sample in train_samples}
    if val_size > 0:
        split_column.update({sample: 1 for sample in val_samples})
    split_column.update({sample: 2 for sample in test_samples})

    metadata["split"] = metadata["sample_name"].map(split_column)
    return metadata


def attention_pooling(embeddings):
    """Applies attention-based pooling to the embeddings."""
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    weights = nn.Softmax(dim=0)(nn.Linear(tensor.shape[1], 1)(tensor))
    pooled_embedding = torch.sum(weights * tensor, dim=0)
    return pooled_embedding.detach().numpy()


def gated_pooling(embeddings):
    """Applies gated pooling to the embeddings."""
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    gate = nn.Sigmoid()(nn.Linear(tensor.shape[1], tensor.shape[1])(tensor))
    pooled_embedding = torch.sum(gate * tensor, dim=0)
    return pooled_embedding.detach().numpy()


def aggregate_embeddings(embeddings, pooling_method):
    if pooling_method == "max_pooling":
        return np.max(embeddings, axis=0)
    if pooling_method == "mean_pooling":
        return np.mean(embeddings, axis=0)
    if pooling_method == "sum_pooling":
        return np.sum(embeddings, axis=0)
    if pooling_method == "min_pooling":
        return np.min(embeddings, axis=0)
    if pooling_method == "median_pooling":
        return np.median(embeddings, axis=0)
    if pooling_method == "l2_norm_pooling":
        return np.linalg.norm(embeddings, axis=0)
    if pooling_method == "geometric_mean_pooling":
        return np.exp(np.mean(np.log(np.clip(embeddings,
                                             1e-10, None)), axis=0))
    if pooling_method == "first_embedding":
        return embeddings[0]
    if pooling_method == "last_embedding":
        return embeddings[-1]
    if pooling_method == "attention_pooling":
        return attention_pooling(embeddings)
    if pooling_method == "gated_pooling":
        return gated_pooling(embeddings)
    raise ValueError(f"Unknown pooling method: {pooling_method}")


def bag_by_sample(df, pooling_method, bag_size):
    all_bags = []

    for _, group in df.groupby("sample_name"):
        embeddings = group.iloc[:, 3:].values
        sample_names = group["sample_name"].values
        labels = group["label"].values
        split = group["split"].iloc[0]

        num_instances = len(group)
        random_bag_size = np.random.randint(bag_size[0], bag_size[1] + 1)
        num_bags = (num_instances + random_bag_size - 1) // random_bag_size

        for i in range(num_bags):
            start_idx = i * random_bag_size
            end_idx = min(start_idx + random_bag_size, num_instances)

            bag_embeddings = embeddings[start_idx:end_idx]
            bag_sample_names = sample_names[start_idx:end_idx]
            bag_labels = labels[start_idx:end_idx]

            aggregated_embeddings = aggregate_embeddings(bag_embeddings,
                                                         pooling_method)
            bag_label = int(any(bag_labels == 1))
            all_bags.append({
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(bag_sample_names),
                "bag_samples": list(bag_sample_names),
                "embedding": aggregated_embeddings
            })

    return all_bags


def bag_turns(df, bag_sizes, pooling_method, repeats):
    all_bags = []
    for _ in range(repeats):
        embeddings_0 = df.loc[df["label"] == 0].values.tolist()
        embeddings_1 = df.loc[df["label"] == 1].values.tolist()

        np.random.shuffle(embeddings_0)
        np.random.shuffle(embeddings_1)

        make_bag_1 = True
        bags = []
        bag_set = set()

        while embeddings_0 or embeddings_1:
            bag_size = np.random.randint(bag_sizes[0],
                                         bag_sizes[1]+1)

            if make_bag_1 and embeddings_1:
                num_1_samples = min(np.random.randint(1, bag_size + 1),
                                    len(embeddings_1))
            else:
                num_1_samples = 0

            selected_embeddings_1 = embeddings_1[:num_1_samples]
            embeddings_1 = embeddings_1[num_1_samples:]

            num_0_samples = min(bag_size - num_1_samples, len(embeddings_0))
            selected_embeddings_0 = embeddings_0[:num_0_samples]
            embeddings_0 = embeddings_0[num_0_samples:]

            # Combine to form the final bag
            bag_embeddings = selected_embeddings_0 + selected_embeddings_1

            # If bag is still not full, fill with remaining class 1 embeddings
            if len(bag_embeddings) < bag_size and embeddings_1:
                num_extra = min(bag_size - len(bag_embeddings),
                                len(embeddings_1))
                extra_embeddings = embeddings_1[:num_extra]
                embeddings_1 = embeddings_1[num_extra:]
                bag_embeddings += extra_embeddings

            # Toggle bag type for next iteration
            make_bag_1 = not make_bag_1

            if len(bag_embeddings) > 0:
                sample_names = [x[0] for x in bag_embeddings]
                sample_labels = [x[1] for x in bag_embeddings]
                sample_split = [x[2] for x in bag_embeddings]
                only_embeddings = [row[3:] for row in bag_embeddings]

                aggregated_embedding = aggregate_embeddings(only_embeddings,
                                                            pooling_method)

                bag_label = int(any(np.array(sample_labels) == 1))
                bag_embeddings_tuple = tuple(map(tuple, only_embeddings))
                bag_samples_tuple = tuple(sample_names)
                bag_key = (bag_embeddings_tuple, len(bag_embeddings),
                           bag_samples_tuple)

                if bag_key not in bag_set:
                    bag_set.add(bag_key)

                    bags.append({
                        "bag_label": bag_label,
                        "split": sample_split[0],
                        "bag_size": len(bag_embeddings),
                        "bag_samples": sample_names,
                        "embedding": aggregated_embedding
                    })
                else:
                    logging.info("A bag was created twice")
        all_bags.extend(bags)
    return all_bags


def bag_random(df_embeddings, bag_sizes, pooling_method, repeats):
    all_bags = []

    for _ in range(repeats):
        available_embeddings = df_embeddings.values.tolist()
        np.random.shuffle(available_embeddings)
        bag_set = set()
        bags = []
        while len(available_embeddings) > 0:
            bag_size = np.random.randint(bag_sizes[0],
                                         bag_sizes[1] + 1)

            bag_embeddings = available_embeddings[:bag_size]
            available_embeddings = available_embeddings[bag_size:]

            sample_names = []
            sample_labels = []
            sample_split = []
            only_embeddings = []
            if len(bag_embeddings) > 0:
                for x in bag_embeddings:
                    sample_names.append(x[0])
                    sample_labels.append(x[1])
                    sample_split.append(x[2])
                    only_embeddings.append(x[3:])

                aggregated_embedding = aggregate_embeddings(only_embeddings,
                                                            pooling_method)

                bag_label = int(any(np.array(sample_labels) == 1))
                bag_embeddings_tuple = tuple(map(tuple, only_embeddings))
                bag_samples_tuple = tuple(sample_names)
                bag_key = (bag_embeddings_tuple,
                           len(bag_embeddings),
                           bag_samples_tuple)

                bag_label = int(any(np.array(sample_labels) == 1))

                if bag_key not in bag_set:
                    bag_set.add(bag_key)

                    bags.append({
                        "bag_label": bag_label,
                        "split": sample_split[0],
                        "bag_size": len(bag_embeddings),
                        "bag_samples": sample_names,
                        "embedding": aggregated_embedding
                    })
                else:
                    logging.info("A bag was created twice")
        all_bags.extend(bags)
    return all_bags


def convert_embedding_to_string(embedding_array):
    return ",".join(map(str, embedding_array))


def transform_bags_for_ludwig(bags):
    trans_bags = []
    for bag in bags:
        trans_bag = bag.copy()
        trans_bag["embedding"] = (
                convert_embedding_to_string(bag["embedding"])
                .replace(",", " ")
        )
        trans_bags.append(trans_bag)

    return trans_bags


def write_csv(output_csv, list_embeddings):
    if not list_embeddings:
        with open(output_csv,
                  mode="w",
                  encoding='utf-8',
                  newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["bag_samples",
                                 "bag_size",
                                 "bag_label",
                                 "split",
                                 "embeddings"])
            logging.info("No valid data found. Empty CSV created.")
        return

    # Determine the format based on the first item's "embedding" field
    first_item = list_embeddings[0]

    with open(output_csv, mode="w", encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Extract common headers
        headers = ["bag_samples", "bag_size", "bag_label", "split"]

        if isinstance(first_item["embedding"], str):
            # Case 1: Embedding is a string
            headers.append("embedding")
            csv_writer.writerow(headers)

            for bag in list_embeddings:
                row = [
                    ",".join(map(str, bag["bag_samples"])),
                    bag["bag_size"],
                    bag["bag_label"],
                    bag["split"],
                    bag["embedding"]
                ]
                csv_writer.writerow(row)

        elif isinstance(first_item["embedding"], np.ndarray):
            # Case 2: Embedding is a NumPy array
            vec_col = [f"vector{i+1}"
                       for i in range(len(first_item["embedding"]))]
            headers.extend(vec_col)
            csv_writer.writerow(headers)

            for bag in list_embeddings:
                row = [
                    ",".join(map(str, bag["bag_samples"])),
                    bag["bag_size"],
                    bag["bag_label"],
                    bag["split"]
                ] + bag["embedding"].tolist()
                csv_writer.writerow(row)

        else:
            raise ValueError("Unknown embedding format. \
                Expected string or NumPy array.")


def bag_processing(embeddings,
                   metadata,
                   pooling_method,
                   balance_enforced=False,
                   bag_sizes=[3, 5],
                   repeats=1,
                   ludwig_format=False,
                   by_sample=False):
    all_bags = []  # Collect all bag data here
    bag_sizes = parse_bag_size(bag_sizes)

    for split in metadata['split'].unique():
        if by_sample:
            embeddings["instance_idx"] = embeddings.groupby("sample_name") \
                                                   .cumcount()
            metadata["instance_idx"] = metadata.groupby("sample_name") \
                                               .cumcount()

            split_embeddings = pd.merge(metadata, embeddings,
                                        on=["sample_name", "instance_idx"],
                                        suffixes=("_meta", "_embed"))

            split_embeddings.drop(columns=["instance_idx"], inplace=True)
            split_embeddings = split_embeddings[
                               split_embeddings["split"] == split
                               ]

            bags = bag_by_sample(split_embeddings,
                                 pooling_method,
                                 bag_sizes)
        else:
            split_metadata = metadata[metadata['split'] == split]
            split_embeddings = pd.merge(split_metadata,
                                        embeddings, on='sample_name')
            if balance_enforced:
                bags = bag_turns(split_embeddings,
                                 bag_sizes,
                                 pooling_method,
                                 repeats)
            else:
                bags = bag_random(split_embeddings,
                                  bag_sizes,
                                  pooling_method,
                                  repeats)
        all_bags.extend(bags)

    if ludwig_format:
        return transform_bags_for_ludwig(all_bags)

    return all_bags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bags")

    parser.add_argument("--embeddings_csv",
                        type=str,
                        required=True,
                        help="The embeddings(Must have 'sample_name' column)")
    parser.add_argument("--metadata_csv",
                        type=str,
                        required=True,
                        help="The metadata(Must contain \
                            'sample_name' and 'label' columns.")
    parser.add_argument("--split_proportions",
                        type=str,
                        default='0.7,0.1,0.2',
                        help="Proportions for train, validation,\
                            and test splits.")
    parser.add_argument("--dataleak",
                        action="store_true",
                        help="Prevents dataleak when splitting the data.")
    parser.add_argument("--balance_enforced",
                        action="store_true",
                        help="Create bags in turns (balanced bags")
    parser.add_argument("--bag_size",
                        type=str,
                        required=True,
                        help="Bag size as a single number (e.g., 4) or \
                        a range (e.g., 3-5).")
    parser.add_argument("--pooling_method",
                        type=str,
                        required=True,
                        help="The method for pooling the embeddings")
    parser.add_argument("--by_sample", action="store_true",
                        help="Create bags using instances of the same sample")
    parser.add_argument("--repeats",
                        type=int,
                        default=1,
                        help="Number of times the entire dataset \
                        can be used to generate bags")
    parser.add_argument("--ludwig_format",
                        action="store_true",
                        help="Prepare CSV file to Ludwig input format")
    parser.add_argument("--output_csv",
                        required=True,
                        help="Path to the output CSV file")

    args = parser.parse_args()

    embeddings_data = load_csv(args.embeddings_csv)
    metadata_csv = load_csv(args.metadata_csv)

    if "split" not in metadata_csv:
        metadata_csv = split_data(
            metadata_csv,
            split_proportions=args.split_proportions,
            dataleak=args.dataleak
        )
    processed_embeddings = bag_processing(
        embeddings_data,
        metadata_csv,
        args.pooling_method,
        args.balance_enforced,
        args.bag_size,
        args.repeats,
        args.ludwig_format,
        args.by_sample
    )

    write_csv(args.output_csv, processed_embeddings)
