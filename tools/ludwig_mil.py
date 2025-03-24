"""
Module for generating bags of embeddings from metadata and embedding data,
with support for dataset splitting, pooling, and class balancing.

This module offers tools to:
1. **Load and parse CSV files** for embeddings and metadata.
2. **Partition the dataset** into train, validation, and test splits using
specified proportions or data leakage prevention.
3. **Create bags of embeddings** via methods like random sampling or balanced
class alternation, with configurable pooling techniques.
4. **Format embeddings** for Ludwig compatibility or export them to a CSV file.

Key Functions:
- `parse_bag_size`: Interprets bag size input as a single integer or
a range (e.g., '3-5').
- `load_csv`: Reads a CSV file into a pandas DataFrame,
supporting optional chunking.
- `split_data`: Divides data into train, validation, and test sets
based on proportions or leakage control.
- `bag_turns`: Produces balanced bags by alternating between classes
across repeated runs.
- `bag_random`: Generates random bags, with optional balancing
over multiple iterations.
- `aggregate_embeddings`: Applies pooling methods
(e.g., max, mean, attention) to combine embeddings.
- `transform_bags_for_ludwig`: Converts bags into a Ludwig-compatible format.
- `write_csv`: Saves processed bags to a CSV file with
flexible embedding output.

Command-Line Interface:
The module includes a CLI with these arguments:
- `--embeddings_csv`: Path to the embeddings CSV
(requires 'sample_name' column).
- `--metadata_csv`: Path to the metadata CSV
(requires 'sample_name' and 'label' columns).
- `--split_proportions`: Train/validation/test split ratios
(e.g., '0.7,0.1,0.2'; default: '0.7,0.1,0.2').
- `--dataleak`: Flag to avoid data leakage during splitting.
- `--balance_enforced`: Flag to enforce balanced bagging by
alternating classes.
- `--bag_size`: Bag size, either a fixed number (e.g., '4')
or range (e.g., '3-5').
- `--random_seed`: Integer seed for reproducible random operations.
- `--pooling_method`: Embedding aggregation method
(e.g., 'max_pooling', 'mean_pooling').
- `--repeats`: Number of times to repeat bagging (default: 1).
- `--ludwig_format`: Flag to output embeddings as strings for Ludwig.
- `--by_sample`: Optional splits (e.g., '0,1') for bagging within samples;
defaults to random/balanced if omitted or invalid.
- `--output_csv`: Path for the output CSV file.
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


def parse_by_sample(value):
    """Parses by_sample argument to handle comma-separated list of splits."""
    try:
        value = str(value)
        splits = [int(x) for x in value.split(",")]
        valid_splits = {0, 1, 2}
        if not all(x in valid_splits for x in splits):
            logging.warning(f"Invalid splits in by_sample: {splits}.\
                            Must be in {valid_splits}.\
                            Defaulting to random/balanced bagging."
                            )
            return None
        return splits
    except (ValueError, AttributeError):
        logging.warning(f"Could not parse by_sample value: {value}.\
                        Defaulting to random/balanced bagging.")
        return None


def load_csv(file_path, chunksize=None):
    """Loads a CSV file, optionally in chunks."""
    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize)
    return pd.read_csv(file_path)


def str_array_split(split_proportions):
    """Converts split proportions string into a list of floats."""
    split_array = [float(p) for p in split_proportions.split(",")]
    if len(split_array) == 2:
        split_array.insert(1, 0.0)
    return split_array


def split_sizes(num_samples, proportions):
    """Calculates sizes of splits based on proportions."""
    sizes = [int(p * num_samples) for p in proportions]
    sizes[-1] = num_samples - sum(sizes[:-1])
    return sizes


def split_data(metadata, split_proportions, dataleak=False):
    """Splits data into train, validation, and test sets."""
    proportions = str_array_split(split_proportions)
    if dataleak:
        list_samples = metadata["sample_name"].unique()
    else:
        list_samples = metadata["sample_name"].values

    num_samples = len(list_samples)
    sizes = split_sizes(num_samples, proportions)

    shuffled_samples = np.random.permutation(list_samples)

    split_values = np.zeros(num_samples, dtype=int)
    if sizes[1] > 0:  # Check if validation size is non-zero
        split_values[sizes[0]:sizes[0] + sizes[1]] = 1
    split_values[sizes[0] + sizes[1]:] = 2

    split_series = pd.Series(split_values, index=shuffled_samples)
    metadata["split"] = metadata["sample_name"].map(split_series)
    return metadata


def attention_pooling(embeddings, use_gpu=torch.cuda.is_available()):
    """Performs attention-based pooling on embeddings."""
    device = 'cuda' if use_gpu else 'cpu'
    tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    weights = nn.Softmax(dim=0)(nn.Linear(tensor.shape[1], 1)
                                .to(device)(tensor))
    pooled_embedding = torch.sum(weights * tensor, dim=0).cpu().numpy()
    return pooled_embedding


def gated_pooling(embeddings, use_gpu=torch.cuda.is_available()):
    """Performs gated pooling on embeddings."""
    device = 'cuda' if use_gpu else 'cpu'
    tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    gate = nn.Sigmoid()(nn.Linear(tensor.shape[1], tensor.shape[1])
                        .to(device)(tensor))
    pooled_embedding = torch.sum(gate * tensor, dim=0).cpu().numpy()
    return pooled_embedding


def aggregate_embeddings(embeddings, pooling_method, use_gpu=False):
    """Aggregates embeddings using the specified pooling method."""
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
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    if pooling_method == "geometric_mean_pooling":
        return np.exp(np.mean(np.log(np.clip(embeddings, 1e-10, None)),
                              axis=0))
    if pooling_method == "first_embedding":
        return embeddings[0]
    if pooling_method == "last_embedding":
        return embeddings[-1]
    if pooling_method == "attention_pooling":
        return attention_pooling(embeddings, use_gpu)
    if pooling_method == "gated_pooling":
        return gated_pooling(embeddings, use_gpu)
    raise ValueError(f"Unknown pooling method: {pooling_method}")


def bag_by_sample(df, pooling_method, bag_size, use_gpu=False):
    """Creates bags within each sample."""
    all_bags = []
    non_embedding_cols = {"sample_name", "label", "split"}
    embedding_cols = [col for col in df.columns
                      if col not in non_embedding_cols]

    for _, group in df.groupby("sample_name"):
        embeddings = group[embedding_cols].values
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
                                                         pooling_method,
                                                         use_gpu)
            bag_label = int(any(bag_labels == 1))
            all_bags.append({
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(bag_sample_names),
                "bag_samples": list(bag_sample_names),
                "embedding": aggregated_embeddings
            })

    return all_bags


def bag_turns(df, bag_sizes, pooling_method, repeats, use_gpu=False):
    """Creates balanced bags by alternating between classes."""
    all_bags = []
    non_embedding_cols = {"sample_name", "label", "split"}
    embedding_cols = [col for col in df.columns
                      if col not in non_embedding_cols]
    data = df[["sample_name", "label", "split"] + embedding_cols].to_numpy()

    for _ in range(repeats):
        indices_0 = np.where(data[:, 1] == 0)[0]
        indices_1 = np.where(data[:, 1] == 1)[0]
        np.random.shuffle(indices_0)
        np.random.shuffle(indices_1)

        make_bag_1 = True
        bags = []
        bag_set = set()

        while len(indices_0) > 0 or len(indices_1) > 0:
            bag_size = np.random.randint(bag_sizes[0], bag_sizes[1] + 1)

            if make_bag_1 and len(indices_1) > 0:
                num_1_samples = min(np.random.randint(1, bag_size + 1),
                                    len(indices_1))
                selected_indices_1 = indices_1[:num_1_samples]
                indices_1 = indices_1[num_1_samples:]
            else:
                selected_indices_1 = []

            num_0_samples = min(bag_size - len(selected_indices_1),
                                len(indices_0))
            selected_indices_0 = indices_0[:num_0_samples]
            indices_0 = indices_0[num_0_samples:]

            bag_indices = np.concatenate([selected_indices_0,
                                          selected_indices_1])
            bag_data = data[bag_indices]

            if len(bag_data) < bag_size and len(indices_1) > 0:
                num_extra = min(bag_size - len(bag_data), len(indices_1))
                extra_indices = indices_1[:num_extra]
                indices_1 = indices_1[num_extra:]
                bag_data = np.vstack([bag_data, data[extra_indices]])

            make_bag_1 = not make_bag_1

            if len(bag_data) > 0:
                sample_names = bag_data[:, 0]
                sample_labels = bag_data[:, 1]
                sample_split = bag_data[:, 2]
                only_embeddings = bag_data[:, 3:]

                aggregated_embedding = aggregate_embeddings(only_embeddings,
                                                            pooling_method,
                                                            use_gpu)

                bag_label = int(any(sample_labels == 1))
                bag_embeddings_tuple = tuple(map(tuple, only_embeddings))
                bag_samples_tuple = tuple(sample_names)
                bag_key = (bag_embeddings_tuple, len(bag_data),
                           bag_samples_tuple)

                if bag_key not in bag_set:
                    bag_set.add(bag_key)
                    bags.append({
                        "bag_label": bag_label,
                        "split": sample_split[0],
                        "bag_size": len(bag_data),
                        "bag_samples": list(sample_names),
                        "embedding": aggregated_embedding
                    })
                else:
                    logging.info("A bag was created twice")
        all_bags.extend(bags)
    return all_bags


def bag_random(df, bag_sizes, pooling_method, repeats, use_gpu=False):
    """Creates random bags from the dataset."""
    all_bags = []
    non_embedding_cols = {"sample_name", "label", "split"}
    embedding_cols = [col for col in df.columns
                      if col not in non_embedding_cols]
    data = df[["sample_name", "label", "split"] + embedding_cols].to_numpy()

    for _ in range(repeats):
        np.random.shuffle(data)
        idx = 0
        bag_set = set()
        while idx < len(data):
            bag_size = np.random.randint(bag_sizes[0], bag_sizes[1] + 1)
            end_idx = min(idx + bag_size, len(data))
            bag_data = data[idx:end_idx]

            sample_names = bag_data[:, 0]
            sample_labels = bag_data[:, 1]
            sample_split = bag_data[:, 2]
            only_embeddings = bag_data[:, 3:]

            aggregated_embedding = aggregate_embeddings(only_embeddings,
                                                        pooling_method,
                                                        use_gpu)

            bag_label = int(any(sample_labels == 1))
            bag_embeddings_tuple = tuple(map(tuple, only_embeddings))
            bag_samples_tuple = tuple(sample_names)
            bag_key = (bag_embeddings_tuple, len(bag_data), bag_samples_tuple)

            if bag_key not in bag_set:
                bag_set.add(bag_key)
                all_bags.append({
                    "bag_label": bag_label,
                    "split": sample_split[0],
                    "bag_size": len(bag_data),
                    "bag_samples": list(sample_names),
                    "embedding": aggregated_embedding
                })
            else:
                logging.info("A bag was created twice")
            idx = end_idx
    return all_bags


def convert_embedding_to_string(embedding_array):
    """Converts an embedding array to a space-separated string."""
    return " ".join(map(str, embedding_array))


def transform_bags_for_ludwig(bags):
    """Transforms bags into Ludwig-compatible format."""
    trans_bags = []
    for bag in bags:
        trans_bag = bag.copy()
        trans_bag["embedding"] = convert_embedding_to_string(bag["embedding"])
        trans_bags.append(trans_bag)
    return trans_bags


def write_csv(output_csv, list_embeddings, chunk_size=10000):
    """Writes bags to a CSV file in chunks."""
    if not list_embeddings:
        with open(output_csv,
                  mode="w",
                  encoding='utf-8',
                  newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["bag_samples",
                                 "bag_size",
                                 "bag_label",
                                 "split"])
            logging.info("No valid data found. Empty CSV created.")
        return

    first_item = list_embeddings[0]
    with open(output_csv, mode="w", encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        headers = ["bag_samples", "bag_size", "bag_label", "split"]

        if isinstance(first_item["embedding"], str):
            headers.append("embedding")
        elif isinstance(first_item["embedding"], np.ndarray):
            embedding_size = len(first_item["embedding"])
            headers.extend([f"vector{i+1}" for i in range(embedding_size)])
        else:
            raise ValueError("Expected string or NumPy array.")

        csv_writer.writerow(headers)

        for i in range(0, len(list_embeddings), chunk_size):
            chunk = list_embeddings[i:i + chunk_size]
            for bag in chunk:
                row = [",".join(map(str,
                       bag["bag_samples"])),
                       bag["bag_size"],
                       bag["bag_label"],
                       bag["split"]]
                if isinstance(bag["embedding"], str):
                    row.append(bag["embedding"])
                else:
                    row.extend(bag["embedding"].tolist())
                csv_writer.writerow(row)


def bag_processing(embeddings_path,
                   metadata,
                   pooling_method,
                   balance_enforced=False,
                   bag_sizes=[3, 5],
                   repeats=1,
                   ludwig_format=False,
                   by_sample=None,
                   use_gpu=False):
    """Processes embeddings and metadata to
    create bags, handling chunked loading."""
    all_bags = []
    bag_sizes = parse_bag_size(bag_sizes)

    # Ensure metadata has required columns
    required_cols = {"sample_name", "label"}
    if not required_cols.issubset(metadata.columns):
        missing = required_cols - set(metadata.columns)
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    # Process each split
    for split in metadata['split'].unique():
        split_metadata = metadata[metadata['split'] == split]
        split_sample_names = split_metadata['sample_name'].unique()

        if by_sample is not None and split in by_sample:
            # Process each sample_name individually for bag_by_sample
            for sample_name in split_sample_names:
                sample_metadata = split_metadata[
                                  split_metadata['sample_name'] == sample_name]
                sample_chunks = []

                # Collect all embeddings for this sample_name across chunks
                for chunk in pd.read_csv(embeddings_path, chunksize=100000):
                    chunk_filtered = chunk[chunk['sample_name'] == sample_name]
                    if not chunk_filtered.empty:
                        sample_chunks.append(chunk_filtered)

                if sample_chunks:
                    sample_embeddings = pd.concat(sample_chunks)
                    logging.info(
                        f"Loaded {len(sample_embeddings)} embeddings for "
                        f"sample_name: {sample_name} in split {split}")
                    sample_df = pd.merge(sample_metadata,
                                         sample_embeddings,
                                         on='sample_name')
                    if sample_df.empty:
                        logging.warning(f"No matching embeddings found for "
                                        f"sample_name: {sample_name} "
                                        f"in split {split}")
                        continue
                    bags = bag_by_sample(sample_df,
                                         pooling_method,
                                         bag_sizes,
                                         use_gpu)
                    all_bags.extend(bags)
                else:
                    logging.warning(f"No embeddings found for sample_name: "
                                    f"{sample_name} in split {split}")
        else:
            # Collect all embeddings for the entire split
            split_embeddings_chunks = []
            for chunk in pd.read_csv(embeddings_path, chunksize=100000):
                chunk_filtered = chunk[chunk['sample_name']
                                       .isin(split_sample_names)]
                if not chunk_filtered.empty:
                    split_embeddings_chunks.append(chunk_filtered)

            if split_embeddings_chunks:
                split_embeddings = pd.concat(split_embeddings_chunks)
                logging.info(f"Loaded {len(split_embeddings)} embeddings for "
                             f"split {split}")
                split_df = pd.merge(split_metadata,
                                    split_embeddings,
                                    on='sample_name')
                if split_df.empty:
                    logging.warning(f"No matching embeddings found for "
                                    f"split {split}")
                    continue
                if balance_enforced:
                    bags = bag_turns(split_df,
                                     bag_sizes,
                                     pooling_method,
                                     repeats,
                                     use_gpu)
                else:
                    bags = bag_random(split_df,
                                      bag_sizes,
                                      pooling_method,
                                      repeats,
                                      use_gpu)
                all_bags.extend(bags)
            else:
                logging.warning(f"No embeddings found for split {split}")

    if ludwig_format:
        return transform_bags_for_ludwig(all_bags)
    return all_bags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                     description="Create bags from embeddings and metadata")
    parser.add_argument(
        "--embeddings_csv",
        type=str,
        required=True,
        help="Path to embeddings CSV (must have 'sample_name' column)"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to metadata CSV ('sample_name' and 'label' columns)"
    )
    parser.add_argument(
        "--split_proportions",
        type=str,
        default='0.7,0.1,0.2',
        help="Proportions for train, validation, and test splits \
            (e.g., '0.7,0.1,0.2')"
    )
    parser.add_argument(
        "--dataleak",
        action="store_true",
        help="Prevents data leakage when splitting"
    )
    parser.add_argument(
        "--balance_enforced",
        action="store_true",
        help="Enforce balanced bagging by alternating classes"
    )
    parser.add_argument(
        "--bag_size",
        type=str,
        required=True,
        help="Bag size as a single number (e.g., '4') or range (e.g., '3-5')"
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        required=True,
        help="Pooling method (e.g., 'mean_pooling', 'attention_pooling')"
    )
    parser.add_argument(
        "--by_sample",
        type=parse_by_sample,
        default=None,
        help="Comma-separated splits (e.g., '0,1') to bag by sample"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat bagging process"
    )
    parser.add_argument(
        "--ludwig_format",
        action="store_true",
        help="Output in Ludwig-compatible format"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If set, ensures consistent \
            results across runs."
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
            # Optional: for full reproducibility, but may impact performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Load and split metadata if necessary
    metadata_csv = load_csv(args.metadata_csv)
    if "split" not in metadata_csv.columns:
        metadata_csv = split_data(metadata_csv,
                                  split_proportions=args.split_proportions,
                                  dataleak=args.dataleak)

    # Process embeddings and create bags
    processed_embeddings = bag_processing(
        args.embeddings_csv,
        metadata_csv,
        args.pooling_method,
        args.balance_enforced,
        args.bag_size,
        args.repeats,
        args.ludwig_format,
        args.by_sample
    )

    # Write results to CSV
    write_csv(args.output_csv, processed_embeddings)
