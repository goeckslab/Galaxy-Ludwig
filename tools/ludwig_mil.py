import argparse
import pandas as pd
import numpy as np
import random
import os 

def load_csv(file_path):
    return pd.read_csv(file_path)

def str_array_split(split_proportions):
    array_split_proportion = split_proportions.split(",")
    return array_split_proportion

def UniqueSamples(load_data):
    """
    Retrieve unique samples from the data.

    Args:
        load_data (pd.DataFrame): The dataset to process.

    Returns:
        tuple: Unique samples and their count.
    """
    unique_samples = load_data["sample_name"].unique()
    num_samples = len(unique_samples)
    print(f"This is the length of unique samples: {num_samples}", flush=True)
    return unique_samples, num_samples

def SplitData(unique_samples, num_samples, data, proportions=[0.7,0.1,0.2]):
    """
    Create train, validation, and test splits based on unique samples.

    Args:
        unique_samples (np.array): Array of unique sample names.
        num_samples (int): Total number of unique samples.
        data (pd.DataFrame): The dataset to split.

    Returns:
        tuple: Updated DataFrame with split column, and split samples.
    """

    train_count = int(proportions[0] * num_samples)
    val_count = int(proportions[1] * num_samples)

    # Randomly shuffle the samples
    shuffled_samples = np.random.permutation(unique_samples)

    # Assign splits
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:train_count + val_count]
    test_samples = shuffled_samples[train_count + val_count:]

    # Map samples to splits
    sample_to_split = {sample: 0 for sample in train_samples}  # Train: 0
    sample_to_split.update({sample: 1 for sample in val_samples})  # Validation: 1
    sample_to_split.update({sample: 2 for sample in test_samples})  # Test: 2 Add split column to the DataFrame
    data["split"] = data["sample_name"].map(sample_to_split)

    return data, train_samples, val_samples, test_samples

def aggregate_embeddings_with_max_pooling(embeddings):
    return np.max(embeddings, axis=0)

def convert_embedding_to_string(embedding):
    return ",".join(map(str, embedding))

def create_bags_from_split(split_data, split, repeats, bag_size, balance_enforce):
    """
    Create bags from split data, ensuring balance if required.

    Args:
        split_data (pd.DataFrame): Data for the current split.
        split (int): Current split (0, 1, or 2).
        repeats (int): Number of repetitions for bag creation.
        bag_size (int): Size of each bag.
        balance_enforce (bool): Whether to enforce class balance in the bags.

    Returns:
        list: List of created bags.
    """
    print("Inside create bags from split function", flush=True)
    bags = []
    bag_set = set()

    if balance_enforce:
        grouped_data = split_data.groupby("label")
        images_0 = grouped_data.get_group(0) if 0 in grouped_data.groups else pd.DataFrame()
        images_1 = grouped_data.get_group(1) if 1 in grouped_data.groups else pd.DataFrame()
    else:
        images_0 = split_data[split_data["label"] == 0]
        images_1 = split_data[split_data["label"] == 1]

    for _ in range(repeats):
        images_0 = images_0.sample(frac=1).reset_index(drop=True)  # Shuffle
        images_1 = images_1.sample(frac=1).reset_index(drop=True)  # Shuffle

        while len(images_0) + len(images_1) > 0:
            make_bag_1 = len(images_1) > 0 and (not balance_enforce or random.random() < 0.5)

            if make_bag_1:
                selected_images_1 = images_1.iloc[:bag_size]
                images_1 = images_1.iloc[bag_size:]
                selected_images_0 = images_0.iloc[:max(0, bag_size - len(selected_images_1))]
                images_0 = images_0.iloc[len(selected_images_0):]
            else:
                selected_images_0 = images_0.iloc[:bag_size]
                images_0 = images_0.iloc[bag_size:]
                selected_images_1 = images_1.iloc[:max(0, bag_size - len(selected_images_0))]
                images_1 = images_1.iloc[len(selected_images_1):]

            bag_images = pd.concat([selected_images_0, selected_images_1])

            if len(bag_images) > 0:
                bag_embeddings = bag_images.iloc[:, 2:].to_numpy()
                bag_label = int(any(bag_images["label"] == 1))
                aggregated_embedding = aggregate_embeddings_with_max_pooling(bag_embeddings)
                embedding_string = convert_embedding_to_string(aggregated_embedding)
                bag_samples = bag_images["sample_name"].tolist()

                bag_key = (tuple(map(tuple, bag_embeddings)), len(bag_images), tuple(bag_samples))

                if bag_key not in bag_set:
                    bag_set.add(bag_key)
                    bags.append({
                        "embedding": embedding_string,
                        "bag_label": bag_label,
                        "split": split,
                        "bag_size": len(bag_images),
                        "bag_samples": bag_samples
                    })
                else:
                    print("A bag was created twice", flush=True)

    return bags

def bag_size(size_numbers):
    if isinstance(size_numbers, str) and '-' in size_numbers:
        min_size, max_size = map(int, size_numbers.split('-'))
    else:
        min_size = max_size = int(size_numbers)

    return min_size, max_size

def process_csv(embeddings_csv, metadata_csv, bag_size, balance_enforce, pooling_method, split_proportions, dataleak):
    """
    Process CSV files to create bags of data based on the specified parameters.

    Args:
        embeddings_csv (str): Path to the embeddings CSV file.
        metadata_csv (str): Path to the metadata CSV file.
        bag_size (str or int): The size of the bags to create. Can be an integer or a range (e.g., "3-5").
        balance_enforce (bool): Whether to enforce class balance in the bags.
        pooling_method (str): The pooling method to apply (e.g., "maxsoft", "meansoft").

    Returns:
        None: Prints the processed output for demonstration.
    """
    embeddings_data = load_csv(embeddings_csv)
    metadata = load_csv(metadata_csv)
    split_proportions = str_array_split(split_proportions)
    bag_min, bag_max = bag_size(bag_size)

    print(f"Loaded {len(embeddings_data)} records from embeddings file.")
    print(f"Loaded {len(metadata)} records from metadata file.")
    print(f"Bag size range: {bag_min}-{bag_max}")
    print(f"Balance enforcement: {balance_enforce}")

    if embeddings_data.shape[1] < 2:
        raise ValueError("Embeddings CSV file must have at least two columns: sample name and vectors.")
    if not all(col in metadata.columns for col in ["sample_name", "label"]):
        raise ValueError("Metadata CSV file must contain 'sample_name' and 'label' columns.")

    if "split" not in metadata.columns and dataleak:
        unique_samples, num_samples = UniqueSamples(metadata)
        metadata, _, _, _ = SplitData(unique_samples, num_samples, metadata)
    elif "split" not in metadata.columns:
        metadata, _, _, _ = SplitData(metadata)

    merged_data = pd.merge(
        metadata, embeddings_data, left_on="sample_name", right_on=embeddings_data.columns[0]
    )

    all_bags = []

    for split in range(3):
        split_data = merged_data[merged_data["split"] == split]
        bags = create_bags_from_split(
            split_data, split, repeats=5, bag_size=max_size, balance_enforce=balance_enforce
        )
        all_bags.extend(bags)
        print(f"Processed {len(bags)} bags for split {split}.")

    print(f"Total bags created: {len(all_bags)}")
    #print("Example bag:", all_bags[0] if all_bags else "No bags created.")
    all_bags_df = pd.DataFrame(all_bags)
    output_file = "all_bags.csv"
    all_bags_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to create bags of data.")

    # Define arguments
    parser.add_argument(
        "--embeddings_csv", 
        type=str,
        required=True,
        help="The embeddings in CSV file."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="The metadata in CSV file."
    )
    parser.add_argument(
        "--bag_size",
        type=str,
        required=True,
        help="Bag size as an integer or range (e.g., '3-5')."
    )
    parser.add_argument(
        "--balance_enforce",
        action="store_true",
        help="Enforce class balance in the bags."
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="maxsoft",
        choices=["maxsoft", "meansoft", "maxpool", "meanpool", "sum", "minpool"],
        help=(
            "Pooling method to apply. Options: "
            "'maxsoft' (softmax-weighted max pooling), "
            "'meansoft' (softmax-weighted mean pooling), "
            "'maxpool' (standard max pooling), "
            "'meanpool' (standard mean pooling), "
            "'sum' (sum pooling), "
            "'minpool' (minimum pooling). Default: maxsoft."
        )
    )
    parser.add_argument(
        "--split_proportions",
        type=str,
        default='0.7,0.1,0.2',
        help="Comma-separated proportions for train, validation, and test splits (e.g., '0.7,0.2,0.1'). If not provided, use metadata CSV."
    )
    parser.add_argument(
        "--dataleak",
        action="store_true",
        help="Prevents dataleak when spliting the data"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the process_csv function with parsed arguments
    process_csv(
        embeddings_csv=args.embeddings_csv,
        metadata_csv=args.metadata_csv,
        bag_size=args.bag_size,
        balance_enforce=args.balance_enforce,
        pooling_method=args.pooling_method,
        split_proportions=args.split_proportions,
        dataleak=args.dataleak,
    )
