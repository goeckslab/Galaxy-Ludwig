"""
This module provides functionality to extract
image embeddings using a specified
pretrained model from the torchvision library.

It includes functions to:
- Load and process images from a ZIP file.
- Apply model-specific preprocessing and transformations.
- Extract embeddings using various models.
- Save the resulting embeddings into a CSV file.

Modules required:
- argparse: For command-line argument parsing.
- os, csv, zipfile: For file handling (ZIP file extraction, CSV writing).
- inspect: For inspecting function signatures and models.
- torch, torchvision: For loading and
using pretrained models to extract embeddings.
- PIL, cv2: For image processing tasks
such as resizing, normalization, and conversion.
"""

import argparse
import csv
import inspect
import logging
import os
import tempfile
import zipfile
import re
from inspect import signature

from PIL import Image

import cv2

import torch

import torchvision.models as models
from torchvision import transforms

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",
    # so write to a file
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)

# Only include the model constructors that are actual models in torchvision
AVAILABLE_MODELS = {
    name: getattr(models, name)
    for name in dir(models)
    if callable(getattr(models, name)) and
    "weights" in signature(getattr(models, name)).parameters
}

# Define the default resize and normalization settings for models
MODEL_DEFAULTS = {
    # Default normalization (ImageNet)
    "default": {
        "resize": (224, 224),
        "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    },

    # Models using (224, 224) resize and ImageNet normalization
    "efficientnet_b1": {"resize": (240, 240)},
    "efficientnet_b2": {"resize": (260, 260)},
    "efficientnet_b3": {"resize": (300, 300)},
    "efficientnet_b4": {"resize": (380, 380)},
    "efficientnet_b5": {"resize": (456, 456)},
    "efficientnet_b6": {"resize": (528, 528)},
    "efficientnet_b7": {"resize": (600, 600)},
    "inception_v3": {"resize": (299, 299)},
    "swin_b": {
        "resize": (224, 224),
        "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])
    },
    "swin_s": {
        "resize": (224, 224),
        "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])
    },
    "swin_t": {
        "resize": (224, 224),
        "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])
    },
    "vit_b_16": {
        "resize": (224, 224),
        "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    },
    "vit_b_32": {
        "resize": (224, 224),
        "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    },
}

# Ensure all models have normalization applied (if not defined)
for model, settings in MODEL_DEFAULTS.items():
    if "normalize" not in settings:
        settings["normalize"] = (
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )


def extract_zip(zip_file):
    """Extracts a ZIP file while preserving the folder structure, only for image files."""
    # Define known image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svs',
                        'tif', 'tiff')

    output_dir = tempfile.mkdtemp(prefix="extracted_zip_")
    try:
        sample_name = []
        file_list = []
        zip_filename = os.path.splitext(os.path.basename(zip_file))[0]

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            for file in zip_ref.namelist():
                if not file.endswith('/') and file.lower().endswith(image_extensions):
                    try:
                        pattern = re.escape(zip_filename) + r"/(.*)"
                        match = re.search(pattern, file)
                        if match:
                            file_path = match.group(1)
                            file_list.append(file)
                            sample_name.append(file_path)
                        else:
                            raise ValueError(f"File path does not match the expected format: {file}")
                    except (re.error, ValueError) as match_error:
                        # Handle the error if the regex fails or the match is not found
                        logging.error(f"Error processing file '{file}': {match_error}")
        logging.info(f"ZIP extracted to: {output_dir}")
        return output_dir, file_list, sample_name
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Invalid ZIP file.") from exc


def load_model(model_name, device):
    """Loads a specified torchvision model and
    modifies it for feature extraction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: "
            f"{list(AVAILABLE_MODELS.keys())}"
        )
    if "weights" in inspect.signature(AVAILABLE_MODELS[model_name]).parameters:
        model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)
    else:
        model = AVAILABLE_MODELS[model_name]().to(device)
    logging.info("Model loaded")

    # Remove classification head dynamically
    if hasattr(model, 'fc'):  # ResNet, EfficientNet, etc.
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):  # MobileNet, DenseNet, etc.
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'head'):  # Vision Transformer
        model.head = torch.nn.Identity()

    model.eval()
    return model


def process_image(image_path, transform, device, transform_type="rgb"):
    """Loads and transforms an image with different preprocessing options."""
    try:
        image = Image.open(image_path)

        # Apply the selected transformation type
        if transform_type == "grayscale":
            image = image.convert("L")  # Convert to grayscale
        elif transform_type == "rgba_to_rgb":
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        elif transform_type == "clahe":
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)  # Apply CLAHE
            image = Image.fromarray(image).convert("RGB")
        elif transform_type == "edges":
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(image, threshold1=100, threshold2=200)
            image = Image.fromarray(edges).convert("RGB")
        else:
            image = image.convert("RGB")  # Default is RGB conversion
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.warning("Skipping %s: %s", image_path, e)
        return None


def write_csv(output_csv, list_embeddings, sample_names):
    """Writes embeddings and sample names to a CSV file."""
    with open(output_csv, mode="w", encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if list_embeddings:
            header = ["sample_name"] + [f"vector{i + 1}" for i in range(len(list_embeddings[0]))]
            csv_writer.writerow(header)

            # Write the rows, combining sample names and embeddings
            for sample_name, embedding in zip(sample_names, list_embeddings):
                row = [sample_name] + embedding
                csv_writer.writerow(row)
            logging.info("CSV created")
        else:
            # If no embeddings, create an empty CSV with only sample names
            csv_writer.writerow(["sample_name"])
            logging.info("No valid images found. Empty CSV created.")


def extract_embeddings(model_name,
                       apply_normalization,
                       output_dir,
                       file_list,
                       transform_type="rgb"):
    """Extracts embeddings from images using the specified model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_model = load_model(model_name, device)
    model_settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = model_settings["resize"]

    # Define the transformation pipeline
    if apply_normalization:
        normalize = model_settings.get("normalize")
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    list_embeddings = []
    with torch.no_grad():
        for file in file_list:
            file = os.path.join(output_dir, file)
            input_tensor = process_image(file,
                                         transform,
                                         device,
                                         transform_type)
            if input_tensor is None:
                continue
            embedding = use_model(input_tensor).squeeze().cpu().numpy()
            list_embeddings.append(embedding.tolist())
    return list_embeddings


def main(zip_file,
         output_csv,
         model_name,
         apply_normalization=False,
         transform_type="rgb"):
    """Main entry point for processing
    the zip file and extracting embeddings."""
    output_dir, file_list, sample_names = extract_zip(zip_file)
    logging.info("ZIP extracted")

    list_embeddings = extract_embeddings(
        model_name, apply_normalization, output_dir, 
        file_list, transform_type)
    logging.info("Embedding extracted")

    write_csv(output_csv, list_embeddings, sample_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings.")

    parser.add_argument('--zip_file',
                        required=True,
                        help="Path to the ZIP file containing images.")
    parser.add_argument('--model_name',
                        required=True,
                        choices=AVAILABLE_MODELS.keys(),
                        help="Model for embedding extraction.")
    parser.add_argument('--normalize',
                        action="store_true",
                        help="Whether to apply normalization.")
    parser.add_argument('--transform_type',
                        required=True,
                        help="Image transformation type.")
    parser.add_argument("--output_csv",
                        required=True,
                        help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.zip_file,
         args.output_csv,
         args.model_name,
         args.normalize,
         args.transform_type)
