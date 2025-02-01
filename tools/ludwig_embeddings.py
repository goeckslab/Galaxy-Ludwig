import os
import argparse
import logging
import inspect
import zipfile
import csv
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from inspect import signature

# Configure logging
logging.basicConfig(
    filename="/tmp/ludwig_embeddings.log",  # Galaxy tools usually don't have stdout, so write to a file
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
# Only include the model constructors that are actual models in torchvision
AVAILABLE_MODELS = {
    name: getattr(models, name)
    for name in dir(models)
    if callable(getattr(models, name)) and "weights" in signature(getattr(models, name)).parameters
}

# Define the default resize and normalization settings for models
MODEL_DEFAULTS = {
    # Default normalization (ImageNet)
    "default": {"resize": (224, 224), "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])},

    # Models using (224, 224) resize and ImageNet normalization
    "efficientnet_b1": {"resize": (240, 240)},
    "efficientnet_b2": {"resize": (260, 260)}, "efficientnet_b3": {"resize": (300, 300)},
    "efficientnet_b4": {"resize": (380, 380)},
    "efficientnet_b5": {"resize": (456, 456)},
    "efficientnet_b6": {"resize": (528, 528)},
    "efficientnet_b7": {"resize": (600, 600)},
    "inception_v3": {"resize": (299, 299)},
    "swin_b": {"resize": (224, 224), "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])},
    "swin_s": {"resize": (224, 224), "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])},
    "swin_t": {"resize": (224, 224), "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])},
    "vit_b_16": {"resize": (224, 224), "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])},
    "vit_b_32": {"resize": (224, 224), "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])},
}

# Ensure all models have normalization applied (if not defined)
for model in MODEL_DEFAULTS:
    if "normalize" not in MODEL_DEFAULTS[model]:
        MODEL_DEFAULTS[model]["normalize"] = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def extract_zip(zip_file):
    """Extracts a ZIP file into a given directory."""
    output_dir = os.path.splitext(zip_file)[0]
    os.makedirs(output_dir, exist_ok=True)
    try:
        file_list = []
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            file_list = zip_ref.namelist()
        logging.info("zip extracted")
        return output_dir, file_list
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid ZIP file.")
    except Exception as e:
        raise RuntimeError(f"Error extracting ZIP file: {e}")

def load_model(model_name, device):
    """Loads a specified torchvision model and modifies it for feature extraction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
    if "weights" in inspect.signature(AVAILABLE_MODELS[model_name]).parameters:
        model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)
    else:
        model = AVAILABLE_MODELS[model_name]().to(device)
    logging.info("model loaded")
    # Remove classification head dynamically
    if hasattr(model, 'fc'):  # ResNet, EfficientNet, etc.
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):  # MobileNet, DenseNet, etc.
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'head'):  # Vision Transformer
        model.head = torch.nn.Identity()

    model.eval()
    return model

def process_image(image_path, transform, device):
    """Loads and transforms an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.warning(f"Skipping {image_path}: {e}")
        return None

def write_csv(output_csv, list_embeddings):
    with open(output_csv, mode="w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if list_embeddings:
            header = ["sample_name"] + [f"vector{i+1}" for i in range(len(list_embeddings[0]) - 1)]
            csv_writer.writerow(header)
            csv_writer.writerows(list_embeddings)
            logging.info("csv created")
        else:
            csv_writer.writerow(["sample_name"])
            print("No valid images found. Empty CSV created.")

def extract_embeddings(model_name, apply_normalization, output_dir, file_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_model = load_model(model_name, device)
    model_settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = model_settings["resize"]
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
            input_tensor = process_image(file, transform, device)
            if input_tensor is None:
                continue
            embedding = use_model(input_tensor).squeeze().cpu().numpy()
            list_embeddings.append([os.path.basename(file)] + embedding.tolist())
    return list_embeddings 

def main(zip_file, output_csv, model_name, apply_normalization=False):
    output_dir, file_list = extract_zip(zip_file)
    logging.info("zip extracted")

    list_embeddings = extract_embeddings(model_name, apply_normalization, output_dir, file_list)
    logging.info("embedding extracted")

    write_csv(output_csv, list_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings using a torchvision model.")

    parser.add_argument('--zip_file', required=True, help="Path to the ZIP file containing images.")
    parser.add_argument('--model_name', required=True, choices=AVAILABLE_MODELS.keys(), help="Model for embedding extraction.")
    parser.add_argument('--normalize', action="store_true", help="Whether to apply normalization.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.zip_file, args.output_csv, args.model_name, args.normalize)
