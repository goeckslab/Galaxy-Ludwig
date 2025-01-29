import os
import zipfile
import argparse
import shutil
import pandas as pd
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define the default resize and normalization settings for models

MODEL_DEFAULTS = {
    # Default normalization (ImageNet)
    "default": {"resize": (224, 224), "normalize": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])},

    # Models using (224, 224) resize and ImageNet normalization
    "alexnet": {"resize": (224, 224)},
    "vgg": {"resize": (224, 224)},
    "resnet": {"resize": (224, 224)},
    "densenet": {"resize": (224, 224)},
    "regnet": {"resize": (224, 224)},
    "mobilenet": {"resize": (224, 224)},
    "shufflenet": {"resize": (224, 224)},
    "convnext": {"resize": (224, 224)},
    "googlenet": {"resize": (224, 224)},

    # EfficientNet models with different resize values
    "efficientnet_b0": {"resize": (224, 224)},
    "efficientnet_b1": {"resize": (240, 240)},
    "efficientnet_b2": {"resize": (260, 260)},
    "efficientnet_b3": {"resize": (300, 300)},
    "efficientnet_b4": {"resize": (380, 380)},
    "efficientnet_b5": {"resize": (456, 456)},
    "efficientnet_b6": {"resize": (528, 528)},
    "efficientnet_b7": {"resize": (600, 600)},

    # Inception-V3 has a unique input size
    "inception_v3": {"resize": (299, 299)},

    # Vision Transformer models with different normalization
    "vit_b_16": {"resize": (384, 384), "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])},
    "vit_b_32": {"resize": (384, 384), "normalize": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])},

    # Swin Transformer models with different normalization
    "swin": {"resize": (224, 224), "normalize": ([0.5, 0.0, 0.5], [0.5, 0.5, 0.5])}
}

# Ensure all models have normalization applied (if not defined)
for model in MODEL_DEFAULTS:
    if "normalize" not in MODEL_DEFAULTS[model]:
        MODEL_DEFAULTS[model]["normalize"] = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Get all model names from torchvision.models
AVAILABLE_MODELS = {name: getattr(models, name) for name in dir(models) if callable(getattr(models, name))}

def extract_zip(zip_file, output_dir):
    """Extracts a ZIP file into a given directory."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted ZIP to {output_dir}")
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid ZIP file.")
    except Exception as e:
        raise RuntimeError(f"Error extracting ZIP file: {e}")

def load_model(model_name, device):
    """Loads a specified torchvision model and modifies it for feature extraction."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")

    model = AVAILABLE_MODELS[model_name](weights="DEFAULT").to(device)

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
        print(f"Skipping {image_path}: {e}")
        return None

def extract_embeddings(image_dir, model_name, output_csv, apply_normalization):
    """Extracts embeddings from images using a specified model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)

    # Retrieve the resize and normalize values for the selected model
    model_settings = MODEL_DEFAULTS.get(model_name, MODEL_DEFAULTS["default"])
    resize = model_settings["resize"]

    # Apply normalization if required by the user
    if apply_normalization:
        normalize = model_settings.get("normalize")
        transform = transforms.Compose([
            transforms.Resize(resize),  # Dynamic size based on model
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),  # Dynamic size based on model
            transforms.ToTensor(),
        ])

    results = []
    image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(image_dir)
        for file in files if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    print(f"Processing {len(image_files)} images...")

    with torch.no_grad():
        for image_path in image_files:
            input_tensor = process_image(image_path, transform, device)
            if input_tensor is None:
                continue  # Skip failed images
            embedding = model(input_tensor).squeeze().cpu().numpy()
            results.append([os.path.basename(image_path)] + embedding.tolist())

    # Save results to CSV
    if results:
        num_features = len(results[0]) - 1  # Subtract 1 for the filename
        header = ["sample_name"] + [f"vector{i+1}" for i in range(num_features)]
        df = pd.DataFrame(results, columns=header)
        df.to_csv(output_csv, index=False)
        print(f"Saved embeddings to {output_csv}")
    else:
        print("No valid images found. CSV not saved.")

def cleanup_directory(directory):
    """Removes a directory and its contents."""
    try:
        shutil.rmtree(directory)
        print(f"Cleaned up temporary directory: {directory}")
    except Exception as e:
        print(f"Error cleaning up directory {directory}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings using a torchvision model.")
    parser.add_argument("--zip_file", required=True, help="Path to the ZIP file containing images.")
    parser.add_argument("--model_name", required=True, choices=AVAILABLE_MODELS.keys(), help="Model for embedding extraction.")
    parser.add_argument("--output_csv", required=True, help="Path to save extracted embeddings.")
    parser.add_argument("--normalize", action="store_true", help="Whether to apply normalization.")
    args = parser.parse_args()

    temp_dir = "temp_images"
    extract_zip(args.zip_file, temp_dir)
    extract_embeddings(temp_dir, args.model_name, args.output_csv, args.normalize)
    cleanup_directory(temp_dir)

