import os
import zipfile
import argparse
import shutil
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, DenseNet121_Weights

# Model configurations
MODEL_CONFIGS = {
    "resnet": {
        "model": lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        "img_size": (224, 224),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "efficientnet": {
        "model": lambda: models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
        "img_size": (224, 224),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "densenet": {
        "model": lambda: models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
        "img_size": (224, 224),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}

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
    """Loads a specified pretrained model with its appropriate settings."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    model = config["model"]().to(device)
    model.fc = torch.nn.Identity() if hasattr(model, 'fc') else model.classifier  # Adjust last layer if exists
    model.eval()
    
    return model, config["img_size"], config["normalization"]

def process_image(image_path, transform, device):
    """Loads and transforms an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

def extract_embeddings(image_dir, model_name, output_csv):
    """Extracts embeddings from images using the specified model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, normalization = load_model(model_name, device)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization["mean"], std=normalization["std"])
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

    # Define the CSV header
    if results:
        num_features = len(results[0]) - 1  # Subtract 1 for the filename
        header = ["sample_name"] + [f"vector{i+1}" for i in range(num_features)]
        
        # Save results to CSV with header
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
    parser = argparse.ArgumentParser(description="Extract image embeddings using a pretrained model.")
    parser.add_argument("--zip_file", required=True, help="Path to the ZIP file containing images.")
    parser.add_argument("--model_name", required=True, choices=MODEL_CONFIGS.keys(), help="Model for embedding extraction.")
    parser.add_argument("--output_csv", required=True, help="Path to save extracted embeddings.")
    args = parser.parse_args()

    temp_dir = "temp_images"
    extract_zip(args.zip_file, temp_dir)
    extract_embeddings(temp_dir, args.model_name, args.output_csv)
    cleanup_directory(temp_dir)

