
import os
import zipfile
import argparse
import pandas as pd
import numpy as np
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
    """
    Extracts the contents of a ZIP file to a specified output directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"ZIP file extracted to {output_dir}")
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        raise

def extract_embeddings(image_dir, model_name, output_csv):
    """
    Extracts embeddings from images in a directory using a specified model.
    """
    # Validate model name
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are: {list(MODEL_CONFIGS.keys())}")

    # Get model-specific settings
    config = MODEL_CONFIGS[model_name]
    img_size = config["img_size"]
    normalization = config["normalization"]
    model_class = config["model"]
    # Initialize the model

    device = torch.device("cpu")
    print("device")
    model = model_class().to(device)
    print("model_class")
    model.fc = torch.nn.Identity()  # Remove classification head
    print("model.fc")
    model.eval()
    print("model.eval")

    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization["mean"], std=normalization["std"])
    ])

    print("transform")
    # Process each image and extract embeddings
    results = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue  # Skip non-image files
            try:
                # Load and transform image
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                # Extract embedding
                with torch.no_grad():
                    embedding = model(input_tensor).squeeze().cpu().numpy()
                # Store result
                results.append([file] + embedding.tolist())
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save results to CSV
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, header=False)
        print(f"Embeddings saved to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image embeddings using a pretrained model.")
    parser.add_argument("--zip_file", required=True, help="Path to the ZIP file containing images.")
    parser.add_argument("--model_name", required=True, choices=MODEL_CONFIGS.keys(), help="Model to use for embedding extraction (resnet, efficientnet, densenet).")
    parser.add_argument("--output_csv", required=True, help="Path to save the extracted embeddings CSV.")
    args = parser.parse_args()

    # Extract ZIP contents
    temp_dir = "temp_images"
    extract_zip(args.zip_file, temp_dir)

    # Extract embeddings from images
    extract_embeddings(temp_dir, args.model_name, args.output_csv)

    # Clean up temporary directory
    for root, _, files in os.walk(temp_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    os.rmdir(temp_dir)

