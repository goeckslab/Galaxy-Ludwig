import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# Model Initialization
device = torch.device("cpu")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification head
model.eval()

# Dummy Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dummy Image Processing
try:
    dummy_image = Image.new("RGB", (224, 224))
    input_tensor = transform(dummy_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()
    print(f"Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"Error during embedding extraction: {e}")

