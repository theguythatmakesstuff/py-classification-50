import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import build_model


def load_model(checkpoint_path: str = "model.pt", num_classes: int = 3):
    """Load the trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


def predict_image(model, device, image_path: str):
    """Predict the class of a single image."""
    transform = transforms.Compose(
        [
            transforms.Resize((288, 384)),  # 4:3 aspect ratio
            transforms.ToTensor(),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        class_names = {0: "Roblox", 1: "Screenshot", 2: "Other"}
        return predicted_class, class_names.get(predicted_class, "Unknown"), confidence

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python output.py <image_path>")
        print("Example: python output.py input/test_image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)

    print("Loading model...")
    model, device = load_model()
    
    print(f"Classifying: {image_path}")
    class_id, class_name, confidence = predict_image(model, device, image_path)
    
    if class_id is not None:
        print(f"\nPrediction: Class {class_id} ({class_name})")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("Failed to classify image")


if __name__ == "__main__":
    main()
