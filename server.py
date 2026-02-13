import io
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms

from train import build_model

app = Flask(__name__)

# Global model and device
model = None
device = None
transform = None


def load_model_once():
    """Load the model once at startup."""
    global model, device, transform
    
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(num_classes=3).to(device)
        
        try:
            model.load_state_dict(torch.load("model.pt", map_location=device))
            model.eval()
            print(f"Model loaded successfully on {device}")
        except FileNotFoundError:
            print("ERROR: model.pt not found. Train the model first!")
            raise
        
        transform = transforms.Compose([
            transforms.Resize((288, 384)),  # 4:3 aspect ratio
            transforms.ToTensor(),
        ])


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "message": "Image AI Classification API",
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the class of an uploaded image.
    
    Expected: multipart/form-data with 'image' field containing the image file
    OR JSON with 'image_url' field
    
    Returns: JSON with prediction, confidence, and class name
    """
    try:
        # Check if image file is in the request
        if "image" not in request.files:
            return jsonify({
                "error": "No image file provided",
                "message": "Send image as multipart/form-data with key 'image'"
            }), 400
        
        file = request.files["image"]
        
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Transform and predict
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        class_names = {0: "Roblox", 1: "Screenshot", 2: "Other"}
        class_name = class_names.get(predicted_class, "Unknown")
        
        return jsonify({
            "success": True,
            "prediction": {
                "class_id": predicted_class,
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),
                "all_probabilities": {
                    "Roblox": round(probabilities[0, 0].item() * 100, 2),
                    "Screenshot": round(probabilities[0, 1].item() * 100, 2),
                    "Other": round(probabilities[0, 2].item() * 100, 2)
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Predict classes for multiple images.
    
    Expected: multipart/form-data with multiple 'images' fields
    
    Returns: JSON array with predictions for each image
    """
    try:
        files = request.files.getlist("images")
        
        if not files:
            return jsonify({
                "error": "No image files provided",
                "message": "Send images as multipart/form-data with key 'images'"
            }), 400
        
        results = []
        
        for file in files:
            if file.filename == "":
                continue
            
            try:
                # Read and process image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Transform and predict
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                
                class_names = {0: "Roblox", 1: "Screenshot", 2: "Other"}
                class_name = class_names.get(predicted_class, "Unknown")
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "class_id": predicted_class,
                    "class_name": class_name,
                    "confidence": round(confidence * 100, 2)
                })
            
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "count": len(results),
            "predictions": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Loading model...")
    load_model_once()
    print("\nStarting Flask server on http://localhost:1000")
    print("Ready for Discord bot connections!\n")
    print("Endpoints:")
    print("  GET  /         - API info")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Single image classification")
    print("  POST /predict_batch - Batch image classification")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host="0.0.0.0", port=1000, debug=False)
