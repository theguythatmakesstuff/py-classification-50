"""
Test script to verify the Flask server is working.
Run this after starting server.py to test the API.
"""

import requests
from pathlib import Path


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:1000/health", timeout=5)
        print(f"✅ Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        return False


def test_predict(image_path):
    """Test prediction with an image file."""
    print(f"Testing /predict endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}\n")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/png")}
            response = requests.post("http://localhost:1000/predict", files=files, timeout=10)
        
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        
        if data.get("success"):
            pred = data["prediction"]
            print(f"   Class: {pred['class_name']}")
            print(f"   Confidence: {pred['confidence']}%")
            print(f"   All probabilities: {pred['all_probabilities']}\n")
            return True
        else:
            print(f"❌ Prediction failed: {data}\n")
            return False
    
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        return False


def main():
    print("=" * 60)
    print("Flask Server Test Script")
    print("=" * 60 + "\n")
    
    # Test health
    if not test_health():
        print("Server is not running or unreachable!")
        print("Start the server first: python server.py")
        return
    
    # Find a test image
    input_dir = Path("input")
    test_images = list(input_dir.glob("*.png"))[:2]
    
    if not test_images:
        print("No test images found in input/")
        return
    
    # Test predictions
    for img_path in test_images:
        test_predict(str(img_path))
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nYou can now connect your Discord bot to http://localhost:1000")


if __name__ == "__main__":
    main()
