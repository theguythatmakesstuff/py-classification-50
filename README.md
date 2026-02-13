# py-classification-50
Integration

A PyTorch-based image classification model trained to distinguish between Roblox screenshots, regular screenshots, and other images. Includes a Flask API server and Discord bot integration for easy deployment and use.

## Features

- 🤖 **PyTorch CNN Model** - Lightweight convolutional neural network trained for 50 epochs
- 🖼️ **3-Class Classification** - Distinguishes between Roblox, Screenshot, and Other images
- 📐 **4:3 Aspect Ratio** - Images processed at 288x384 resolution
- 🌐 **Flask REST API** - HTTP server on port 1000 for easy integration
- 💬 **Discord Bot** - Slash commands and text commands for classification
- 🔄 **In-Discord Retraining** - Use `/retrain` to retrain the model without stopping the bot
- 💾 **Automatic Backups** - Model automatically backed up before retraining
- 📊 **Visualization Tools** - Generate prediction grids and augmented samples

## Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone <your-repo-url>
cd "python image ai"

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install torch torchvision pillow flask discord.py requests
```

### 2. Train the Model

Place your training images in the `input/` folder. Images will be automatically labeled based on filename:
- `Roblox*.png` → Class 0 (Roblox)
- `Screenshot*.png` → Class 1 (Screenshot)
- Other filenames → Class 2 (Other)

```powershell
python train.py
```

Training runs for 50 epochs and saves the model to `model.pt`.

### 3. Start the API Server

```powershell
python server.py
```

Server runs on `http://localhost:1000`

### 4. Run the Discord Bot (Optional)

1. Create a bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Copy your bot token
3. Edit `discord_bot_example.py` and replace `BOT_TOKEN` with your token
4. Enable **Message Content Intent** in the bot settings
5. Run the bot:

```powershell
python discord_bot_example.py
```

## Usage

### Command Line Prediction

Classify a single image:

```powershell
python output.py input/test_image.png
```

### Flask API

**Health Check:**
```bash
curl http://localhost:1000/health
```

**Classify Image:**
```bash
curl -X POST -F "image=@path/to/image.png" http://localhost:1000/predict
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class_id": 1,
    "class_name": "Screenshot",
    "confidence": 99.88,
    "all_probabilities": {
      "Roblox": 0.05,
      "Screenshot": 99.88,
      "Other": 0.07
    }
  }
}
```

### Discord Bot Commands

**Slash Commands (Recommended):**
- `/classify` - Attach an image to classify
- `/ping` - Check bot and API status
- `/retrain` - Retrain the model (50 epochs, auto-backup)

**Text Commands:**
- `!classify` - Attach an image to classify
- `!ping` - Check bot and API status

## Project Structure

```
python image ai/
├── input/                      # Training images folder
├── model.pt                    # Trained model weights
├── model_backup_*.pt          # Automatic model backups
├── train.py                   # Training script (50 epochs)
├── server.py                  # Flask API server (port 1000)
├── output.py                  # CLI prediction tool
├── discord_bot_example.py     # Discord bot with slash commands
├── generate_images.py         # Visualization and augmentation tools
├── test_server.py             # API testing script
└── API_README.md              # Additional API documentation
```

## Model Architecture

```python
Sequential(
  Conv2d(3, 16, kernel_size=3, padding=1)
  ReLU()
  MaxPool2d(2)
  Conv2d(16, 32, kernel_size=3, padding=1)
  ReLU()
  MaxPool2d(2)
  Conv2d(32, 64, kernel_size=3, padding=1)
  ReLU()
  AdaptiveAvgPool2d((8, 8))
  Flatten()
  Linear(4096, 128)
  ReLU()
  Linear(128, 3)  # 3 classes
)
```

- **Model Name:** `py-classification-50`
- **Input Size:** 288×384 (4:3 aspect ratio)
- **Classes:** 3 (Roblox, Screenshot, Other)
- **Training:** 50 epochs with Adam optimizer
- **Loss:** CrossEntropyLoss

## Advanced Features

### Generate Visualization Grid

Create a grid showing predictions on random images:

```powershell
python generate_images.py
```

Outputs:
- `output_grid.png` - 4×4 grid with predictions (green=correct, red=wrong)
- `generated_samples/` - 20 augmented training samples

### Retrain from Discord

Use the `/retrain` command in Discord to:
1. Automatically backup current model with timestamp
2. Run full 50-epoch training
3. Disable classification during training
4. Notify when complete

### Multi-Image Classification

Batch API endpoint for multiple images:

```bash
curl -X POST -F "images=@image1.png" -F "images=@image2.png" \
  http://localhost:1000/predict_batch
```

## Configuration

### Discord Bot

Edit `discord_bot_example.py`:

```python
BOT_TOKEN = "your_bot_token_here"
API_URL = "http://localhost:1000/predict"
MODEL_NAME = "py-classification-50"
```

### Training Parameters

Edit `train.py`:

```python
def train(num_epochs: int = 50, batch_size: int = 16, lr: float = 1e-3):
```

### Flask Server

Default port: 1000. Change in `server.py`:

```python
app.run(host="0.0.0.0", port=1000, debug=False)
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
flask>=3.0.0
discord.py>=2.3.0
requests>=2.31.0
```

## Error Handling

- **Corrupted Images:** Automatically skipped during training
- **Missing Model:** Server refuses to start without `model.pt`
- **Training in Progress:** Classification commands disabled during `/retrain`
- **Connection Errors:** Bot notifies if Flask server is unreachable

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Pull requests welcome! For major changes, please open an issue first.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Flask REST API
- [discord.py](https://discordpy.readthedocs.io/) for Discord integration

---
