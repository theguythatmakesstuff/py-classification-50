# Flask API Server for Discord Bot Integration

Flask server running on **port 1000** for image classification.

## Quick Start

### 1. Start the server:
```powershell
python server.py
```

The server will load `model.pt` and listen on `http://localhost:1000`

### 2. Test the server:
```powershell
python test_server.py
```

This will verify the API is working correctly.

### 3. Connect Discord Bot (optional):

#### Setup:
1. Install discord.py:
   ```powershell
   pip install discord.py
   ```

2. Create a bot at https://discord.com/developers/applications

3. Copy your bot token

4. Edit `discord_bot_example.py` and replace `YOUR_BOT_TOKEN_HERE` with your token

5. Run the bot:
   ```powershell
   python discord_bot_example.py
   ```

#### Discord Commands:
- `!classify` - Upload an image to classify (attach image with command)
- `!ping` - Check if bot and API are online

---

## API Endpoints

### `GET /` 
API information and available endpoints

### `GET /health`
Health check - returns server status

### `POST /predict`
Classify a single image

**Request:** multipart/form-data with `image` field containing image file

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class_id": 1,
    "class_name": "Screenshot",
    "confidence": 99.88,
    "all_probabilities": {
      "Roblox": 0.12,
      "Screenshot": 99.88
    }
  }
}
```

### `POST /predict_batch`
Classify multiple images at once

**Request:** multipart/form-data with multiple `images` fields

---

## Testing with curl

```powershell
# Health check
curl http://localhost:1000/health

# Predict single image
curl -X POST -F "image=@input/test.png" http://localhost:1000/predict
```

---

## Files

- `server.py` - Flask API server
- `test_server.py` - Test script to verify API
- `discord_bot_example.py` - Example Discord bot integration
- `train.py` - Original training script
- `output.py` - CLI prediction tool
- `generate_images.py` - Generate visualization grids
