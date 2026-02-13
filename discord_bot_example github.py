"""
Example Discord bot that sends images to the Flask classification server.

Install discord.py first:
    pip install discord.py requests

Setup:
1. Create a bot at https://discord.com/developers/applications
2. Get your bot token
3. Replace YOUR_BOT_TOKEN_HERE with your actual token
4. Run server.py first: python server.py
5. Run this bot: python discord_bot_example.py
"""

import asyncio
import io
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import discord
import requests
from discord.ext import commands

# Configuration
BOT_TOKEN = "MTQ3MTcxNjUzMjAzMDY2ODk0Mw.GsqxFd.lfV19h_zv4r_PRICrb4L4vY_hWG7jLUZ-kr2Ik"
API_URL = "http://localhost:1000/predict"
MODEL_NAME = "py-classification-50"

# Training state
training_in_progress = False

# Create bot with message content intent
intents = discord.Intents.default()
intents.message_content = True  # Required for reading message content
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# NOTE: You MUST enable "Message Content Intent" in Discord Developer Portal:
# 1. Go to https://discord.com/developers/applications
# 2. Select your bot
# 3. Go to "Bot" section
# 4. Scroll down to "Privileged Gateway Intents"
# 5. Enable "MESSAGE CONTENT INTENT"
# 6. Click "Save Changes"


@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")
    print(f"Connected to API at {API_URL}")
    print(f"\nBot is ready! Use these commands:")
    print(f"  !classify - Upload an image to classify")
    print(f"  !ping - Check bot status")
    print(f"\nOr use slash commands:")
    print(f"  /classify - Upload an image to classify")
    print(f"  /ping - Check bot status")
    print(f"  /retrain - Retrain the model (50 epochs)")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


@bot.command(name="classify")
async def classify_image(ctx):
    """
    Classify an attached image.
    Usage: !classify (attach an image)
    """
    global training_in_progress
    
    if training_in_progress:
        await ctx.send("⏳ Model is currently retraining. Please wait until training completes.")
        return
    
    if not ctx.message.attachments:
        await ctx.send("❌ Please attach an image to classify!")
        return
    
    attachment = ctx.message.attachments[0]
    
    # Check if it's an image
    if not any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
        await ctx.send("❌ Please attach a valid image file (png, jpg, jpeg, gif, webp)")
        return
    
    await ctx.send("🔄 Classifying image...")
    
    try:
        # Download the image
        image_bytes = await attachment.read()
        
        # Send to Flask API
        files = {"image": (attachment.filename, io.BytesIO(image_bytes), attachment.content_type)}
        response = requests.post(API_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                pred = data["prediction"]
                
                # Create embed for nice formatting
                embed = discord.Embed(
                    title="🤖 Image Classification Result",
                    color=discord.Color.green()
                )
                embed.add_field(
                    name="Model",
                    value=f"`{MODEL_NAME}`",
                    inline=False
                )
                embed.add_field(
                    name="Prediction",
                    value=f"**{pred['class_name']}**",
                    inline=True
                )
                embed.add_field(
                    name="Confidence",
                    value=f"{pred['confidence']}%",
                    inline=True
                )
                embed.add_field(
                    name="All Probabilities",
                    value=f"Roblox: {pred['all_probabilities']['Roblox']}%\nScreenshot: {pred['all_probabilities']['Screenshot']}%",
                    inline=False
                )
                embed.set_thumbnail(url=attachment.url)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ Classification failed: {data.get('error', 'Unknown error')}")
        else:
            await ctx.send(f"❌ Server error (status {response.status_code})")
    
    except requests.exceptions.ConnectionError:
        await ctx.send("❌ Cannot connect to classification server. Is server.py running?")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")


@bot.command(name="ping")
async def ping(ctx):
    """Check if the bot and API are responsive."""
    try:
        response = requests.get("http://localhost:1000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            await ctx.send(f"✅ Bot is online! API status: {data.get('status', 'unknown')}")
        else:
            await ctx.send(f"⚠️ Bot is online but API returned status {response.status_code}")
    except:
        await ctx.send("⚠️ Bot is online but cannot reach the API server")


# Slash commands (modern Discord commands)
@bot.tree.command(name="classify", description="Classify an image (attach with message)")
async def slash_classify(interaction: discord.Interaction, image: discord.Attachment):
    """Slash command version of classify."""
    global training_in_progress
    
    if training_in_progress:
        await interaction.response.send_message("⏳ Model is currently retraining. Please wait until training completes.")
        return
    
    await interaction.response.defer()
    
    try:
        # Download the image
        image_bytes = await image.read()
        
        # Send to Flask API
        files = {"image": (image.filename, io.BytesIO(image_bytes), image.content_type)}
        response = requests.post(API_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("success"):
                pred = data["prediction"]
                
                # Create embed for nice formatting
                embed = discord.Embed(
                    title="🤖 Image Classification Result",
                    color=discord.Color.green()
                )
                embed.add_field(
                    name="Model",
                    value=f"`{MODEL_NAME}`",
                    inline=False
                )
                embed.add_field(
                    name="Prediction",
                    value=f"**{pred['class_name']}**",
                    inline=True
                )
                embed.add_field(
                    name="Confidence",
                    value=f"{pred['confidence']}%",
                    inline=True
                )
                embed.add_field(
                    name="All Probabilities",
                    value=f"Roblox: {pred['all_probabilities']['Roblox']}%\nScreenshot: {pred['all_probabilities']['Screenshot']}%\nOther: {pred['all_probabilities']['Other']}%",
                    inline=False
                )
                embed.set_thumbnail(url=image.url)
                
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"❌ Classification failed: {data.get('error', 'Unknown error')}")
        else:
            await interaction.followup.send(f"❌ Server error (status {response.status_code})")
    
    except requests.exceptions.ConnectionError:
        await interaction.followup.send("❌ Cannot connect to classification server. Is server.py running?")
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@bot.tree.command(name="ping", description="Check if the bot and API are online")
async def slash_ping(interaction: discord.Interaction):
    """Slash command version of ping."""
    try:
        response = requests.get("http://localhost:1000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            await interaction.response.send_message(f"✅ Bot is online! API status: {data.get('status', 'unknown')}")
        else:
            await interaction.response.send_message(f"⚠️ Bot is online but API returned status {response.status_code}")
    except:
        await interaction.response.send_message("⚠️ Bot is online but cannot reach the API server")


@bot.tree.command(name="retrain", description="Retrain the model (runs train.py for 50 epochs)")
async def slash_retrain(interaction: discord.Interaction):
    """Retrain the model - runs train.py."""
    global training_in_progress
    
    if training_in_progress:
        await interaction.response.send_message("❌ Training is already in progress!")
        return
    
    # Backup existing model if it exists
    model_path = Path("model.pt")
    backup_path = None
    
    if model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"model_backup_{timestamp}.pt")
        try:
            shutil.copy2(model_path, backup_path)
            await interaction.response.send_message(
                f"🔄 Starting model retraining (50 epochs)...\n"
                f"📦 Backed up existing model to: `{backup_path.name}`\n"
                f"⏳ Classification commands are disabled until training completes."
            )
        except Exception as e:
            await interaction.response.send_message(f"⚠️ Failed to backup model: {e}\nStarting training anyway...")
            backup_path = None
    else:
        await interaction.response.send_message(
            "🔄 Starting model retraining (50 epochs)...\n"
            "⏳ Classification commands are disabled until training completes."
        )
    
    training_in_progress = True
    
    async def run_training():
        global training_in_progress
        try:
            # Get the Python executable path (same one running this script)
            python_exe = sys.executable
            
            # Run train.py as subprocess
            process = await asyncio.create_subprocess_exec(
                python_exe, "train.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                msg = "✅ Training completed successfully! Model saved to model.pt\n"
                msg += "✨ Classification commands are now re-enabled.\n\n"
                msg += "**Note:** Restart the Flask server (`python server.py`) to load the new model."
                if backup_path:
                    msg += f"\n📦 Previous model backed up as: `{backup_path.name}`"
                await interaction.followup.send(msg)
            else:
                error_msg = stderr.decode()[:1000]  # Limit error message length
                msg = f"❌ Training failed with error:\n```\n{error_msg}\n```"
                if backup_path:
                    msg += f"\n📦 Your previous model is safe at: `{backup_path.name}`"
                await interaction.followup.send(msg)
        
        except Exception as e:
            msg = f"❌ Training error: {str(e)}"
            if backup_path:
                msg += f"\n📦 Your previous model is safe at: `{backup_path.name}`"
            await interaction.followup.send(msg)
        
        finally:
            training_in_progress = False
    
    # Run training in background
    asyncio.create_task(run_training())


if __name__ == "__main__":
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("ERROR: Please set your bot token in the script!")
        print("Get your token from https://discord.com/developers/applications")
    else:
        print("Starting Discord bot...")
        bot.run(BOT_TOKEN)
