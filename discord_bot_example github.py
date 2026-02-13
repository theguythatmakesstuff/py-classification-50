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
BOT_TOKEN = ""
API_URL = "http://localhost:1000/predict"
MODEL_NAME = "py-classification-50"
SUPERUSER_ID = 0  # Replace with your Discord user ID (right-click your profile -> Copy User ID)

# Training state
training_in_progress = False
server_process = None  # Track Flask server process

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
    print(f"  !backup - Backup the current model [SUPERUSER]")
    print(f"  !listbackups - List all available backups")
    print(f"  !restore [backup_name] - Restore from backup [SUPERUSER]")
    print(f"  !retrain - Retrain the model (50 epochs) [SUPERUSER]")
    print(f"  !restartserver - Restart the Flask API server [SUPERUSER]")
    print(f"\nOr use slash commands:")
    print(f"  /classify - Upload an image to classify")
    print(f"  /ping - Check bot status")
    print(f"  /backup - Backup the current model [SUPERUSER]")
    print(f"  /listbackups - List all available backups")
    print(f"  /restore [backup_name] - Restore from backup [SUPERUSER]")
    print(f"  /retrain - Retrain the model (50 epochs) [SUPERUSER]")
    print(f"  /restartserver - Restart the Flask API server [SUPERUSER]")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    
    # Print superuser status
    if SUPERUSER_ID == 0:
        print(f"\n⚠️  WARNING: SUPERUSER_ID not set! Sensitive commands will be restricted.")
    else:
        print(f"\n✅ Superuser ID: {SUPERUSER_ID}")


def is_superuser(user_id: int) -> bool:
    """Check if user is a superuser."""
    return SUPERUSER_ID != 0 and user_id == SUPERUSER_ID


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


@bot.command(name="backup")
async def backup_model(ctx):
    """Backup the current model."""
    # Check superuser
    if not is_superuser(ctx.author.id):
        await ctx.send("❌ Only the bot owner can backup the model!")
        return
    
    model_path = Path("model.pt")
    
    if not model_path.exists():
        await ctx.send("❌ No model found to backup (model.pt does not exist)")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"model_backup_{timestamp}.pt")
        
        # Get file size for info
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        shutil.copy2(model_path, backup_path)
        
        await ctx.send(
            f"✅ Model backed up successfully!\n"
            f"📦 Backup saved as: `{backup_path.name}`\n"
            f"💾 Size: {size_mb:.2f} MB"
        )
    except Exception as e:
        await ctx.send(f"❌ Backup failed: {str(e)}")


# Slash commands (modern Discord commands)
@bot.tree.command(name="classify", description="Classify an image (attach with message)")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
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
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
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


@bot.tree.command(name="backup", description="Create a timestamped backup of the current model")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def slash_backup(interaction: discord.Interaction):
    """Backup the current model."""
    # Check superuser
    if not is_superuser(interaction.user.id):
        await interaction.response.send_message("❌ Only the bot owner can backup the model!")
        return
    
    model_path = Path("model.pt")
    
    if not model_path.exists():
        await interaction.response.send_message("❌ No model found to backup (model.pt does not exist)")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"model_backup_{timestamp}.pt")
        
        # Get file size for info
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        shutil.copy2(model_path, backup_path)
        
        await interaction.response.send_message(
            f"✅ Model backed up successfully!\n"
            f"📦 Backup saved as: `{backup_path.name}`\n"
            f"💾 Size: {size_mb:.2f} MB"
        )
    except Exception as e:
        await interaction.response.send_message(f"❌ Backup failed: {str(e)}")


@bot.tree.command(name="listbackups", description="List all available model backups")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def slash_listbackups(interaction: discord.Interaction):
    """List all available model backups."""
    try:
        # Find all backup files
        backups = sorted(Path(".").glob("model_backup_*.pt"), reverse=True)
        
        if not backups:
            await interaction.response.send_message("📦 No backups found.")
            return
        
        # Build message with backup info
        msg = f"📦 **Found {len(backups)} backup(s):**\n\n"
        
        for i, backup in enumerate(backups[:10], 1):  # Show max 10
            size_mb = backup.stat().st_size / (1024 * 1024)
            # Parse timestamp from filename
            try:
                timestamp_str = backup.stem.replace("model_backup_", "")
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                date_str = "Unknown date"
            
            msg += f"`{i}.` **{backup.name}**\n"
            msg += f"   📅 {date_str} | 💾 {size_mb:.2f} MB\n\n"
        
        if len(backups) > 10:
            msg += f"\n_...and {len(backups) - 10} more_"
        
        msg += f"\n\nUse `/restore <filename>` to restore a backup"
        
        await interaction.response.send_message(msg)
    except Exception as e:
        await interaction.response.send_message(f"❌ Error listing backups: {str(e)}")


@bot.tree.command(name="restore", description="Restore model from a backup")
@discord.app_commands.describe(backup_name="Name of the backup file (optional, defaults to most recent)")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def slash_restore(interaction: discord.Interaction, backup_name: str = None):
    """Restore model from a backup file."""
    # Check superuser
    if not is_superuser(interaction.user.id):
        await interaction.response.send_message("❌ Only the bot owner can restore models!")
        return
    
    global training_in_progress
    
    if training_in_progress:
        await interaction.response.send_message("❌ Cannot restore while training is in progress!")
        return
    
    try:
        # If no backup name provided, use most recent
        if backup_name is None:
            backups = sorted(Path(".").glob("model_backup_*.pt"), reverse=True)
            if not backups:
                await interaction.response.send_message("❌ No backups found. Create a backup first with `/backup`")
                return
            backup_path = backups[0]
        else:
            # Remove .pt if user included it
            if not backup_name.endswith(".pt"):
                backup_name += ".pt"
            backup_path = Path(backup_name)
            
            if not backup_path.exists():
                await interaction.response.send_message(
                    f"❌ Backup file not found: `{backup_name}`\n"
                    f"Use `/listbackups` to see available backups"
                )
                return
        
        model_path = Path("model.pt")
        
        # Create safety backup of current model before restoring
        if model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safety_backup = Path(f"model_backup_before_restore_{timestamp}.pt")
            shutil.copy2(model_path, safety_backup)
            safety_msg = f"\n🔒 Current model backed up to: `{safety_backup.name}`"
        else:
            safety_msg = ""
        
        # Restore the backup
        shutil.copy2(backup_path, model_path)
        
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        
        await interaction.response.send_message(
            f"✅ Model restored successfully!\n"
            f"📦 Restored from: `{backup_path.name}`\n"
            f"💾 Size: {size_mb:.2f} MB{safety_msg}\n\n"
            f"**⚠️ Important:** Restart the Flask server (`python server.py`) to load the restored model."
        )
    except Exception as e:
        await interaction.response.send_message(f"❌ Restore failed: {str(e)}")


@bot.tree.command(name="retrain", description="Retrain the model (runs train.py for 50 epochs)")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def slash_retrain(interaction: discord.Interaction):
    """Retrain the model - runs train.py."""
    # Check superuser
    if not is_superuser(interaction.user.id):
        await interaction.response.send_message("❌ Only the bot owner can retrain the model!")
        return
    
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


@bot.tree.command(name="restartserver", description="Restart the Flask API server")
@discord.app_commands.allowed_installs(guilds=True, users=True)
@discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def slash_restartserver(interaction: discord.Interaction):
    """Restart the Flask server."""
    # Check superuser
    if not is_superuser(interaction.user.id):
        await interaction.response.send_message("❌ Only the bot owner can restart the server!")
        return
    
    await interaction.response.defer()
    
    try:
        # Start server.py as subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Give it a moment to start
        await asyncio.sleep(2)
        
        # Check if server is responding
        try:
            response = requests.get("http://localhost:1000/health", timeout=5)
            if response.status_code == 200:
                await interaction.followup.send(
                    "✅ Flask server restarted successfully!\n"
                    "🌐 API is ready at http://localhost:1000"
                )
            else:
                await interaction.followup.send(
                    "⚠️ Server process started but API not responding properly.\n"
                    "Check terminal logs for errors."
                )
        except requests.exceptions.RequestException:
            await interaction.followup.send(
                "⚠️ Server process started but API not responding yet.\n"
                "It may still be loading. Check terminal logs."
            )
        
    except Exception as e:
        await interaction.followup.send(f"❌ Failed to restart server: {str(e)}")


# -------------------- TEXT COMMANDS (Legacy) --------------------

@bot.command(name="listbackups")
async def list_backups_command(ctx):
    """List all available model backups (text command)."""
    try:
        # Find all backup files
        backups = sorted(Path(".").glob("model_backup_*.pt"), reverse=True)
        
        if not backups:
            await ctx.send("📦 No backups found.")
            return
        
        # Build message with backup info
        msg = f"📦 **Found {len(backups)} backup(s):**\n\n"
        
        for i, backup in enumerate(backups[:10], 1):  # Show max 10
            size_mb = backup.stat().st_size / (1024 * 1024)
            # Parse timestamp from filename
            try:
                timestamp_str = backup.stem.replace("model_backup_", "").replace("before_restore_", "")
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                date_str = "Unknown date"
            
            msg += f"`{i}.` **{backup.name}**\n"
            msg += f"   📅 {date_str} | 💾 {size_mb:.2f} MB\n\n"
        
        if len(backups) > 10:
            msg += f"\n_...and {len(backups) - 10} more_"
        
        msg += f"\n\nUse `!restore <filename>` to restore a backup"
        
        await ctx.send(msg)
    except Exception as e:
        await ctx.send(f"❌ Error listing backups: {str(e)}")


@bot.command(name="restore")
async def restore_command(ctx, backup_name: str = None):
    """Restore model from backup (text command)."""
    # Check superuser
    if not is_superuser(ctx.author.id):
        await ctx.send("❌ Only the bot owner can restore models!")
        return
    
    global training_in_progress
    
    if training_in_progress:
        await ctx.send("❌ Cannot restore while training is in progress!")
        return
    
    try:
        # If no backup name provided, use most recent
        if backup_name is None:
            backups = sorted(Path(".").glob("model_backup_*.pt"), reverse=True)
            if not backups:
                await ctx.send("❌ No backups found. Create a backup first with `!backup`")
                return
            backup_path = backups[0]
        else:
            # Remove .pt if user included it
            if not backup_name.endswith(".pt"):
                backup_name += ".pt"
            backup_path = Path(backup_name)
            
            if not backup_path.exists():
                await ctx.send(
                    f"❌ Backup file not found: `{backup_name}`\n"
                    f"Use `!listbackups` to see available backups"
                )
                return
        
        model_path = Path("model.pt")
        
        # Create safety backup of current model before restoring
        if model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safety_backup = Path(f"model_backup_before_restore_{timestamp}.pt")
            shutil.copy2(model_path, safety_backup)
            safety_msg = f"\n🔒 Current model backed up to: `{safety_backup.name}`"
        else:
            safety_msg = ""
        
        # Restore the backup
        shutil.copy2(backup_path, model_path)
        
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        
        await ctx.send(
            f"✅ Model restored successfully!\n"
            f"📦 Restored from: `{backup_path.name}`\n"
            f"💾 Size: {size_mb:.2f} MB{safety_msg}\n\n"
            f"**⚠️ Important:** Restart the Flask server (`python server.py`) to load the restored model."
        )
    except Exception as e:
        await ctx.send(f"❌ Restore failed: {str(e)}")


@bot.command(name="retrain")
async def retrain_command(ctx):
    """Retrain the model (text command)."""
    # Check superuser
    if not is_superuser(ctx.author.id):
        await ctx.send("❌ Only the bot owner can retrain the model!")
        return
    
    global training_in_progress
    
    if training_in_progress:
        await ctx.send("❌ Training is already in progress!")
        return
    
    # Backup existing model if it exists
    model_path = Path("model.pt")
    backup_path = None
    
    if model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"model_backup_{timestamp}.pt")
        try:
            shutil.copy2(model_path, backup_path)
            await ctx.send(
                f"🔄 Starting model retraining (50 epochs)...\n"
                f"📦 Backed up existing model to: `{backup_path.name}`\n"
                f"⏳ Classification commands are disabled until training completes."
            )
        except Exception as e:
            await ctx.send(f"⚠️ Failed to backup model: {e}\nStarting training anyway...")
            backup_path = None
    else:
        await ctx.send(
            "🔄 Starting model retraining (50 epochs)...\n"
            "⏳ Classification commands are disabled until training completes."
        )
    
    training_in_progress = True
    
    async def run_training():
        try:
            # Run train.py as subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, "train.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                msg = "✅ Training completed successfully!\n🎉 New model is ready for use."
            else:
                msg = f"❌ Training failed with error:\n```\n{stderr.decode()[:500]}\n```"
                if backup_path:
                    msg += f"\n📦 Your previous model is safe at: `{backup_path.name}`"
            await ctx.send(msg)
        
        except Exception as e:
            msg = f"❌ Training error: {str(e)}"
            if backup_path:
                msg += f"\n📦 Your previous model is safe at: `{backup_path.name}`"
            await ctx.send(msg)
        
        finally:
            training_in_progress = False
    
    # Run training in background
    asyncio.create_task(run_training())


@bot.command(name="restartserver")
async def restartserver_command(ctx):
    """Restart the Flask server (text command)."""
    # Check superuser
    if not is_superuser(ctx.author.id):
        await ctx.send("❌ Only the bot owner can restart the server!")
        return
    
    msg = await ctx.send("🔄 Attempting to restart Flask server...")
    
    try:
        # Start server.py as subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Give it a moment to start
        await asyncio.sleep(2)
        
        # Check if server is responding
        try:
            response = requests.get("http://localhost:1000/health", timeout=5)
            if response.status_code == 200:
                await msg.edit(content=
                    "✅ Flask server restarted successfully!\n"
                    "🌐 API is ready at http://localhost:1000"
                )
            else:
                await msg.edit(content=
                    "⚠️ Server process started but API not responding properly.\n"
                    "Check terminal logs for errors."
                )
        except requests.exceptions.RequestException:
            await msg.edit(content=
                "⚠️ Server process started but API not responding yet.\n"
                "It may still be loading. Check terminal logs."
            )
        
    except Exception as e:
        await msg.edit(content=f"❌ Failed to restart server: {str(e)}")


if __name__ == "__main__":
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("ERROR: Please set your bot token in the script!")
        print("Get your token from https://discord.com/developers/applications")
    else:
        print("Starting Discord bot...")
        bot.run(BOT_TOKEN)
