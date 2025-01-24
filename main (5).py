import os
import asyncio
import logging
import datetime
import threading
from openai import OpenAI
import base64
import random
import string
import discord
import aiohttp
import httpx
import openai
from dotenv import load_dotenv
import time
import subprocess
import google.generativeai as genai
from discord import File
import yt_dlp
import httpx
import io

# Import bot-specific utilities and prompts
from tommy_utils import get_response as tommy_get_response
from bob_utils import generate_meme, describe_image_with_openai, get_response as bob_get_response
from claude_utils import get_anthropic_response
from george_utils import get_response as george_get_response
from sybil_utils import get_response as sybil_get_response
from prompts import tommy_system_prompt, bob_system_prompt, claude_system_prompt, george_system_prompt, sybil_system_prompt
from allowed import allowed_channels, allowed_users


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.INFO)

# Load environment variables and configure APIs
load_dotenv()

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Required API keys and tokens
openai.api_key = os.environ['OPENAI_API_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
CLAUDE_BOT_TOKEN = os.environ['CLAUDE_BOT_TOKEN']
TOMMY_BOT_TOKEN = os.environ['TOMMY_BOT_TOKEN']
BOB_BOT_TOKEN = os.environ['BOB_BOT_TOKEN']
GEORGE_BOT_TOKEN = os.environ['GEORGE_BOT_TOKEN']
SYBIL_BOT_TOKEN = os.environ['SYBIL_BOT_TOKEN']
IDEAOGRAM_API_KEY = os.environ.get('IDEAOGRAM_API_KEY', '')
character_name = os.environ.get('BOT_NAME', '').lower()

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def start_server(character_name, port):
    """
    Start a server instance for a specific bot character.

    Args:
        character_name (str): Name of the bot character
        port (int): Port number for the server

    Returns:
        threading.Thread: The server thread instance
    """
    env = os.environ.copy()
    env['PORT'] = str(port)
    thread = threading.Thread(
        target=lambda: subprocess.run(["python", "server.py", character_name], env=env),
        daemon=True
    )
    thread.start()
    return thread

async def fetch_messages(channel_id, bot_token, total_limit=35):
    """
    Fetch recent messages from a given channel or thread.

    Args:
        channel_id (int): Discord channel ID
        bot_token (str): Bot authorization token
        total_limit (int): Maximum number of messages to fetch

    Returns:
        list: Formatted message history
    """
    url = f'https://discord.com/api/v9/channels/{channel_id}/messages'

    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://discord.com/api/v9/channels/{channel_id}', 
                             headers={'Authorization': f'Bot {bot_token}'}) as response:
            if response.status == 200:
                channel_data = await response.json()
                if channel_data.get('type') == 11:
                    total_limit = min(total_limit, 100)

    headers = {'Authorization': f'Bot {bot_token}'}
    messages = []
    last_message_id = None

    while len(messages) < total_limit:
        params = {'limit': 35}
        if last_message_id:
            params['before'] = last_message_id

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    batch = await response.json()
                    if not batch:
                        logger.info("No more messages to fetch.")
                        break
                    messages.extend(batch)
                    last_message_id = batch[-1]['id']
                    logger.info(f"Fetched {len(batch)} messages. Total: {len(messages)}")
                else:
                    logger.error(f"Failed to fetch messages: {response.status}")
                    break

    messages = list(reversed(messages))
    logger.info(f"Total messages fetched: {len(messages)}")

    formatted_messages = [
        {
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', ''),
            'username': msg['author']['username'],
            'user_id': msg['author']['id'],
            'attachments': [attachment['url'] for attachment in msg.get('attachments', [])]
        }
        for msg in messages
    ]

    return formatted_messages

def split_message(message, limit=2000):
    """
    Split a message into chunks that fit Discord's character limit.

    Args:
        message (str): The message to split
        limit (int): Maximum characters per chunk (default: 2000)

    Returns:
        list[str]: List of message chunks
    """
    return [message[i:i + limit] for i in range(0, len(message), limit)]



class MultiBot(discord.Client):
    """Base class for all bots, providing shared functionality."""

    def __init__(self, name, system_prompt, get_response_func, token, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.system_prompt = system_prompt
        self.get_response = get_response_func
        self.token = token
        self.memory = {}
        self.allowed_users = allowed_users
        self.allowed_channels = allowed_channels
        self.ai_users = {
            1170147847719616542: "Tommy",
            1128960694172258344: "Bob",
            1215201664974331904: "Claude",
            1317683969847988295: "George",
            1321179670649114714: "Sybil"
        }

    async def delete_previous_messages(self, message, num_messages):
        """Delete a specified number of previous messages in DMs."""
        if not isinstance(message.channel, discord.DMChannel):
            await message.channel.send("This command can only be used in DMs.")
            return

        try:
            num = int(num_messages)
            if num <= 0:
                await message.channel.send("Please provide a positive number of messages to delete.")
                return
        except ValueError:
            await message.channel.send("Please provide a valid number of messages to delete.")
            return

        messages_to_delete = []
        async for msg in message.channel.history(limit=num + 1):
            if msg.author == self.user:
                messages_to_delete.append(msg)

        deleted_count = 0
        for msg in messages_to_delete:
            try:
                await msg.delete()
                deleted_count += 1
                await asyncio.sleep(1.0)  # Rate limit protection
            except discord.errors.NotFound:
                pass
            except discord.errors.Forbidden:
                await message.channel.send("I don't have permission to delete some messages.")
                return
            except Exception as e:
                await message.channel.send(f"Error deleting messages: {str(e)}")
                return

        await message.channel.send(f"Deleted {deleted_count - 1} messages.")

    async def process_commands(self, message):
        """Process any known commands."""
        if isinstance(message.channel, discord.DMChannel):
            if message.content.startswith('!delete'):
                parts = message.content.split()
                if len(parts) == 2:
                    await self.delete_previous_messages(message, parts[1])
                else:
                    await message.channel.send("Usage: !delete [number_of_messages]")
                return True

        if message.content.startswith('!generate'):
            await self.handle_generate_command(message)
            return True

        if message.content.startswith('!meme'):
            await self.handle_meme_command(message)
            return True

        return False

    async def handle_generate_command(self, message):
        """Handle the !generate command for image generation."""
        prompt = message.content[9:].strip()
        if not prompt:
            await message.channel.send("Please provide a prompt after !generate.")
            return

        initial_response = await message.channel.send("Generating your image, please wait...")

        url = "https://api.ideogram.ai/generate"
        payload = {"image_request": {"model": "V_2", "magic_prompt_option": "ON", "prompt": prompt}}
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Api-Key": IDEAOGRAM_API_KEY
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data') and isinstance(data['data'], list) and len(data['data']) > 0:
                            image_data = data['data'][0]
                            image_url = image_data.get('url')
                            if image_url:
                                await initial_response.edit(content=f"Image generated successfully! {image_url}")
                            else:
                                await initial_response.edit(content="Failed to generate image. No image URL found.")
                        else:
                            await initial_response.edit(content="Failed to generate image. Unexpected response structure.")
                    else:
                        await initial_response.edit(content=f"Failed to generate image. Status code: {response.status}")
        except Exception as e:
            await initial_response.edit(content=f"An error occurred while generating the image: {str(e)}")

    async def handle_meme_command(self, message):
        """Handle the !meme command for generating memes."""
        parts = message.content.split('!meme')
        meme_prompt = parts[1].strip() if len(parts) > 1 else ''

        if meme_prompt:
            await message.channel.send("Generating your meme, please wait...")
            meme_image_url = await generate_meme(meme_prompt)
            if meme_image_url:
                await message.channel.send(meme_image_url)
            else:
                await message.channel.send("Sorry, I couldn't generate the meme.")
        else:
            await message.channel.send("Please provide some text for the meme.")

    async def send_chunked_message(self, channel, content):
        """Send a long response in multiple chunks if needed."""
        chunks = split_message(content)
        for chunk in chunks:
            await channel.send(chunk)

    def format_history(self, channel_history):
        """Format channel history into a text block."""
        history_text = '\n'.join(
            [f"{msg['timestamp']}: {msg['username']}: {msg['content']}" for msg in channel_history]
        )
        history_text += f"\nTotal messages fetched: {len(channel_history)}"
        return history_text

    def is_respond_condition_met(self, message):
        """Check if the bot should respond to the given message."""
        channel_id = message.channel.id
        is_mentioned = self.user.mentioned_in(message)
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_in_thread = isinstance(message.channel, discord.Thread)

        if is_in_thread and is_mentioned:
            return True

        is_allowed_location = channel_id in self.allowed_channels
        return (is_mentioned and is_allowed_location) or is_dm

    async def handle_standard_reactions(self, payload, message, user):
        """Handle standard emoji reactions shared across all bots."""
        if payload.emoji.name == '👍':
            await message.channel.send(f"Thanks for the like, {user.mention}!")
        elif payload.emoji.name == '❓':
            async with message.channel.typing():
                clarification_prompt = f"The user needs more clarification on this query: {message.content}\nThis was my original response: {message.content}\nPlease expand or provide a more detailed explanation."
                new_response = await self.get_response(clarification_prompt, self.system_prompt, "")
                await self.send_chunked_message(message.channel, new_response)
        elif payload.emoji.name == '❌':
            async with message.channel.typing():
                new_response = await self.get_response(message.content, self.system_prompt, "")
                await message.edit(content=f"{new_response} (edited)")

    async def on_message(self, message):
        """Handle incoming messages."""
        if message.author == self.user:
            return

        if await self.process_commands(message):
            return

        if not self.is_respond_condition_met(message):
            return

        await self.handle_message(message)

    async def handle_message(self, message):
        """Default message handling implementation."""
        async with message.channel.typing():
            user_id = message.author.id
            user_name = self.ai_users.get(user_id, self.allowed_users.get(user_id, str(user_id)))
            user_message = message.content.strip()

            channel_history_text = await fetch_messages(message.channel.id, self.token)
            history_text = self.format_history(channel_history_text)

            response_text = await self.get_response(user_message, self.system_prompt, history_text)

            if response_text:
                mention_prefix = f"{message.author.mention} "
                response_text = mention_prefix + response_text
                await self.send_chunked_message(message.channel, response_text)

class BobBot(MultiBot):
    """BobBot with user mention resolution and custom message handling."""

    def build_user_map(self, channel_history):
        """Build map of usernames to user IDs."""
        return {
            msg['username'].lower(): msg['user_id']
            for msg in channel_history
        }

    def replace_user_mentions(self, response_text, user_map, message):
        """Replace @mentions with proper Discord mentions."""
        words = response_text.split()
        final_words = []

        for w in words:
            if w.startswith('@'):
                raw_name = w[1:].lower()
                if raw_name not in user_map and message.guild is not None:
                    member = discord.utils.find(
                        lambda m: m.name.lower() == raw_name or (m.nick and m.nick.lower() == raw_name),
                        message.guild.members
                    )
                    if member:
                        user_map[raw_name] = member.id

                if raw_name in user_map:
                    final_words.append(f"<@{user_map[raw_name]}>")
                else:
                    final_words.append(w)
            else:
                final_words.append(w)
        return " ".join(final_words)

    async def handle_message(self, message):
        """Custom message handling for Bob."""
        async with message.channel.typing():
            user_id = message.author.id
            user_name = self.ai_users.get(user_id, self.allowed_users.get(user_id, str(user_id)))
            user_message = message.content.strip()

            channel_history_text = await fetch_messages(message.channel.id, self.token)
            history_text = self.format_history(channel_history_text)
            user_map = self.build_user_map(channel_history_text)

            response = await self.get_response(user_message, history_text, self.system_prompt, message.channel)

            if response:
                mention_prefix = f"{message.author.mention} "
                if response.get("content"):
                    content_str = self.replace_user_mentions(response["content"], user_map, message)
                    content_str = mention_prefix + content_str
                    await self.send_chunked_message(message.channel, content_str)

                if response.get("image_url"):
                    await message.channel.send(response["image_url"])

class ClaudeBot(MultiBot):
    """ClaudeBot with message queue and attachment processing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = asyncio.Queue()

    async def process_attachments(self, message, user_input, user_name):
        """Process message attachments."""
        if not message.attachments:
            return user_input

        attachment = message.attachments[0]
        if attachment.filename.lower().endswith('.txt'):
            text_content = await attachment.read()
            return f"This message is from {user_name}\n\n{user_input}\n\nAttached text:\n{text_content.decode('utf-8')}"

        if attachment.content_type and attachment.content_type.startswith('image/'):
            image_url = attachment.url
            image_media_type = attachment.content_type
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        return user_input

    async def handle_message(self, message):
        """Custom message handling with queue for Claude."""
        async with message.channel.typing():
            user_id = message.author.id
            user_name = self.ai_users.get(user_id, self.allowed_users.get(user_id, str(user_id)))
            user_message = message.content.strip()

            user_input = await self.process_attachments(message, user_message, user_name)
            channel_history_text = await fetch_messages(message.channel.id, self.token)
            history_text = self.format_history(channel_history_text)

            await self.message_queue.put((user_input, message.channel, history_text))

    async def process_message_queue(self):
        """Process messages in queue."""
        while True:
            user_input, channel, history_text = await self.message_queue.get()
            response_text = await get_anthropic_response(self.system_prompt, history_text, user_input)
            if response_text:
                await self.send_chunked_message(channel, response_text)
            self.message_queue.task_done()

    async def start_bot(self):
        """Start bot and message queue processing."""
        await asyncio.gather(
            super().start(self.token),
            self.process_message_queue()
        )

class ImageProcessingBot(MultiBot):
    """Base class for bots that process images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_image_types = ['.png', '.jpg', '.jpeg']
        self.client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def is_supported_image(self, filename):
        """Check if the file is a supported image type."""
        return any(filename.lower().endswith(ext) for ext in self.supported_image_types)

    def encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def process_image(self, image_path, prompt):
        """Process image using Gemini API."""
        try:
            base64_image = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model="models/gemini-2.0-flash-exp",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Sorry, I encountered an error processing the image: {str(e)}"

    async def handle_image_message(self, message, user_message):
        """Handle messages with image attachments."""
        for attachment in message.attachments:
            if self.is_supported_image(attachment.filename):
                try:
                    timestamp = int(time.time())
                    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                    file_ext = os.path.splitext(attachment.filename)[1].lower()
                    download_path = f"temp_image_{timestamp}_{random_str}{file_ext}"

                    await attachment.save(download_path)
                    try:
                        response_text = await self.process_image(download_path, user_message)
                        mention_prefix = f"{message.author.mention} "
                        response_text = mention_prefix + response_text
                        await self.send_chunked_message(message.channel, response_text)
                    finally:
                        if os.path.exists(download_path):
                            os.remove(download_path)
                except Exception as e:
                    await message.channel.send(f"Sorry, I encountered an error processing your image: {str(e)}")
                return True
        return False

class GeorgeBot(ImageProcessingBot):
    """GeorgeBot with image processing capabilities."""

    async def handle_message(self, message):
        """Custom message handling for George."""
        async with message.channel.typing():
            user_message = message.content.strip()

            # Handle image if present
            if message.attachments and await self.handle_image_message(message, user_message):
                return

            # Handle regular message
            channel_history_text = await fetch_messages(message.channel.id, self.token, total_limit=150)
            history_text = self.format_history(channel_history_text)

            response_text = await self.get_response(user_message, history_text, self.system_prompt)
            if response_text:
                mention_prefix = f"{message.author.mention} "
                response_text = mention_prefix + response_text
                await self.send_chunked_message(message.channel, response_text)

class SybilBot(ImageProcessingBot):
    """SybilBot with image processing and custom reaction handling."""

    async def on_raw_reaction_add(self, payload):
        """Custom reaction handling for Sybil."""
        channel = await self.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)

        # Special approval handling
        if (payload.channel_id == 1019641944181329941 and
            payload.emoji.name == '👍' and
            payload.user_id in [373957009617453057, 450847938156167168]):
            await message.channel.send("Approved!")

        # Standard reaction handling
        if message.author == self.user:
            user = await self.fetch_user(payload.user_id)
            await self.handle_standard_reactions(payload, message, user)

    async def handle_message(self, message):
        """Custom message handling for Sybil."""
        async with message.channel.typing():
            user_message = message.content.strip()

            # Handle image if present
            if message.attachments and await self.handle_image_message(message, user_message):
                return

            # Handle regular message
            channel_history_text = await fetch_messages(message.channel.id, self.token, total_limit=100)
            history_text = self.format_history(channel_history_text)

            response_text, image_url = await self.get_response(user_message, history_text, self.system_prompt)

            if response_text or image_url:
                try:
                    # Send text response if exists
                    if response_text:
                        mention_prefix = f"{message.author.mention} "
                        full_response = mention_prefix + response_text
                        await self.send_chunked_message(message.channel, full_response)

                    # Send image if exists
                    if image_url:
                        if image_url.startswith(('http://', 'https://')):
                            async with aiohttp.ClientSession() as session:
                                async with session.get(image_url) as img_response:
                                    if img_response.status == 200:
                                        img_data = await img_response.read()
                                        await message.channel.send(
                                            file=discord.File(
                                                io.BytesIO(img_data),
                                                filename='generated_image.png'
                                            )
                                        )
                except Exception as e:
                    logging.error(f"Error sending message: {str(e)}")
                    await message.channel.send("I encountered an error while trying to send the response.")

async def main():
    servers = []
    bots = []
    try:
        # Start servers for each character
        base_port = int(os.environ.get('PORT', 8080))
        server_configs = [
            ("tommy", base_port),
            ("bob", base_port + 1),
            ("claude", base_port + 2),
            ("george", base_port + 3),
            ("sybil", base_port + 4)
        ]

        for name, port in server_configs:
            server = start_server(name, port)
            servers.append(server)

        # Wait for servers to start
        await asyncio.sleep(2)

        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.reactions = True

        logger.info(f"Allowed channels: {allowed_channels}")
        logger.info(f"Allowed users: {allowed_users}")

        # Initialize bots with error handling
        bot_configs = [
            (MultiBot, 'Tommy', tommy_system_prompt, tommy_get_response, TOMMY_BOT_TOKEN),
            (BobBot, 'Bob', bob_system_prompt, bob_get_response, BOB_BOT_TOKEN),
            (ClaudeBot, 'Claude', claude_system_prompt, get_anthropic_response, CLAUDE_BOT_TOKEN),
            (GeorgeBot, 'George', george_system_prompt, george_get_response, GEORGE_BOT_TOKEN),
            (SybilBot, 'Sybil', sybil_system_prompt, sybil_get_response, SYBIL_BOT_TOKEN)
        ]

        for BotClass, name, prompt, response_func, token in bot_configs:
            try:
                bot = BotClass(name, prompt, response_func, token, intents=intents)
                bots.append(bot)
                logger.info(f"Successfully initialized {name} bot")
            except Exception as e:
                logger.error(f"Failed to initialize {name} bot: {e}", exc_info=True)
                raise

        # Start all bots
        await asyncio.gather(
            *(bot.start_bot() if isinstance(bot, ClaudeBot) else bot.start(bot.token) for bot in bots)
        )

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
        # Attempt graceful shutdown
        for bot in bots:
            try:
                await bot.close()
            except Exception as close_error:
                logger.error(f"Error closing bot {bot.name}: {close_error}")
        raise
    finally:
        # Cleanup any remaining resources
        for server in servers:
            if server.is_alive():
                server.join(timeout=1.0)

def upload_to_gemini(path, mime_type=None):
    """
    Upload a file to Gemini API.

    Args:
        path (str): Path to the file
        mime_type (str, optional): MIME type of the file

    Returns:
        genai.File: Uploaded file object
    """
    file = genai.upload_file(path, mime_type=mime_type)
    logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """
    Wait for Gemini files to become ACTIVE.

    Args:
        files (list): List of Gemini file objects to monitor

    Raises:
        Exception: If any file fails to process
    """
    logger.info("Waiting for file processing...")
    for f in files:
        file_name = f.name
        file_info = genai.get_file(file_name)
        while file_info.state.name == "PROCESSING":
            logger.debug(".", end="", flush=True)
            time.sleep(10)
            file_info = genai.get_file(file_name)
        if file_info.state.name != "ACTIVE":
            raise Exception(f"File {file_name} failed to process")
    logger.info("All files ready")

if __name__ == "__main__":
    try:
        logger.info("Starting bot application")
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        raise