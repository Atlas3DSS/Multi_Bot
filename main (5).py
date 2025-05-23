import os
import asyncio
import logging
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
import discord
import aiofiles
import os
from datetime import datetime
import nacl
import multiprocessing 
from tommy_utils import get_response as tommy_get_response
from bob_utils import get_response as bob_get_response
from claude_utils import get_anthropic_response, analyze_codebase, create_summary_report
from prompts import tommy_system_prompt, bob_system_prompt, claude_system_prompt
from allowed import allowed_channels, allowed_users
import anthropic

# Configure logging
todays_date = datetime.now().strftime("%Y-%m-%d")
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

def start_server():
    """
    Start a unified server instance for all bots.

    Returns:
        multiprocessing.Process: The server process instance
    """
    # Use environment port or default to 80
    port = int(os.environ.get('PORT', 80))
    os.environ['PORT'] = str(port)  # Ensure port is set in environment

    # Import and use the run_server function from server.py
    from server import run_server

    # Start server in a separate process
    def run_flask_server():
        logger.info(f"Starting web server on port 80")
        run_server()

    server = multiprocessing.Process(
        target=run_flask_server,
        daemon=True
    )
    server.start()
    logger.info(f"Web server process started with PID: {server.pid}")
    return server

async def fetch_messages(channel_id, bot_token, total_limit=50, exclude_last=False):
    """
    Fetch recent messages from a given channel or thread.

    Args:
        channel_id (int): Discord channel ID
        bot_token (str): Bot authorization token
        total_limit (int): Maximum number of messages to fetch
        exclude_last (bool): Whether to exclude the most recent message

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
                    total_limit = min(total_limit, 50)

    headers = {'Authorization': f'Bot {bot_token}'}
    messages = []
    last_message_id = None

    while len(messages) < total_limit:
        params = {'limit': 50}
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

    messages = list(reversed(messages[:-1]))

    # If exclude_last is True and we have messages, remove the last message
    if exclude_last and messages:
        messages = messages[:-1]

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

class MultiBot(discord.Client):
    """Enhanced base class for all bots, providing comprehensive functionality."""

    def __init__(self, name, system_prompt, get_response_func, token, *args, **kwargs):
        intents = kwargs.get('intents', discord.Intents.default())
        intents.message_content = True
        intents.dm_messages = True
        intents.reactions = True
        super().__init__(intents=intents)

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
        # Image processing attributes
        self.supported_image_types = ['.png', '.jpg', '.jpeg']
        self.client = kwargs.get('openai_client')  # Optional OpenAI client for image processing
        # Message queue for async processing
        self.message_queue = asyncio.Queue() if kwargs.get('use_queue') else None
        # User mention handling
        self.user_map = {}
        # New attributes for MMF username collection
        self.announcement_messages = {}  # Store messages Bob was tagged in
        self.pending_mmf_usernames = {}  # Track users who need to provide MMF username

        if not os.path.exists('mmf_usernames.csv'):
            with open('mmf_usernames.csv', 'w') as file:
                file.write('userid,username,time,uploaded_to_mmf\n')
        self.mmf_csv_path = 'mmf_usernames.csv'


    async def setup_hook(self):
        """Setup hook for Discord.py."""
        logger.info(f"Setting up {self.name} bot")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"{self.name} bot is ready and connected to Discord!")

        #if self.name == "Bob":
            #await self.check_missing_reactions()

            # Console log everyone it is sending a message to
            #for user_id in self.pending_mmf_usernames:
                #try:
                    #user = await self.fetch_user(user_id)
                    #if user:
                        #logger.info(f"Sending message to {user.name} (ID: {user_id})")
                #except Exception as e:
                    #logger.error(f"Error fetching user {user_id}: {e}")

    async def on_message(self, message):
        """Handle incoming messages."""
        # Ignore messages from the bot itself.
        if message.author == self.user:
            return 
        # Handle MMF username collection in DMs specific to Bob.
        if self.name == "Bob":
            if isinstance(message.channel, discord.DMChannel):
                if message.author.id in self.pending_mmf_usernames:
                    await self.handle_mmf_username_response(message)
                    return
            # Store announcement where Bob is tagged, in specific channels.
            if message.channel.id in {772954142075387914, 775589136752443403} and self.user.mentioned_in(message):
                self.announcement_messages[message.id] = message.content 
                return
        # General DM command handling.
        if isinstance(message.channel, discord.DMChannel):
            if message.content.startswith('!delete'):
                try:
                    _, count = message.content.split()
                    count = int(count)
                    if count > 0:
                        deleted = 0
                        async for msg in message.channel.history(limit=100):
                            if deleted >= count:
                                break
                            if msg.author == self.user:
                                await msg.delete()
                                deleted += 1
                                await asyncio.sleep(0.5)  # Delay to avoid rate limit
                        await message.add_reaction('✅')
                        return
                except (ValueError, IndexError):
                    await message.channel.send("Please use the format: !delete [number]")
                    return      

        # Check if the bot is mentioned in a public or allowed context.
        is_mentioned = self.user.mentioned_in(message)
        is_in_thread = isinstance(message.channel, discord.Thread)

        # Proceed if mentioned where applicable.
        if ((is_in_thread and is_mentioned) or 
            (is_mentioned and message.channel.id in self.allowed_channels) or 
            isinstance(message.channel, discord.DMChannel)):
            try:
                await self.handle_message(message)
            except Exception as e:
                logger.error(f"Error handling message in {self.name} bot: {e}", exc_info=True)
                await message.channel.send(f"Sorry, I encountered an error: {str(e)}")

    async def on_raw_reaction_add(self, payload):
        """Handle reaction events for MMF username collection."""
        # Only process reactions for Bob 
        if self.name != "Bob":
            return

        # Always process specific message ID
        if payload.message_id == 1353993100619550730:
            try:
                # Check if user already provided MMF username
                async with aiofiles.open('mmf_usernames.csv', 'r') as file:
                    async for line in file:
                        if line.startswith(str(payload.user_id)):
                            logger.info(f"User {payload.user_id} already provided MMF username.")
                            return

                # Track this user as needing to provide MMF username
                self.pending_mmf_usernames[payload.user_id] = payload.message_id

                # Get the user object and send DM
                user = await self.fetch_user(payload.user_id)
                if user:
                    try:
                        await user.send("""Hello! I'm messaging you because you reacted to our D'Jenacy announcement; 
if you want our new Devil girl model added to your MyMiniFactory library, please type your MyMiniFactory USERNAME without any additional words here:""")
                    except discord.Forbidden:
                        logger.error(f"Cannot send DM to user {payload.user_id}")
                return
            except Exception as e:
                logger.error(f"Error handling reaction for specific message: {e}", exc_info=True)

        # Regular announcement message handling
        if payload.message_id not in self.announcement_messages:
            return

        # Check if announcement message contains bot mention
        announcement_content = self.announcement_messages[payload.message_id]
        if not f'<@{self.user.id}>' in announcement_content:
            return

        # Check if the reaction is on a stored announcement message
        if payload.message_id in self.announcement_messages:
            try:
                # Don't process bot reactions
                if payload.user_id == self.user.id:
                    return

                # Check if the user ID is already present in mmf_usernames.csv
                async with aiofiles.open('mmf_usernames.csv', 'r') as file:
                    async for line in file:
                        if line.startswith(str(payload.user_id)):
                            logger.info(f"User {payload.user_id} already provided MMF username.")
                            return

                # Track this user as needing to provide MMF username
                self.pending_mmf_usernames[payload.user_id] = payload.message_id

                # Get the user object and send DM
                user = await self.fetch_user(payload.user_id)
                if user:
                    try:
                        await user.send("""Hello! I'm messaging you because you reacted to our D'Jenacy announcement; 
if you want our new Devil girl model added to your MyMiniFactory library, please type your MyMiniFactory USERNAME without any additional words here:""")
                    except discord.Forbidden:
                        logger.error(f"Cannot send DM to user {payload.user_id}")
            except Exception as e:
                logger.error(f"Error handling reaction: {e}", exc_info=True)

    async def check_missing_reactions(self):
        """Check for users who reacted but haven't provided MMF usernames and send them DMs."""
        if self.name != "Bob":
            return

        target_message_id = 1353993100619550730
        channel_id = 772954142075387914

        try:
            # Create missing_ids.csv if it doesn't exist
            if not os.path.exists('missing_ids.csv'):
                async with aiofiles.open('missing_ids.csv', 'w') as f:
                    await f.write('userid,username,time,uploaded_to_mmf\n')

            # Get the message
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error("Couldn't find channel")
                return

            message = await channel.fetch_message(target_message_id)
            if not message:
                logger.error("Couldn't find target message")
                return

            # Get all reaction user IDs
            reacted_users = set()
            for reaction in message.reactions:
                async for user in reaction.users():
                    if not user.bot:
                        reacted_users.add(str(user.id))

            # Get existing MMF usernames
            existing_users = set()
            async with aiofiles.open('mmf_usernames.csv', 'r') as f:
                async for line in f:
                    if line.strip() and not line.startswith('userid'):
                        user_id = line.split(',')[0]
                        existing_users.add(user_id)

            # Find missing users
            missing_users = reacted_users - existing_users

            if missing_users:
                # Log missing users
                async with aiofiles.open('missing_ids.csv', 'a') as f:
                    for user_id in missing_users:
                        await f.write(f"{user_id},,{datetime.now().isoformat()},false\n")

                logger.info(f"Found {len(missing_users)} users who need to provide MMF usernames")

                # Send DMs to missing users
                for user_id in missing_users:
                    try:
                        # Track this user as needing to provide MMF username
                        self.pending_mmf_usernames[int(user_id)] = target_message_id

                        # Get the user object and send DM
                        user = await self.fetch_user(int(user_id))
                        if user:
                            try:
                                await user.send("""Hello! I'm messaging you because you reacted to our D'Jenacy announcement; 
if you want our new Devil girl model added to your MyMiniFactory library, please type your MyMiniFactory USERNAME without any additional words here:""")
                                logger.info(f"Sent DM to user {user_id}")
                            except discord.Forbidden:
                                logger.error(f"Cannot send DM to user {user_id}")
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error checking missing reactions: {e}", exc_info=True)

    async def handle_mmf_username_response(self, message):
        """Process MMF username responses in DMs."""
        try:
            username = message.content.strip()
            user_id = message.author.id
            message_id = self.pending_mmf_usernames[user_id]

            # Append to MMF usernames CSV file
            async with aiofiles.open(self.mmf_csv_path, 'a', newline='') as file:
                await file.write(f"{user_id},{username},{datetime.now().isoformat()},false\n")

            # Remove from pending list
            del self.pending_mmf_usernames[user_id]

            # Remove from missing_ids.csv if present
            if os.path.exists('missing_ids.csv'):
                # Read current missing ids
                async with aiofiles.open('missing_ids.csv', 'r') as file:
                    lines = await file.readlines()

                # Write back without this user
                async with aiofiles.open('missing_ids.csv', 'w') as file:
                    for line in lines:
                        if not line.startswith(str(user_id)):
                            await file.write(line)

            # Send confirmation
            await message.channel.send("""Thank you - your Username has been recorded and D'Jenacy will show up in your library within 24 hours.
    Have a nice day!""")

        except Exception as e:
            logger.error(f"Error processing MMF username: {e}", exc_info=True)
            await message.channel.send("Sorry, there was an error processing your username. Please try again later.")

    async def _log_process_output(self, process, user_id, username):
        """Helper method to log the output from the puppeteer script."""
        stdout, stderr = await process.communicate()
        if stdout:
            logger.info(f"puppeteer_MMF.js output for {username} (ID: {user_id}):\n{stdout.decode()}")
        if stderr:
            logger.error(f"puppeteer_MMF.js error for {username} (ID: {user_id}):\n{stderr.decode()}")

        exit_code = process.returncode
        if exit_code == 0:
            logger.info(f"puppeteer_MMF.js completed successfully for {username}")
        else:
            logger.error(f"puppeteer_MMF.js failed with exit code {exit_code} for {username}")

    async def handle_message(self, message):
        """Enhanced message handling with support for all bot features."""
        async with message.channel.typing():
            user_id = message.author.id
            user_name = self.ai_users.get(user_id, self.allowed_users.get(user_id, str(user_id)))
            user_message = message.content.strip()
            #if channel is 775589136752443403 return
            if message.channel.id == 775589136752443403:
                return

            # Prepare variables for image handling
            image_urls = []
            user_input_for_llm = user_message # Default to user message text

            # Specific logic for Bob to handle images
            if self.name == "Bob" and message.attachments:
                logger.info("Bob is handling a message with attachments.")
                for attachment in message.attachments:
                    if self.is_supported_image(attachment.filename):
                        image_urls.append(attachment.url)
                        logger.info(f"Added image URL for Bob: {attachment.url}")
                # User input text remains the message content
                user_input_for_llm = user_message
                logger.info(f"Bob will process {len(image_urls)} images with text: '{user_input_for_llm}'")
            else:
                # Fallback or other bots: use existing attachment processing
                # This might create a list payload if an image is attached for other bots
                user_input_for_llm = await self.process_attachments(message, user_message, user_name)

            # Handle image if present (This seems redundant now, consider removing or integrating)
            # if message.attachments and self.client and self.is_supported_image(message.attachments[0].filename):
            #     if await self.handle_image_message(message, user_message):
            #         return

            # Get channel history
            channel_history_text = await fetch_messages(message.channel.id, self.token)
            history_text = self.format_history(channel_history_text)

            # Build user map for mention replacement
            self.build_user_map(channel_history_text)

            # Handle message queue if enabled
            if self.message_queue:
                await self.message_queue.put((user_message, message.channel, history_text))
                return

            # Prepare arguments for get_response
            response_args = {
                'user_input': user_input_for_llm,
                'history_text': history_text,
                ('bob_system_prompt' if self.name == "Bob" else 'system_prompt'): self.system_prompt, 
                'discord_channel': message.channel
            }

            # Add image_urls specifically for Bob
            if self.name == "Bob" and image_urls:
                response_args['image_urls'] = image_urls

            # Clean up args: remove system_prompt if bob_system_prompt exists, and vice versa
            if self.name == "Bob":
                response_args.pop('system_prompt', None)
            else:
                response_args.pop('bob_system_prompt', None)

            # Get response
            response = await self.get_response(**response_args)

            # Process response based on type
            if isinstance(response, tuple):  # Thinking bot response
                thinking_text, response_text = response
                thinking_file = None
                if thinking_text:
                    thinking_file = await self.save_thinking_process(thinking_text, message.id)
                if response_text:
                    mention_prefix = f"{message.author.mention} "
                    response_text = mention_prefix + self.replace_user_mentions(response_text, message)
                    await self.send_chunked_response(message.channel, response_text, thinking_file)
            elif isinstance(response, dict):  # Bot response with potential image
                if response.get("content"):
                    mention_prefix = f"{message.author.mention} "
                    content_str = self.replace_user_mentions(response["content"], message)
                    content_str = mention_prefix + content_str
                    await self.send_chunked_message(message.channel, content_str)
                if response.get("image_url"):
                    await message.channel.send(response["image_url"])
            elif response:  # Standard text response
                # Convert string literal newlines to actual newlines
                if isinstance(response, list):
                    response_text = response[0]
                else:
                    response_text = str(response)

                # explicitly convert string literal newlines to actual newlines
                response_text = response_text.replace('\\n', '\n')

                mention_prefix = f"{message.author.mention} "
                response_text = mention_prefix + self.replace_user_mentions(response_text, message)

                # split into chunks on actual newlines
                await self.send_chunked_message(message.channel, response_text)

    async def send_chunked_message(self, channel, content):
        """Send a long response in multiple chunks with improved async handling."""
        if not content:
            return

        max_length = 1900  # Leave room for mention prefix and formatting
        chunks = []
        current_pos = 0

        while current_pos < len(content):
            if len(content) - current_pos <= max_length:
                chunks.append(content[current_pos:])
                break

            split_candidates = [
                content.rfind('. ', current_pos, current_pos + max_length),
                content.rfind('? ', current_pos, current_pos + max_length),
                content.rfind('! ', current_pos, current_pos + max_length),
                content.rfind('\n', current_pos, current_pos + max_length),
                content.rfind(' ', current_pos, current_pos + max_length)
            ]

            split_point = max(x for x in split_candidates if x != -1)

            if split_point <= current_pos:
                split_point = current_pos + max_length

            chunks.append(content[current_pos:split_point].strip(' \t'))
            current_pos = split_point + 1

        for i, chunk in enumerate(chunks):
            try:
                if len(chunks) > 1:
                    chunk = f"[{i+1}/{len(chunks)}] {chunk}"
                await channel.send(chunk)
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
            except discord.errors.HTTPException as e:
                logger.error(f"Error sending message chunk {i+1}/{len(chunks)}: {str(e)}")
                try:
                    await channel.send("Error: Message chunk failed to send. Please try again.")
                except:
                    pass
                break
            await asyncio.sleep(0)

    def replace_user_mentions(self, response_text, message):
        """Replace @mentions with proper Discord mentions while preserving formatting."""
        if not hasattr(self, 'user_map'):
            return response_text

        lines = response_text.split('\n')
        final_lines = []

        for line in lines:
            words = line.split()
            final_words = []

            for w in words:
                if w.startswith('@'):
                    raw_name = w[1:].lower()
                    if raw_name not in self.user_map and message.guild is not None:
                        member = discord.utils.find(
                            lambda m: m.name.lower() == raw_name or (m.nick and m.nick.lower() == raw_name),
                            message.guild.members
                        )
                        if member:
                            self.user_map[raw_name] = member.id

                    if raw_name in self.user_map:
                        final_words.append(f"<@{self.user_map[raw_name]}>")
                    else:
                        final_words.append(w)
                else:
                    final_words.append(w)

            final_lines.append(" ".join(final_words).strip())

        return "\n".join(line for line in final_lines if line)

    async def process_message_queue(self):
        """Process messages in queue for bots that use message queues."""
        if not self.message_queue:
            return

        while True:
            try:
                user_input, channel, history_text = await self.message_queue.get()

                original_message = None
                if hasattr(channel, 'last_message') and channel.last_message:
                    original_message = channel.last_message
                elif hasattr(channel, 'history'):
                    async for message in channel.history(limit=1):
                        original_message = message
                        break

                response = await self.get_response(user_input, self.system_prompt, history_text)

                def get_mention_prefix():
                    if original_message and hasattr(original_message.author, 'mention'):
                        return f"{original_message.author.mention} "
                    return ""

                if hasattr(response, 'content') and isinstance(response.content, list):
                    thinking_text = ""
                    response_content = []

                    for block in response.content:
                        if hasattr(block, 'type'):
                            if block.type == 'thinking':
                                thinking_text = block.thinking
                            elif block.type == 'text':
                                text = block.text.strip()
                                if text:
                                    lines = text.split('\n')

                                    for line in lines:
                                        line = line.strip()
                                        if line:
                                            if line.startswith(('- ', '• ', '* ')):
                                                response_content.append(f"{line}")
                                            elif line.startswith('#'):
                                                response_content.append(f"\n{line}")
                                            elif line.startswith('```'):
                                                response_content.append(line)
                                            else:
                                                response_content.append(f"{line}")
                                        else:
                                            response_content.append('')

                    # Send thinking as a separate markdown block if available
                    if thinking_text:
                        thinking_message = f"**Thinking Process:**\n```\n{thinking_text}\n```"
                        await self.send_chunked_message(channel, thinking_message)

                    if response_content:
                        mention_prefix = get_mention_prefix()
                        response_text = '\n'.join(response_content)
                        response_text = mention_prefix + self.replace_user_mentions(response_text, original_message) if original_message else response_text
                        await self.send_chunked_message(channel, response_text)

                elif isinstance(response, tuple):
                    thinking_text, response_text = response
                    thinking_file = None
                    if thinking_text:
                        thinking_file = await self.save_thinking_process(thinking_text, original_message.id if original_message else None)
                    if response_text:
                        mention_prefix = get_mention_prefix()
                        response_text = mention_prefix + self.replace_user_mentions(response_text, original_message) if original_message else response_text
                        await self.send_chunked_response(channel, response_text, thinking_file)

                elif isinstance(response, dict):
                    if response.get("content"):
                        mention_prefix = get_mention_prefix()
                        content_str = self.replace_user_mentions(response["content"], original_message) if original_message else response["content"]
                        content_str = mention_prefix + content_str
                        await self.send_chunked_message(channel, content_str)
                    if response.get("image_url"):
                        await channel.send(response["image_url"])

                elif response:
                    mention_prefix = get_mention_prefix()
                    response_text = mention_prefix + self.replace_user_mentions(response, original_message) if original_message else response
                    await self.send_chunked_message(channel, response_text)

                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing message from queue: {e}", exc_info=True)
                try:
                    error_message = f"Sorry, I encountered an error while processing your message: {str(e)}"
                    await channel.send(error_message)
                except:
                    logger.error("Failed to send error message to channel")
                finally:
                    self.message_queue.task_done()
                    continue

    def is_supported_image(self, filename):
        """Check if the file is a supported image type."""
        return any(filename.lower().endswith(ext) for ext in self.supported_image_types)

    def encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def process_image(self, image_path, prompt):
        """Process image using Gemini API."""
        if not self.client:
            return "Image processing is not configured for this bot."

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

    def build_user_map(self, channel_history):
        """Build map of usernames to user IDs."""
        self.user_map = {
            msg['username'].lower(): msg['user_id']
            for msg in channel_history
        }

    def format_history(self, channel_history):
        """Format channel history into a text block."""
        history_text = '\n'.join(
            [f"{msg['timestamp']}: {msg['username']}: {msg['content']}" for msg in channel_history]
        )
        history_text += f"\nTotal messages fetched: {len(channel_history)}"
        return history_text

    async def start_bot(self):
        """Start the bot with message queue if enabled."""
        if self.message_queue:
            await asyncio.gather(
                self.start(self.token),
                self.process_message_queue()
            )
        else:
            await self.start(self.token)

class ClaudeBot(MultiBot):
    """Enhanced Claude bot with advanced file handling and analysis capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_cache = {}  # Cache for file analyses
        self.conversation_memory = {}  # Enhanced memory per channel

    async def handle_message(self, message):
        """Enhanced message handling for Claude with file support"""
        async with message.channel.typing():
            user_id = message.author.id
            user_name = self.ai_users.get(user_id, self.allowed_users.get(user_id, str(user_id)))
            user_message = message.content.strip()

            # Skip if in specific channels
            if message.channel.id == 775589136752443403:
                return

            # Initialize conversation memory for this channel
            channel_id = message.channel.id
            if channel_id not in self.conversation_memory:
                self.conversation_memory[channel_id] = {
                    "file_history": [],
                    "topics": [],
                    "last_activity": datetime.now()
                }

            # Process any attachments
            attachments_to_process = []
            if message.attachments:
                logger.info(f"Claude processing {len(message.attachments)} attachments")
                attachments_to_process = message.attachments

                # Store file info in conversation memory
                for att in message.attachments:
                    self.conversation_memory[channel_id]["file_history"].append({
                        "filename": att.filename,
                        "type": os.path.splitext(att.filename)[1],
                        "timestamp": datetime.now().isoformat(),
                        "user": user_name
                    })

            # Check for special commands
            if user_message.startswith("!analyze"):
                # Analyze all recent files in conversation
                await self.analyze_conversation_files(message)
                return
            elif user_message.startswith("!summary"):
                # Provide conversation summary
                await self.provide_conversation_summary(message)
                return
            elif user_message.startswith("!export"):
                # Export analysis results
                await self.export_analysis(message)
                return

            # Get channel history
            channel_history_text = await fetch_messages(message.channel.id, self.token)
            history_text = self.format_history(channel_history_text)

            # Build user map for mention replacement
            self.build_user_map(channel_history_text)

            # Get response with attachments
            response = await get_anthropic_response(
                claude_system_prompt=self.system_prompt,
                history_text=history_text,
                user_input=user_message,
                attachments=attachments_to_process
            )

            # Process response
            if response:
                await self.send_claude_response(message, response)
            else:
                await message.channel.send("*sighs* something went wrong with my processing. maybe try again?")

    async def send_claude_response(self, message, response):
        """Send Claude's response with proper formatting"""
        if hasattr(response, 'content') and isinstance(response.content, list):
            response_parts = []

            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        response_parts.append(block.text)

            if response_parts:
                full_response = '\n'.join(response_parts)
                mention_prefix = f"{message.author.mention} "
                full_response = mention_prefix + self.replace_user_mentions(full_response, message)

                # Claude's responses often have rich formatting, preserve it
                await self.send_chunked_message(message.channel, full_response)

    async def analyze_conversation_files(self, message):
        """Analyze all files shared in the conversation"""
        channel_id = message.channel.id
        if channel_id not in self.conversation_memory:
            await message.channel.send("*shrugs* no files to analyze in this conversation yet")
            return

        file_history = self.conversation_memory[channel_id]["file_history"]
        if not file_history:
            await message.channel.send("*looks around* haven't seen any files here")
            return

        analysis_msg = "## file analysis summary\n\n"
        analysis_msg += f"found **{len(file_history)}** files in this conversation:\n\n"

        for file_info in file_history[-10:]:  # Last 10 files
            analysis_msg += f"- `{file_info['filename']}` ({file_info['type']}) by {file_info['user']}\n"

        await self.send_chunked_message(message.channel, analysis_msg)

    async def provide_conversation_summary(self, message):
        """Provide a summary of the conversation including files and topics"""
        channel_id = message.channel.id
        memory = self.conversation_memory.get(channel_id, {})

        summary = "## conversation summary\n\n"

        if memory.get("file_history"):
            summary += f"**files shared:** {len(memory['file_history'])}\n"
            file_types = {}
            for f in memory['file_history']:
                ext = f['type']
                file_types[ext] = file_types.get(ext, 0) + 1

            summary += "**file types:**\n"
            for ext, count in file_types.items():
                summary += f"- {ext}: {count}\n"

        summary += f"\n*confidence: ~85%*"

        await self.send_chunked_message(message.channel, summary)

    async def export_analysis(self, message):
        """Export analysis results as a markdown file"""
        channel_id = message.channel.id
        memory = self.conversation_memory.get(channel_id, {})

        export_content = f"# Discord Analysis Export\n"
        export_content += f"**Channel:** {message.channel.name}\n"
        export_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if memory.get("file_history"):
            export_content += "## Files Analyzed\n\n"
            for f in memory['file_history']:
                export_content += f"- {f['filename']} ({f['type']}) - {f['timestamp']}\n"

        # Save to file
        filename = f"analysis_export_{channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(export_content)

        # Send file
        with open(filename, 'rb') as f:
            await message.channel.send(
                "*exports analysis results*",
                file=discord.File(f, filename=filename)
            )

        # Cleanup
        os.remove(filename)

async def main():
    bots = []
    server = None
    try:
        # Start single unified server
        server = start_server()
        # Wait for server to start
        await asyncio.sleep(5)
        logger.info("Server started successfully")

        logger.info(f"Allowed channels: {allowed_channels}")
        logger.info(f"Allowed users: {allowed_users}")

        # Initialize OpenAI client for image processing bots
        openai_client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # Configure bots with their specific requirements
        bot_configs = [
            # Tommy - Standard MultiBot
            {
                'class': MultiBot,
                'name': 'Tommy',
                'prompt': tommy_system_prompt,
                'response_func': tommy_get_response,
                'token': TOMMY_BOT_TOKEN,
                'kwargs': {}
            },
            # Bob - MultiBot with user mention handling
            {
                'class': MultiBot,
                'name': 'Bob',
                'prompt': bob_system_prompt,
                'response_func': bob_get_response,
                'token': BOB_BOT_TOKEN,
                'kwargs': {'intents': discord.Intents.all()  # Need all intents for reaction handling
                          }
            },
            # Claude - MultiBot with message queue
            {
                'class': ClaudeBot,  # Use the enhanced ClaudeBot class
                'name': 'Claude',
                'prompt': claude_system_prompt,
                'response_func': get_anthropic_response,
                'token': CLAUDE_BOT_TOKEN,
                'kwargs': {
                    'use_queue': True,
                    'openai_client': openai_client  # Pass the client for potential cross-functionality
                }
            }
        ]
        # Initialize bots with error handling
        for config in bot_configs:
            try:
                bot = config['class'](
                    name=config['name'],
                    system_prompt=config['prompt'],
                    get_response_func=config['response_func'],
                    token=config['token'],
                    **config['kwargs']
                )
                bots.append(bot)
                logger.info(f"Successfully initialized {config['name']} bot")
            except Exception as e:
                logger.error(f"Failed to initialize {config['name']} bot: {e}", exc_info=True)
                raise

        # Start all bots
        await asyncio.gather(*(bot.start_bot() for bot in bots))

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
        if 'server' in locals() and server and server.is_alive():
            logger.info("Shutting down server...")
            server.join(timeout=1.0)
            logger.info("Server shutdown complete")

if __name__ == "__main__":
    try:
        # Validate tokens before starting
        required_tokens = {
            'TOMMY_BOT_TOKEN': TOMMY_BOT_TOKEN,
            'BOB_BOT_TOKEN': BOB_BOT_TOKEN,
            'CLAUDE_BOT_TOKEN': CLAUDE_BOT_TOKEN,
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
            'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY
        }

        missing_tokens = [name for name, token in required_tokens.items() if not token]
        if missing_tokens:
            logger.critical(f"Missing required environment variables: {', '.join(missing_tokens)}")
            print(f"ERROR: Missing required tokens: {', '.join(missing_tokens)}")
            exit(1)

        logger.info("All required tokens found. Starting bot application.")
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        print(f"Critical error: {str(e)}")
