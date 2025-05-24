from dotenv import load_dotenv
load_dotenv()
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
from george_utils import get_response as george_get_response
from prompts import tommy_system_prompt, bob_system_prompt, claude_system_prompt, george_system_prompt
from allowed import allowed_channels, allowed_users
import anthropic
import uuid # Added missing import that was used in handle_image_dm_reaction
from pathlib import Path # Added missing import
import tempfile # Added missing import

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
                if channel_data.get('type') == 11: # Thread
                    total_limit = min(total_limit, 50)

    headers = {'Authorization': f'Bot {bot_token}'}
    messages = []
    last_message_id = None

    while len(messages) < total_limit:
        params = {'limit': min(50, total_limit - len(messages))} # Fetch in batches of 50 or remaining
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
    
    # Reverse messages to be in chronological order
    messages.reverse()

    # If exclude_last is True and we have messages, remove the most recent one (which is now at the end after reversing)
    if exclude_last and messages:
        messages = messages[:-1]

    logger.info(f"Total messages finalized for history: {len(messages)}")

    formatted_messages = [
        {
            'content': msg.get('content', ''),
            'timestamp': msg.get('timestamp', ''),
            'username': msg['author']['username'],
            'user_id': msg['author']['id'],
            'attachments': [attachment['url'] for attachment in msg.get('attachments', [])]
        }
        for msg in messages # Iterate over the potentially sliced list
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
        # Image DM feature configuration
        image_dm_config = kwargs.get('image_dm_config', {})
        self.image_dm_enabled = image_dm_config.get('enabled', False)
        self.image_dm_channel = image_dm_config.get('channel_id', 1019641944181329941)
        self.image_dm_emoji = image_dm_config.get('emoji', '🖼️')
        self.processed_reactions = set()

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


    async def handle_image_dm_reaction(self, payload):
        """Handle emoji reactions for DMing images with UUID rename"""
        # Only process if it's the configured emoji in the configured channel
        # Check if feature is enabled for this bot
        if not self.image_dm_enabled:
            return
        if payload.channel_id != self.image_dm_channel:
            return
        
        if str(payload.emoji) != self.image_dm_emoji:
            return
        
        # Don't process bot's own reactions
        if payload.user_id == self.user.id:
            return
        
        # Avoid processing the same reaction twice
        reaction_key = f"{payload.message_id}-{payload.user_id}-{payload.emoji}"
        if reaction_key in self.processed_reactions:
            return
        self.processed_reactions.add(reaction_key)
        
        try:
            # Get the channel and message
            channel = self.get_channel(payload.channel_id)
            if not channel:
                return
            
            message = await channel.fetch_message(payload.message_id)
            if not message.attachments:
                return
            
            # Get the user who reacted
            user = await self.fetch_user(payload.user_id)
            if not user:
                return
            
            # Process each image attachment
            images_sent = 0
            for attachment in message.attachments:
                # Check if it's an image
                if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    continue
                
                try:
                    # Download the image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as response:
                            if response.status != 200:
                                continue
                            
                            image_data = await response.read()
                    
                    # Generate UUID and new filename
                    short_uuid = str(uuid.uuid4())[:6]
                    file_extension = Path(attachment.filename).suffix
                    new_filename = f"{Path(attachment.filename).stem}_{short_uuid}{file_extension}"
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_file.write(image_data)
                        temp_path = temp_file.name
                    
                    try:
                        # Send to user's DM
                        with open(temp_path, 'rb') as f:
                            await user.send(
                                f"Here's the image you requested from {channel.name}:",
                                file=discord.File(f, filename=new_filename)
                            )
                        images_sent += 1
                        
                        logger.info(f"Sent image {new_filename} to {user.name} via DM")
                        
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                            
                except discord.Forbidden:
                    logger.error(f"Cannot send DM to user {user.name} - DMs may be disabled")
                    # Optionally, react to the message to indicate DM failed
                    try:
                        await message.add_reaction('❌')
                    except:
                        pass
                    break
                except Exception as e:
                    logger.error(f"Error processing image {attachment.filename}: {e}")
                    continue
            
            # Add a checkmark if images were sent successfully
            if images_sent > 0:
                try:
                    await message.add_reaction('✅')
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in handle_image_dm_reaction: {e}", exc_info=True)

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
                        async for msg in message.channel.history(limit=100): # Adjusted limit to ensure it can fetch enough messages
                            if deleted >= count:
                                break
                            if msg.author == self.user:
                                await msg.delete()
                                deleted += 1
                                await asyncio.sleep(0.5)  # Delay to avoid rate limit
                        await message.add_reaction('✅') # React to the command message
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
        # Process image DM reactions first, if applicable for any bot
        if self.image_dm_enabled and payload.channel_id == self.image_dm_channel and str(payload.emoji) == self.image_dm_emoji:
            await self.handle_image_dm_reaction(payload)
            # If it was an image DM reaction, don't process further for MMF (Bob specific)
            if self.name != "Bob": # Or some other logic if other bots might use this hook differently
                return


        # Only process reactions for Bob for MMF username collection
        if self.name != "Bob":
            return

        # Always process specific message ID for MMF
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
                if user and not user.bot: # Ensure user is not a bot
                    try:
                        await user.send("""Hello! I'm messaging you because you reacted to our D'Jenacy announcement;
if you want our new Devil girl model added to your MyMiniFactory library, please type your MyMiniFactory USERNAME without any additional words here:""")
                    except discord.Forbidden:
                        logger.error(f"Cannot send DM to user {payload.user_id}")
                return # Return after handling the specific message ID
            except Exception as e:
                logger.error(f"Error handling reaction for specific message: {e}", exc_info=True)
                return # Important to return to not fall through

        # Regular announcement message handling for MMF (Bob specific)
        if payload.message_id not in self.announcement_messages:
            return

        # Check if announcement message contains bot mention (Bob's user ID)
        announcement_content = self.announcement_messages[payload.message_id]
        # Ensure self.user is populated before accessing self.user.id
        if not self.user or not f'<@{self.user.id}>' in announcement_content:
            return

        # Check if the reaction is on a stored announcement message
        if payload.message_id in self.announcement_messages:
            try:
                # Don't process bot reactions (including self)
                if payload.user_id == self.user.id: # Check against self.user.id
                    return
                
                # Check if user is a bot
                reacting_user = await self.fetch_user(payload.user_id)
                if reacting_user and reacting_user.bot:
                    return

                # Check if the user ID is already present in mmf_usernames.csv
                async with aiofiles.open('mmf_usernames.csv', 'r') as file:
                    async for line in file:
                        if line.startswith(str(payload.user_id)):
                            logger.info(f"User {payload.user_id} already provided MMF username for this announcement.")
                            return

                # Track this user as needing to provide MMF username
                self.pending_mmf_usernames[payload.user_id] = payload.message_id

                # Get the user object and send DM
                user = await self.fetch_user(payload.user_id) # Already fetched as reacting_user
                if user: # No need to check for bot again, already done
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
                await f.readline() # Skip header
                async for line in f:
                    if line.strip():
                        user_id = line.split(',')[0]
                        existing_users.add(user_id)

            # Find missing users
            missing_users = reacted_users - existing_users

            if missing_users:
                # Log missing users to missing_ids.csv
                async with aiofiles.open('missing_ids.csv', 'a', newline='') as f:
                    for user_id_str in missing_users:
                        # Fetch user details to get username if possible, otherwise leave blank
                        user_obj = await self.fetch_user(int(user_id_str))
                        username = user_obj.name if user_obj else ""
                        await f.write(f"{user_id_str},{username},{datetime.now().isoformat()},false\n")


                logger.info(f"Found {len(missing_users)} users who need to provide MMF usernames")

                # Send DMs to missing users
                for user_id_str in missing_users:
                    user_id_int = int(user_id_str)
                    try:
                        # Track this user as needing to provide MMF username
                        self.pending_mmf_usernames[user_id_int] = target_message_id

                        # Get the user object and send DM
                        user = await self.fetch_user(user_id_int)
                        if user: # Already ensured not a bot when adding to reacted_users
                            try:
                                await user.send("""Hello! I'm messaging you because you reacted to our D'Jenacy announcement;
if you want our new Devil girl model added to your MyMiniFactory library, please type your MyMiniFactory USERNAME without any additional words here:""")
                                logger.info(f"Sent DM to user {user_id_str}")
                            except discord.Forbidden:
                                logger.error(f"Cannot send DM to user {user_id_str}")
                    except Exception as e:
                        logger.error(f"Error processing user {user_id_str}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error checking missing reactions: {e}", exc_info=True)

    async def handle_mmf_username_response(self, message):
        """Process MMF username responses in DMs."""
        try:
            username = message.content.strip()
            user_id = message.author.id
            # message_id = self.pending_mmf_usernames[user_id] # Not strictly needed here unless for logging context

            # Append to MMF usernames CSV file
            async with aiofiles.open(self.mmf_csv_path, 'a', newline='') as file:
                await file.write(f"{user_id},{username},{datetime.now().isoformat()},false\n")

            # Remove from pending list
            if user_id in self.pending_mmf_usernames: # Check before deleting
                del self.pending_mmf_usernames[user_id]

            # Remove from missing_ids.csv if present
            if os.path.exists('missing_ids.csv'):
                lines_to_keep = []
                async with aiofiles.open('missing_ids.csv', 'r') as file:
                    header = await file.readline() # Keep header
                    lines_to_keep.append(header)
                    async for line in file:
                        if not line.startswith(str(user_id)):
                            lines_to_keep.append(line)

                async with aiofiles.open('missing_ids.csv', 'w') as file:
                    await file.writelines(lines_to_keep)


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

           # Get channel history - CUSTOM FOR GEORGE
            if self.name == "George":
                channel_history_text = await fetch_messages(message.channel.id, self.token, total_limit=250)
            else:
                # Get channel history, exclude the current message being processed
                channel_history_text = await fetch_messages(message.channel.id, self.token, exclude_last=True)
            history_text = self.format_history(channel_history_text)


            # Build user map for mention replacement
            self.build_user_map(channel_history_text) # Pass the fetched history

            # Handle message queue if enabled
            if self.message_queue:
                await self.message_queue.put((user_message, message.channel, history_text, message)) # Pass original message for mentions
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
                thinking_file = None # This variable is not used for sending response
                if thinking_text:
                    # thinking_file = await self.save_thinking_process(thinking_text, message.id) # save_thinking_process not defined
                    logger.info(f"Thinking process for message {message.id}: {thinking_text}") # Log thinking
                if response_text:
                    mention_prefix = f"{message.author.mention} "
                    response_text = mention_prefix + self.replace_user_mentions(response_text, message)
                    await self.send_chunked_message(message.channel, response_text) # Removed thinking_file argument
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
                if isinstance(response, list): # Assuming response might be a list with one string element
                    response_text = response[0] if response else ""
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
        content_length = len(content)

        while current_pos < content_length:
            end_pos = current_pos + max_length
            if end_pos >= content_length:
                chunks.append(content[current_pos:])
                break

            # Try to split at a newline, then sentence-ending punctuation, then space
            split_point = -1
            # Prefer splitting at newlines first if possible
            newline_split = content.rfind('\n', current_pos, end_pos)
            if newline_split != -1:
                split_point = newline_split + 1 # Include the newline in the current chunk if splitting after it, or split before
            else:
                # If no newline, try sentence punctuation
                punctuation_splits = [
                    content.rfind('. ', current_pos, end_pos),
                    content.rfind('? ', current_pos, end_pos),
                    content.rfind('! ', current_pos, end_pos),
                ]
                valid_punctuation_splits = [p for p in punctuation_splits if p != -1]
                if valid_punctuation_splits:
                    split_point = max(valid_punctuation_splits) + 2 # Split after the punctuation and space
                else:
                    # If no sentence end, try space
                    space_split = content.rfind(' ', current_pos, end_pos)
                    if space_split != -1:
                        split_point = space_split + 1 # Split after the space
                    else:
                        # Force split if no natural break found
                        split_point = end_pos
            
            if split_point <= current_pos : # handles edge case where rfind returns 0 or a point before current_pos
                split_point = end_pos


            chunks.append(content[current_pos:split_point].strip())
            current_pos = split_point
        
        # Filter out any empty chunks that might have been created
        chunks = [c for c in chunks if c]


        for i, chunk in enumerate(chunks):
            try:
                # Prefix for multi-part messages, but not for single chunk messages
                prefix = f"[{i+1}/{len(chunks)}] " if len(chunks) > 1 else ""
                await channel.send(prefix + chunk)
                if i < len(chunks) - 1: # If not the last chunk
                    await asyncio.sleep(0.5) # Short delay between chunks
            except discord.errors.HTTPException as e:
                logger.error(f"Error sending message chunk {i+1}/{len(chunks)}: {str(e)}")
                try:
                    await channel.send("Error: Message chunk failed to send. Please try again.")
                except:
                    pass # Avoid error loop if even error message fails
                break # Stop sending further chunks on error
            # await asyncio.sleep(0) # This sleep(0) is likely not needed due to await channel.send

    def replace_user_mentions(self, response_text, message):
        """Replace @mentions with proper Discord mentions while preserving formatting."""
        if not hasattr(self, 'user_map') or not message: # Ensure message object is available
            return response_text


        lines = response_text.split('\n')
        final_lines = []

        for line in lines:
            words = line.split(' ') # Split by space to preserve multiple spaces if needed later by join
            final_words = []

            for w in words:
                if w.startswith('@'):
                    # Attempt to find username, removing potential trailing punctuation for matching
                    potential_username = w[1:]
                    cleaned_username = potential_username
                    trailing_punctuation = ''

                    # Common punctuation that might trail a mention
                    common_punctuation = ['.', ',', '!', '?', ':', ';']
                    if potential_username and potential_username[-1] in common_punctuation:
                        cleaned_username = potential_username[:-1]
                        trailing_punctuation = potential_username[-1]
                    
                    raw_name = cleaned_username.lower()

                    # Check existing user_map first
                    mentioned_user_id = self.user_map.get(raw_name)

                    # If not in map, try to find in guild members (if message is from a guild)
                    if not mentioned_user_id and message.guild is not None:
                        member = discord.utils.find(
                            lambda m: m.name.lower() == raw_name or (m.nick and m.nick.lower() == raw_name),
                            message.guild.members
                        )
                        if member:
                            mentioned_user_id = member.id
                            self.user_map[raw_name] = member.id # Cache for future use

                    if mentioned_user_id:
                        final_words.append(f"<@{mentioned_user_id}>{trailing_punctuation}")
                    else:
                        final_words.append(w) # Keep original word if no user found
                else:
                    final_words.append(w)

            final_lines.append(" ".join(final_words)) # Join with single space, assuming original spacing isn't critical between words


        return "\n".join(final_lines) # Join lines with newline

    async def process_message_queue(self):
        """Process messages in queue for bots that use message queues."""
        if not self.message_queue:
            return

        while True:
            try:
                # Unpack including the original message for mention context
                user_input, channel, history_text, original_message = await self.message_queue.get()

                response = await self.get_response(user_input, self.system_prompt, history_text)

                def get_mention_prefix():
                    if original_message and hasattr(original_message.author, 'mention'):
                        return f"{original_message.author.mention} "
                    return ""

                if hasattr(response, 'content') and isinstance(response.content, list): # Claude-like response
                    thinking_text = ""
                    response_content_parts = []

                    for block in response.content:
                        if hasattr(block, 'type'):
                            if block.type == 'text': # Assuming Claude uses 'text' for main content
                                text_content = block.text.strip()
                                if text_content:
                                    response_content_parts.append(text_content)
                            # Add handling for 'thinking' if it's a separate block type from Claude
                            # elif block.type == 'thinking':
                            # thinking_text = getattr(block, 'thinking', '') # Or however thinking is structured

                    if thinking_text: # This part needs Claude's actual thinking block structure
                        thinking_message = f"**Thinking Process:**\n```\n{thinking_text}\n```"
                        await self.send_chunked_message(channel, thinking_message)
                    
                    if response_content_parts:
                        mention_prefix = get_mention_prefix()
                        response_text = '\n'.join(response_content_parts)
                        # Apply user mention replacement to the combined text
                        response_text = self.replace_user_mentions(response_text, original_message) if original_message else response_text
                        final_response = mention_prefix + response_text
                        await self.send_chunked_message(channel, final_response)


                elif isinstance(response, tuple): # (thinking_text, response_text)
                    thinking_text, response_text = response
                    # thinking_file = None # save_thinking_process not defined
                    if thinking_text:
                        # thinking_file = await self.save_thinking_process(thinking_text, original_message.id if original_message else None)
                        logger.info(f"Thinking process for queued message: {thinking_text}") # Log thinking
                    if response_text:
                        mention_prefix = get_mention_prefix()
                        response_text_processed = self.replace_user_mentions(response_text, original_message) if original_message else response_text
                        final_response = mention_prefix + response_text_processed
                        await self.send_chunked_message(channel, final_response) # Removed thinking_file

                elif isinstance(response, dict): # For potential image_url responses
                    if response.get("content"):
                        mention_prefix = get_mention_prefix()
                        content_str = self.replace_user_mentions(response["content"], original_message) if original_message else response["content"]
                        content_str = mention_prefix + content_str
                        await self.send_chunked_message(channel, content_str)
                    if response.get("image_url"):
                        await channel.send(response["image_url"])

                elif response: # Standard text response
                    mention_prefix = get_mention_prefix()
                    response_text = str(response) # Ensure string
                    response_text_processed = self.replace_user_mentions(response_text, original_message) if original_message else response_text
                    final_response = mention_prefix + response_text_processed
                    await self.send_chunked_message(channel, final_response)

                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing message from queue: {e}", exc_info=True)
                if 'channel' in locals() and channel: # Check if channel is defined
                    try:
                        error_message = f"Sorry, I encountered an error while processing your message: {str(e)}"
                        await channel.send(error_message)
                    except Exception as send_error: # Catch error during sending error message
                        logger.error(f"Failed to send error message to channel: {send_error}")
                # Ensure task_done is called even if there's an error sending the message
                if self.message_queue: # Check if queue exists
                    self.message_queue.task_done()
                continue # Continue processing other messages

    def is_supported_image(self, filename):
        """Check if the file is a supported image type."""
        return any(filename.lower().endswith(ext) for ext in self.supported_image_types)

    def encode_image(self, image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def process_image(self, image_path, prompt):
        """Process image using Gemini API."""
        if not self.client: # Should be openai_client from kwargs, or Gemini specific client
            return "Image processing is not configured for this bot."


        try:
            # This method seems tailored for OpenAI's format. If using Gemini directly,
            # it would need genai.GenerativeModel("gemini-pro-vision") or similar.
            # The current self.client is passed as openai_client (Gemini with OpenAI wrapper).
            base64_image = self.encode_image(image_path)
            # Assuming the client is the OpenAI compatible one for Gemini
            response = self.client.chat.completions.create(
                model="models/gemini-pro-vision", # Or appropriate Gemini vision model
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} # Assuming jpeg, might need to detect type
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
        filename_lower = attachment.filename.lower()

        if filename_lower.endswith('.txt'):
            try:
                text_content_bytes = await attachment.read()
                text_content = text_content_bytes.decode('utf-8')
                # Sanitize or truncate if very long, depending on LLM input limits
                # For now, just append. Consider token limits.
                return f"This message is from {user_name}\n\n{user_input}\n\nAttached text file '{attachment.filename}':\n{text_content}"
            except Exception as e:
                logger.error(f"Error reading text attachment {attachment.filename}: {e}")
                return f"{user_input}\n[Error reading attached text file: {attachment.filename}]"


        # Check if it's a supported image type for Claude-style list payload
        # This part is more for models like Claude that accept a list of content blocks
        if attachment.content_type and attachment.content_type.startswith('image/'):
            if any(filename_lower.endswith(ext) for ext in self.supported_image_types):
                try:
                    # Download image data
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                image_data_bytes = await resp.read()
                                image_base64 = base64.b64encode(image_data_bytes).decode("utf-8")
                                
                                # Construct payload for multimodal LLM
                                return [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": attachment.content_type, # Use actual content type
                                            "data": image_base64,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        # Prepend user name to the text part if it's not already in user_input
                                        "text": f"User {user_name} says: {user_input}" if user_name not in user_input else user_input
                                    }
                                ]
                            else:
                                logger.error(f"Failed to download image {attachment.url}, status: {resp.status}")
                                return f"{user_input}\n[Failed to download image: {attachment.filename}]"
                except Exception as e:
                    logger.error(f"Error processing image attachment {attachment.filename}: {e}")
                    return f"{user_input}\n[Error processing image attachment: {attachment.filename}]"
        
        # If not a text or processed image, return original input with a note about the attachment
        return f"{user_input}\n[Attachment present: {attachment.filename} of type {attachment.content_type}]"


    def build_user_map(self, channel_history):
        """Build map of usernames to user IDs."""
        self.user_map = {} # Reset or update as needed
        for msg in channel_history:
            # Store lowercase username for case-insensitive matching
            self.user_map[msg['username'].lower()] = msg['user_id']
            # Potentially add display names/nicks if available and desired
            # if msg.get('member') and msg['member'].get('nick'):
            #     self.user_map[msg['member']['nick'].lower()] = msg['user_id']

    def format_history(self, channel_history):
        """Format channel history into a text block."""
        # Ensure channel_history is a list of dicts
        if not isinstance(channel_history, list):
            logger.error(f"format_history received non-list: {type(channel_history)}")
            return "Error: Could not format channel history."

        history_lines = []
        for msg in channel_history:
            if isinstance(msg, dict):
                ts = msg.get('timestamp', 'Unknown Time')
                un = msg.get('username', 'Unknown User')
                ct = msg.get('content', '')
                # attachments_info = ""
                # if msg.get('attachments'):
                #     attachments_info = f" [attachments: {len(msg['attachments'])}]"
                history_lines.append(f"{ts}: {un}: {ct}") # {attachments_info} - Decided against for brevity
            else:
                logger.warning(f"Skipping non-dict item in channel_history: {msg}")

        history_text = '\n'.join(history_lines)
        # history_text += f"\nTotal messages in formatted history: {len(channel_history)}" # Redundant with fetch_messages log
        return history_text


    async def start_bot(self):
        """Start the bot with message queue if enabled."""
        if self.message_queue:
            # Start the bot client and the queue processor concurrently
            await asyncio.gather(
                self.start(self.token),      # Starts the Discord client
                self.process_message_queue() # Starts the message queue consumer
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
                    "topics": [], # This would need actual topic extraction logic
                    "last_activity": datetime.now()
                }
            else: # Update last activity time
                self.conversation_memory[channel_id]["last_activity"] = datetime.now()


            # Process any attachments specifically for Claude's input format
            attachments_for_claude = []
            if message.attachments:
                logger.info(f"Claude processing {len(message.attachments)} attachments for message {message.id}")
                for att in message.attachments:
                    # Store file info in conversation memory
                    self.conversation_memory[channel_id]["file_history"].append({
                        "filename": att.filename,
                        "type": os.path.splitext(att.filename)[1].lower(), # Store extension
                        "timestamp": datetime.now().isoformat(),
                        "user": user_name,
                        "url": att.url # Store URL for potential direct use by Claude or for download
                    })

                    # Prepare attachments for Claude API (if it supports direct URLs or needs base64)
                    # This assumes get_anthropic_response can handle discord.Attachment objects
                    # or that we convert them here. For now, passing the raw attachment object.
                    attachments_for_claude.append(att)


            # Check for special commands (ensure these don't conflict with LLM interpretation)
            if user_message.lower().startswith("!analyze"): # Case insensitive
                await self.analyze_conversation_files(message)
                return
            elif user_message.lower().startswith("!summary"):
                await self.provide_conversation_summary(message)
                return
            elif user_message.lower().startswith("!export"):
                await self.export_analysis(message)
                return

            # Get channel history, excluding the current message
            channel_history_list = await fetch_messages(message.channel.id, self.token, exclude_last=True)
            history_text = self.format_history(channel_history_list)


            # Build user map for mention replacement
            self.build_user_map(channel_history_list) # Use the fetched history

            # Get response with attachments
            # Ensure get_anthropic_response is designed to handle attachments_for_claude correctly
            # (e.g., download and base64 encode them, or pass URLs if supported)
            response = await get_anthropic_response(
                claude_system_prompt=self.system_prompt,
                history_text=history_text,
                user_input=user_message,
                attachments=attachments_for_claude # Pass the prepared attachments
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
                if hasattr(block, 'type') and block.type == 'text': # Standard check for Claude's text blocks
                    response_parts.append(block.text)


            if response_parts:
                full_response = '\n'.join(response_parts).strip()
                if full_response: # Ensure there's content to send
                    mention_prefix = f"{message.author.mention} "
                    # Replace mentions in the combined response
                    full_response_processed = self.replace_user_mentions(full_response, message)
                    final_text_to_send = mention_prefix + full_response_processed
                    await self.send_chunked_message(message.channel, final_text_to_send)
                else:
                    logger.info("Claude's response was empty after processing blocks.")
                    # Optionally send a generic "I have no specific text response" message
            else: # No text blocks found
                 logger.info("Claude's response contained no text blocks.")
                 await message.channel.send("*seems to have thought but produced no words*")


    async def analyze_conversation_files(self, message):
        """Analyze all files shared in the conversation"""
        channel_id = message.channel.id
        if channel_id not in self.conversation_memory or not self.conversation_memory[channel_id]["file_history"]:
            await message.channel.send("*shrugs* no files to analyze in this conversation yet, or I don't remember them.")
            return

        file_history = self.conversation_memory[channel_id]["file_history"]
        
        analysis_msg = f"## File Analysis Summary for this Conversation ({message.channel.name})\n\n"
        analysis_msg += f"I recall **{len(file_history)}** file(s) being shared here:\n\n"

        # Display info for the last few files, e.g., last 5 or 10
        for file_info in file_history[-10:]: # Show up to the last 10 files
            timestamp_dt = datetime.fromisoformat(file_info['timestamp'])
            formatted_time = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
            analysis_msg += f"- `{file_info['filename']}` (type: {file_info['type']}), shared by **{file_info['user']}** around {formatted_time}\n"
        
        if len(file_history) > 10:
            analysis_msg += f"\n...and {len(file_history)-10} more older files."

        await self.send_chunked_message(message.channel, analysis_msg)


    async def provide_conversation_summary(self, message):
        """Provide a summary of the conversation including files and topics"""
        channel_id = message.channel.id
        memory = self.conversation_memory.get(channel_id)

        if not memory:
            await message.channel.send("*tilts head* I don't have any specific memories of this conversation to summarize yet.")
            return

        summary = f"## Conversation Summary for {message.channel.name}\n\n"
        summary += f"My last noted activity here was around: {memory['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}\n"


        if memory.get("file_history"):
            summary += f"**Files Shared:** {len(memory['file_history'])}\n"
            file_types = {}
            for f_info in memory['file_history']:
                ext = f_info.get('type', 'unknown').strip('.') or 'unknown' # Get extension, handle empty or '.'
                file_types[ext] = file_types.get(ext, 0) + 1

            if file_types:
                summary += "**Breakdown by File Types:**\n"
                for ext, count in sorted(file_types.items()): # Sort for consistent output
                    summary += f"- `{ext if ext else 'no extension'}`: {count}\n"
            else:
                summary += "No specific file types recorded for the shared files.\n"
        else:
            summary += "No files appear to have been shared in my memory of this conversation.\n"

        # Topic summary would require actual NLP topic modeling, which is complex.
        # For now, a placeholder or simple heuristic.
        if memory.get("topics") and memory["topics"]: # Check if topics list exists and is not empty
             summary += f"\n**Key Topics (simple recall):** {', '.join(memory['topics'][:5])}..." # Display first 5
        else:
            summary += "\nI haven't specifically cataloged key topics for this conversation yet."


        summary += f"\n\n*Confidence in this summary is based on my current memory state. Details might vary.*"

        await self.send_chunked_message(message.channel, summary)

    async def export_analysis(self, message):
        """Export analysis results as a markdown file"""
        channel_id = message.channel.id
        memory = self.conversation_memory.get(channel_id)

        if not memory:
            await message.channel.send("I don't have any analysis to export for this channel.")
            return

        export_content = f"# Discord Analysis Export for Channel: {message.channel.name}\n"
        export_content += f"**Generated by:** {self.name} ({self.user.name})\n"
        export_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        export_content += f"## Conversation Details\n"
        export_content += f"Last known activity: {memory['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"


        if memory.get("file_history"):
            export_content += "## Files Shared/Analyzed\n\n"
            export_content += "| Filename | Type | User | Timestamp | URL |\n"
            export_content += "|----------|------|------|-----------|-----|\n"
            for f_info in memory['file_history']:
                ts_dt = datetime.fromisoformat(f_info['timestamp'])
                formatted_ts = ts_dt.strftime('%Y-%m-%d %H:%M')
                file_type = f_info.get('type', 'N/A').strip('.')
                export_content += f"| `{f_info['filename']}` | {file_type} | {f_info['user']} | {formatted_ts} | [link]({f_info['url']}) |\n"
            export_content += "\n"
        else:
            export_content += "No files recorded in memory for this channel.\n\n"
        
        # Placeholder for topic data if it were implemented
        if memory.get("topics") and memory["topics"]:
            export_content += "## Identified Topics (Simple List)\n"
            for topic in memory["topics"]:
                export_content += f"- {topic}\n"
            export_content += "\n"
        else:
            export_content += "No specific topics were cataloged.\n\n"

        # Save to temporary file
        # tempfile.NamedTemporaryFile is better for auto-cleanup
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.md', encoding='utf-8') as tmp_file:
                tmp_file.write(export_content)
                tmp_file_path = tmp_file.name
            
            # Send file
            with open(tmp_file_path, 'rb') as f_to_send:
                await message.channel.send(
                    "*exports analysis results as requested...*",
                    file=discord.File(f_to_send, filename=f"analysis_export_{message.channel.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                )
        except Exception as e:
            logger.error(f"Error creating or sending export file: {e}")
            await message.channel.send("Sorry, I couldn't create the export file.")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path): # Ensure tmp_file_path is defined
                os.remove(tmp_file_path) # Clean up the temp file


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

        # Initialize OpenAI client for image processing bots (Gemini with OpenAI compatibility layer)
        openai_client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"], # Using GEMINI_API_KEY
            base_url="https://generativelanguage.googleapis.com/v1beta" # Corrected base URL
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
            # Bob - with different emoji for image DM and MMF features
            {
                'class': MultiBot,
                'name': 'Bob',
                'prompt': bob_system_prompt,
                'response_func': bob_get_response,
                'token': BOB_BOT_TOKEN,
                'kwargs': {
                    'intents': discord.Intents.all(), # Bob needs all intents for reactions etc.
                    'image_dm_config': {
                        'enabled': True,
                        'channel_id': 1019641944181329941, # Example channel ID
                        'emoji': '💾' # Bob's custom emoji
                    }
                    # openai_client could be passed if Bob also uses it for non-DM image processing
                }
            },
            # Claude - MultiBot with message queue
            {
                'class': ClaudeBot,  # Use the enhanced ClaudeBot class
                'name': 'Claude',
                'prompt': claude_system_prompt,
                'response_func': get_anthropic_response, # This is Anthropic's, not OpenAI/Gemini client
                'token': CLAUDE_BOT_TOKEN,
                'kwargs': {
                    'use_queue': True,
                    # Claude typically uses Anthropic API, openai_client might not be directly used by its get_response
                    # but could be passed if ClaudeBot class had other general image processing methods
                }
            },
            # George - New Gemini-powered bot with eigenrobot personality
            {
                'class': MultiBot,
                'name': 'George',
                'prompt': george_system_prompt,
                'response_func': george_get_response, # Assumed to use Gemini
                'token': GEORGE_BOT_TOKEN,
                'kwargs': {
                    'openai_client': openai_client # Pass the Gemini client (OpenAI compatible)
                }
            }
        ]
        # Initialize bots with error handling
        for config in bot_configs:
            try:
                # Prepare intents. Default if not specified, else use from kwargs.
                current_intents = config['kwargs'].get('intents', discord.Intents.default())
                current_intents.message_content = True # Ensure this is always True
                current_intents.dm_messages = True
                current_intents.reactions = True
                config['kwargs']['intents'] = current_intents


                bot = config['class'](
                    name=config['name'],
                    system_prompt=config['prompt'],
                    get_response_func=config['response_func'],
                    token=config['token'],
                    **config['kwargs'] # Pass all other kwargs including intents and openai_client
                )
                bots.append(bot)
                logger.info(f"Successfully initialized {config['name']} bot")
            except Exception as e:
                logger.error(f"Failed to initialize {config['name']} bot: {e}", exc_info=True)
                raise # Re-raise to stop main if a bot fails to init

        # Start all bots
        # Each bot.start_bot() will run its own event loop essentially for discord client
        # and potentially a queue processor if self.message_queue is enabled.
        # asyncio.gather will run all these start_bot() coroutines concurrently.
        await asyncio.gather(*(bot.start_bot() for bot in bots))


    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
        # Attempt graceful shutdown
        for bot in bots:
            try:
                if bot.is_ready(): # Check if bot was connected
                    await bot.close()
                    logger.info(f"Closed bot {bot.name}")
            except Exception as close_error:
                logger.error(f"Error closing bot {bot.name}: {close_error}")
        # raise # Re-raise the original error after attempting cleanup
    finally:
        # Cleanup any remaining resources
        if server and server.is_alive(): # Check if server was initialized and is alive
            logger.info("Shutting down web server...")
            server.terminate() # More forceful if join times out
            server.join(timeout=5.0) # Wait for server process to terminate
            if server.is_alive():
                logger.warning("Web server did not shut down gracefully, killing.")
                server.kill()
            logger.info("Web server shutdown process completed.")


if __name__ == "__main__":
    try:
        # Validate tokens before starting - ensure all necessary tokens are present
        # OPENAI_API_KEY is used for the Gemini client via OpenAI library wrapper
        required_env_vars = {
            'TOMMY_BOT_TOKEN': TOMMY_BOT_TOKEN,
            'BOB_BOT_TOKEN': BOB_BOT_TOKEN,
            'CLAUDE_BOT_TOKEN': CLAUDE_BOT_TOKEN,
            'GEORGE_BOT_TOKEN': GEORGE_BOT_TOKEN, # Added George's token
            'SYBIL_BOT_TOKEN': SYBIL_BOT_TOKEN,   # Added Sybil's token
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'), # Used for some bots, or as Gemini key
            'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY,
            'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY') # Explicit Gemini key
        }


        missing_vars = [name for name, var_val in required_env_vars.items() if not var_val]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.critical(error_msg)
            print(f"ERROR: {error_msg}")
            exit(1) # Exit if critical tokens/keys are missing

        logger.info("All required tokens and API keys found. Starting bot application.")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shut down by KeyboardInterrupt (Ctrl+C).")
    except Exception as e: # Catch-all for other exceptions during startup or shutdown
        logger.critical(f"Critical error in main application execution: {e}", exc_info=True)
        import traceback
        traceback.print_exc() # Print stack trace to console
        print(f"Critical error: {str(e)}")
        # exit(1) # Consider exiting with error code for critical failures
