import os
import logging
from google import genai
from google.genai import types
from colorama import Fore, Style
import asyncio

# Create module-level logger
logger = logging.getLogger(__name__)

# Initialize the client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

async def get_response(user_input, history_text, system_prompt, discord_channel=None):
    """Get response from George using Google's Genai API"""
    logger.info("Starting get_response function for George")
    
    try:
        # Build the conversation history
        contents = []
        
        # Add history context if available
        if history_text:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"here's our recent conversation history:\n{history_text}")
                    ],
                )
            )
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text="got it, i'll keep that context in mind")
                    ],
                )
            )
        
        # Add the current user input
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_input)
                ],
            )
        )
        
        # Configure tools
        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(google_search=types.GoogleSearch()),
        ]
        
        # Set up generation config with George's personality
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=system_prompt),
            ],
        )
        
        logger.info("Making API call to Google Genai")
        
        # Collect the streaming response
        response_text = ""
        
        # Run the async generator in a thread pool since the SDK might not be fully async
        def generate_sync():
            full_response = ""
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-pro-preview-05-06",  # Using the latest model
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    full_response += chunk.text
            return full_response
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, generate_sync)
        
        # Log the response
        logger.info(f"{Fore.MAGENTA}George response: {response_text[:100]}...{Style.RESET_ALL}")
        
        return {"content": response_text, "image_url": None}
        
    except Exception as e:
        logger.error(f"Error in get_response for George: {e}", exc_info=True)
        return {
            "content": "ugh, something borked. try again maybe? idk man",
            "image_url": None
        }
