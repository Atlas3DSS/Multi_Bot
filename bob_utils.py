from openai import OpenAI
import logging
import json
import aiohttp
import os
from dotenv import load_dotenv
import httpx
from colorama import Fore, Style

# Create module-level logger
logger = logging.getLogger(__name__)

client = OpenAI()
client.http_client = httpx.Client(
    transport=httpx.HTTPTransport(retries=3),
    timeout=60.0,
    event_hooks={
        "request": [lambda r: logger.info(f"Request: {r.method} {r.url}")],
        "response": [lambda r: logger.info(f"Response: {r.status_code} {r.reason_phrase}")],
    }
)

history = []  # Initialize an empty list to store the conversation history


async def get_response(user_input, history_text, bob_system_prompt, discord_channel=None, image_urls=None):
    logger.info("Starting get_response function for Bob")
    logger.info(f"Discord channel present: {discord_channel is not None}")
    logger.info(f"Image URLs provided: {image_urls is not None}")

    # Define tools Bob can use (currently just web search)
    tools = [{ "type": "web_search_preview",
              "search_context_size": "high", }]

    try:
        logger.info("Preparing input payload for OpenAI responses API")

        # Prepare the base text input including system prompt, history, and user query
        base_text_input = (
            f"System Prompt: {bob_system_prompt}\n\n"
            f"[History Start]\n{history_text}\n[History End]\n\n"
            f"Format responses using Discord message styling\n"
            f"Maintain consistency with chat history context\n"
            f"*vibes intensify*\n\n"
            f"User Query: {user_input}"  # user_input should be the text part
        )

        # Start building the content list with the text input
        input_content = [{"type": "input_text", "text": base_text_input}]

        # Add image URLs to the content list if they exist
        if image_urls:
            for url in image_urls:
                input_content.append({"type": "input_image", "image_url": url})
                logger.info(f"Added image URL to payload: {url}")

        # Structure the final input payload for the API
        input_payload = [{"role": "user", "content": input_content}]

        logger.debug("Constructed input payload: %s", input_payload)

        logger.info("Making API call to OpenAI using the responses API")
        response = client.responses.create(
            model='gpt-4.1-2025-04-14', 
            input=input_payload,  # Pass the constructed payload
            tools=tools
        )

        logger.debug("OpenAI API response: %s", response)

        # Extract the output text from the response
        final_response = {"content": "", "image_url": None} # Assuming Bob doesn't generate images himself here
        ##LOG THIS TO CONSOLE IN GREEN
        logger.info(f"{Fore.GREEN}OpenAI API response for Bob: {response}{Style.RESET_ALL}") # Added colorama logging

        # Process the output which now comes as a list of output items
        for output_item in response.output:
            if output_item.type == "message":
                # Extract text content from message
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        final_response["content"] = content_item.text
                        break # Assuming only one text output block per message

        # If no content was found in the structured output, check for output_text directly (fallback)
        if not final_response["content"] and hasattr(response, "output_text"):
            logger.warning("Output found in response.output_text instead of structured output.")
            final_response["content"] = response.output_text

        # Bob doesn't generate images in this flow, so image_url remains None

        logger.info(f"Final response content for Bob: {final_response['content'][:100]}...") # Log snippet
        return final_response

    except Exception as e:
        logger.error("Error in get_response for Bob: %s", e, exc_info=True)
        return {
            "content": "I apologize, but I encountered an error processing your request.",
            "image_url": None
        }
