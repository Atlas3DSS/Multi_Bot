from openai import OpenAI
import logging
import json
import aiohttp
import os
from dotenv import load_dotenv
import httpx

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

async def generate_meme(prompt):
    logging.info("Starting to generate meme...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="hd",
            n=1,
        )
        # Log the raw response object for debugging purposes
        logging.debug("Response received: %s", response)

        # Extract the image URL from the response object
        image_url = response.data[0].url  # Extract the URL

        logging.info("Meme generated successfully. URL: %s", image_url)
        return image_url
    except Exception as e:
        logging.error("An error occurred while generating the meme: %s", e)
        return None


async def generate_meme_2(prompt):
    logging.info("Starting to generate meme...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",
            quality="hd",
            n=1,
        )
        # Log the raw response object for debugging purposes
        logging.debug("Response received: %s", response)

        # Extract the image URL from the response object
        image_url = response.data[0].url  # Extract the URL

        logging.info("Meme generated successfully. URL: %s", image_url)
        return image_url
    except Exception as e:
        logging.error("An error occurred while generating the meme: %s", e)
        return None


async def describe_image_with_openai(bob_system_prompt, history_text, image_url,
                                     accompanying_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            'role': 'system',
            'content': bob_system_prompt,
        }, {
            'role':
            'user',
            'content':
            "I hope that clears everything up about who you are?"
        }, {
            'role':
            'assistant',
            'content':
            "It does, thank you so much. I realize now, I am Bob, it all makes sense now. "
        }, {
            'role':
            'user',
            'content':
            f'This is a chronological log of the chat you are particiatping in [start of history]{history_text}.[end of history]'
        }, {
            'role':
            'assistant',
            'content':
            "acknowledged, thank you for providing the chat history. "
        }, {
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": accompanying_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    },
                },
            ],
        }],
        max_tokens=3000,
    )

    print(response.choices[0])
    description_chunks = response.choices[0].message.content
    return description_chunks


async def generate_image_ideogram(prompt, aspect_ratio="ASPECT_1_1", model="V_2"):
    ideaogram_api_key = os.getenv("IDEAOGRAM_API_KEY")
    logging.info("Starting to generate image with Ideogram...")
    try:
        url = "https://api.ideogram.ai/generate"
        headers = {
            "Api-Key": ideaogram_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "image_request": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "model": model,
                "magic_prompt_option": "AUTO"
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_text = await response.text()
                print("Ideaogram response:", response_text)
                if response.status == 200:
                    data = json.loads(response_text)
                    #log response_text
                    logging.debug("Response received: %s", response_text)
                    # Updated to match actual API response structure
                    if 'data' in data and len(data['data']) > 0 and 'url' in data['data'][0]:
                        image_url = data['data'][0]['url']
                        logging.info("Image generated successfully. URL: %s", image_url)
                        return image_url
                    else:
                        logging.error(f"Ideogram API returned unexpected data structure: {data}")
                        return None
                logging.error(f"Failed to generate image. Status: {response.status}, Response: {response_text}")
                return None
    except Exception as e:
        logging.error("An error occurred while generating the image: %s", e)
        return None

async def get_response(user_input, history_text, bob_system_prompt, discord_channel=None, image_description=None):
    logger.info("Starting get_response function")
    tools = [{
        "type": "function",
        "function": {
            "name": "generate_image_ideogram",
            "description": "Generates an image based on a text prompt using the Ideogram AI API",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["ASPECT_1_1", "ASPECT_16_9", "ASPECT_9_16", "ASPECT_4_3", "ASPECT_3_4", "ASPECT_10_16"],
                        "description": "The aspect ratio for the generated image"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["V_1", "V_1_TURBO", "V_2", "V_2_TURBO"],
                        "description": "The model to use for generation"
                    }
                },
                "required": ["prompt"]
            }
        }
    }]

    messages = [
        {'role': 'system', 'content': bob_system_prompt},
        {'role': 'user', 'content': "I hope that clears everything up about who you are?"},
        {'role': 'assistant', 'content': "It does, thank you so much. I realize now, I am Bob, it all makes sense now."},
        {'role': 'user', 'content': f'This is a chronological log of the chat you are participating in [start of history]{history_text}.[end of history]'},
        {'role': 'assistant', 'content': "acknowledged, thank you for providing the chat history."},
        {'role': 'user', 'content': user_input}
    ]

    if image_description:
        logger.debug("Adding image description to messages")
        messages.append({'role': 'assistant', 'content': image_description})

    try:
        logger.info("Making initial API call to OpenAI")
        response = client.chat.completions.create(
            model='gpt-4o-2024-11-20',
            messages=messages,
            tools=tools
        )
        logger.debug("OpenAI API response: %s", response)
        print(response)

        message = response.choices[0].message
        final_response = {"content": "", "image_url": None}

        if message.tool_calls:
            logger.info("Processing tool calls")
            for tool_call in message.tool_calls:
                if tool_call.function.name == "generate_image_ideogram":
                    function_args = json.loads(tool_call.function.arguments)
                    prompt = function_args.get("prompt")
                    aspect_ratio = function_args.get("aspect_ratio", "ASPECT_1_1")
                    model = function_args.get("model", "V_2")

                    # Send initial status message
                    if discord_channel:
                        await discord_channel.send("Generating your image, please wait...")

                    logger.info(f"Attempting to generate image with prompt: {prompt}")
                    image_url = await generate_image_ideogram(prompt, aspect_ratio, model)
                    if image_url:
                        logger.info(f"Successfully generated image with URL: {image_url}")
                        final_response["image_url"] = image_url
                        messages.append({
                            "role": "assistant",
                            "content": f"I've generated an image for you: {image_url}"
                        })
                    else:
                        logger.warning("Image generation failed or returned no URL")

                    logger.info("Getting final completion with image context")
                    try:
                        final_completion = client.chat.completions.create(
                            model='gpt-4o-2024-11-20',
                            messages=messages
                        )
                        final_response["content"] = final_completion.choices[0].message.content
                        logger.info("Final response prepared successfully")
                        logger.debug("Final completion: %s", final_completion)
                        return final_response
                    except Exception as e:
                        logger.error("Error getting final completion: %s", e)
                        raise

        # If no tool calls, return original message
        logger.info("No tool calls detected, returning original message")
        final_response["content"] = message.content if message.content else ""
        return final_response

    except Exception as e:
        logger.error("Error in get_response: %s", e, exc_info=True)
        return {
            "content": "I apologize, but I encountered an error processing your request.",
            "image_url": None
        }