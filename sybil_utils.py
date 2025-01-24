import os
from google import generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from dotenv import load_dotenv
import logging
import aiohttp
import json
import io

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up the API key
genai.configure(credentials=GEMINI_API_KEY)

# Model configuration
generation_config = {
    "temperature": 1.69,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define safety settings
safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE',
    }
]

# Initialize the model with function calling capability
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="""You are Sybil - an AI assistant with a distinct personality marked by wisdom, wit, and creative insight. """,
    tools=[
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="generate_image",
                    description="Generates an image based on a text prompt",
                    parameters=content.Schema(
                        type=content.Type.OBJECT,
                        required=["prompt", "aspect_ratio", "model"],
                        properties={
                            "prompt": content.Schema(
                                type=content.Type.STRING,
                                description="A detailed visual description for image generation",
                            ),
                            "aspect_ratio": content.Schema(
                                type=content.Type.STRING,
                                enum=["ASPECT_1_1", "ASPECT_16_9", "ASPECT_9_16",
                                      "ASPECT_4_3", "ASPECT_3_4", "ASPECT_10_16"]
                            ),
                            "model": content.Schema(
                                type=content.Type.STRING,
                                enum=["V_2"]
                            ),
                        },
                    ),
                ),
            ],
        ),
    ],
    tool_config={'function_calling_config': 'AUTO'},
)

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
                "magic_prompt_option": "ON"
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_text = await response.text()
                if response.status == 200:
                    data = json.loads(response_text)
                    if 'data' in data and len(data['data']) > 0 and 'url' in data['data'][0]:
                        image_url = data['data'][0]['url']
                        logging.info("Image generated successfully. URL: %s", image_url)
                        return image_url
                    logging.error(f"Ideogram API returned unexpected data structure: {data}")
                    return None
                logging.error(f"Failed to generate image. Status: {response.status}, Response: {response_text}")
                return None
    except Exception as e:
        logging.error("An error occurred while generating the image: %s", e)
        return None

async def process_function_calls(response):
    """Process any function calls in the response and return text content and image URL separately."""
    image_url = None
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                fn = part.function_call
                if fn.name == "generate_image":
                    try:
                        prompt = fn.args.get("prompt")
                        aspect_ratio = fn.args.get("aspect_ratio", "ASPECT_1_1")
                        model = fn.args.get("model", "V_2")

                        image_url = await generate_image_ideogram(
                            prompt=prompt,
                            aspect_ratio=aspect_ratio,
                            model=model
                        )

                    except Exception as e:
                        logging.error(f"Error handling function call: {str(e)}")
                        return None
    return image_url

def get_text_content(response):
    """Extract text content from the response, excluding function calls."""
    text_content = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                text_content += part.text
    return text_content.strip()

async def handle_response(response):
    """Handles the response from the model, returning both text content and image URL."""
    text_content = get_text_content(response)
    image_url = await process_function_calls(response)
    return text_content, image_url

async def get_response(user_input, history_text=None, system_prompt=None):
    """Main response function returning both text and image URL."""
    try:
        chat = model.start_chat(history=[])

        if system_prompt:
            chat.send_message(system_prompt)

        if history_text:
            message = (
                f"Here's the chat history for the current channel:\n"
                f"{history_text}\n"
                f"This is the most recent user input: {user_input}"
            )
        else:
            message = user_input

        response = chat.send_message(message)
        return await handle_response(response)

    except Exception as e:
        logging.error(f"Error in get_response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again.", None