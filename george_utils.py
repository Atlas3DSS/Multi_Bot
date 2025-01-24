from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration
MODEL_ID = "gemini-2.0-flash-thinking-exp-01-21"

# Define safety settings
safety_settings = [
    types.SafetySetting(
        category='HARM_CATEGORY_HARASSMENT',
        threshold='BLOCK_NONE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_HATE_SPEECH',
        threshold='BLOCK_NONE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold='BLOCK_NONE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold='BLOCK_NONE',
    )
]

# Define search tool
search_tool = {'google_search': {}}

# Configure the client
client = genai.Client(
    http_options={'api_version': 'v1alpha'},
    api_key=GEMINI_API_KEY
)

def handle_response(response):
    """Handles the response from the model, including grounding metadata."""
    print("DEBUG - Full response:", response)  # Keep debug print for monitoring
    response_text = ""
    search_info = []

    for candidate in response.candidates:
        # Get main response text
        for part in candidate.content.parts:
            if part.text:
                print("Model:", part.text)
                response_text += part.text

        # Extract search information
        if candidate.grounding_metadata:
            # Get search queries used
            if candidate.grounding_metadata.web_search_queries:
                search_info.append("Search queries used:")
                for query in candidate.grounding_metadata.web_search_queries:
                    search_info.append(f"- {query}")

            # Get sources referenced
            if candidate.grounding_metadata.grounding_chunks:
                search_info.append("\nSources checked:")
                seen_sources = set()
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if chunk.web and chunk.web.title not in seen_sources:
                        search_info.append(f"- {chunk.web.title}")
                        seen_sources.add(chunk.web.title)

    # Format the final output
    final_text = response_text

    if search_info:
        final_text += "\n\nSearch Information:\n" + "\n".join(search_info)

    return final_text

async def get_response(user_input, history_text, george_system_prompt):
    """Main response function with history and safety settings."""
    try:
        # Create chat with complete configuration
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings,
                system_instruction=george_system_prompt,
            )
        )

        # Format conversation history if present
        if history_text:
            chat.send_message("heres the chat history for the current channel:" + history_text)

        # Send user message
        response = chat.send_message(
            f"{user_input}\n"
        )

        return handle_response(response)

    except Exception as e:
        logging.error(f"Error in get_response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

# Optional test function
def test_chat():
    """Test function for direct interaction."""
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                tools=[search_tool],
                safety_settings=safety_settings
            )
        )

        print("Start chatting (type 'exit' to quit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = chat.send_message(
                user_input + "\nPlease search for and include current information about this topic."
            )
            result = handle_response(response)
            print(result)

    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_chat()