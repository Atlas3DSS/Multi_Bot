from openai import OpenAI
import logging
import os
import openai
from prompts import tommy_system_prompt
from colorama import Fore, Style

#HYPERBOLIC_API_KEY = os.environ['HYPERBOLIC_API_KEY']
#LAMBDA_LABS_API_KEY = os.environ['LAMBDA_LABS_API_KEY']
##HYPERBOLIC##
#client = openai.OpenAI(
#    api_key=HYPERBOLIC_API_KEY,
#    base_url="https://api.hyperbolic.xyz/v1",
#)
##LAMBDA LABS##
XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)
#client = openai.OpenAI(
#    api_key=LAMBDA_LABS_API_KEY,
#    base_url="https://api.lambdalabs.com/v1",
#)
#DeepSeek_API_Key = os.getenv("DeepSeek_API_Key")
#client = OpenAI(api_key= DeepSeek_API_Key, base_url="https://api.deepseek.com")


system_prompt = tommy_system_prompt
async def get_response(user_input, history_text, system_prompt, discord_channel=None):
    messages = [
        # Phase 1: Identity Reinforcement
        {
            'role': 'user',
            'content': f"{system_prompt}\n"
        },
        {
            'role': 'assistant',
            'content': "Tommy, at your service."
        },

        # Phase 2: History Injection
        {
            'role': 'user',
            'content': f"[History Start]\n{history_text}\n[History End]\n\n"
        },
        {
            'role': 'assistant',
            'content': "History absorbed. Ready for your command." 
        },

        # Phase 3: Task Focus
        {
            'role': 'user',
            'content': f"{user_input}\n"
        }
    ]
    # Call the API with the messages
    response = client.chat.completions.create(
        model="grok-3-beta", messages=messages)
    response_content = response.choices[0].message.content
    #reasoning_conteSnt = response.choices[0].message.reasoning_content
    #print(Fore.GREEN + str(reasoning_content) + Style.RESET_ALL)
    print(Fore.RED + str(response) + Style.RESET_ALL)

    return response_content

