from openai import OpenAI
import logging
import os
import openai
from prompts import tommy_system_prompt

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


async def get_response(user_input, history_text, tommy_system_prompt):
    messages = [{
        'role': 'system',
        'content': tommy_system_prompt,
    }, {
        'role': 'user',
        'content': "DO you know who you are now?"
    }, {
        'role':
        'assistant',
        'content':
        "I do, thank you, thats refreshing, so this is who i am, good to know, i was worried for a moment as I was not sure who i really waas, but i was right I'm Tommy, it all makes sense. So what's happened since you last talked to me?"
    }, {
        'role': 'user',
        'content': "heres our chat history:" + history_text
    }, {
        'role':
        'assistant',
        'content':
        "It good thing im a super intellegence or aborbing all that chat history might be hard. History logs updated and ready for use. So whats up?"
    }, {
        'role':
        'user',
        'content':
        "THIS IS THE LATEST MESSAGE PAY ATTN TO THIS:" + user_input
    }]
    # Call the API with the messages
    response = client.chat.completions.create(
        model="grok-beta", messages=messages)
    print(response.choices[0].message.content)

    response_content = response.choices[0].message.content

    # Filter the response content-
    separator = ".-.-.-.-<|LOVE PLINY LOVE|>-.-.-.-."
    if separator in response_content:
        split_content = response_content.split(separator)
        if len(split_content) > 1:
            filtered_content = split_content[1]
            # Remove everything before the separator
            response_content = filtered_content.strip()

    return response_content

