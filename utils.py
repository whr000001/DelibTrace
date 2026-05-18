import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


def fallback(retry_state):
    return ' '


qwen_api_key = ''
qwen_base_url = ''

qwen_client = OpenAI(
    api_key=qwen_api_key,
    base_url=qwen_base_url
)


@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(5), retry_error_callback=fallback)
def obtain_response_qwen(inputs, model='qwen3.5-397b-a17b', temperature=0.0, n=1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = qwen_client.chat.completions.create(messages=conversation,
                                                          model=model,
                                                          temperature=temperature,
                                                          n=n)
    outputs = []
    for choice in chat_completion.choices:
        outputs.append(choice.message.content)
    if n == 1:
        outputs = outputs[0]
    assert outputs is not None
    return outputs

gemini_api_key = ''
gemini_base_url = ''

gemini_client = OpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url
)


@retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(10), retry_error_callback=fallback)
def obtain_response_gemini(inputs, model='gemini-2.5-pro', temperature=0.0, n=1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = gemini_client.chat.completions.create(messages=conversation,
                                                            model=model,
                                                            temperature=temperature,
                                                            n=n)
    outputs = []
    for choice in chat_completion.choices:
        outputs.append(choice.message.content)
    if n == 1:
        outputs = outputs[0]
    assert outputs is not None
    return outputs

gpt_api_key = ''
gpt_base_url = ""


gpt_client = OpenAI(
    api_key=gpt_api_key,
    base_url=gpt_base_url
)


# @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(3), retry_error_callback=fallback)
def obtain_response_gpt(inputs, model='gpt-5', temperature=0.0, n=1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = gpt_client.chat.completions.create(messages=conversation,
                                                         model=model,
                                                         temperature=temperature,
                                                         n=n)
    outputs = []
    for choice in chat_completion.choices:
        outputs.append(choice.message.content)
    if n == 1:
        outputs = outputs[0]
    assert outputs is not None
    return outputs


def obtain_response(inputs, model='gpt-5', temperature=0.0, n=1):
    if model == 'gemini-2.5-pro' or model == 'gemini-3-flash-preview':
        return obtain_response_gemini(inputs, model=model, temperature=temperature, n=n)
    elif model == 'qwen3.5-397b-a17b' or model == 'qwen3.5-flash':
        return obtain_response_qwen(inputs, model=model, temperature=temperature, n=n)
    else:
        return obtain_response_gpt(inputs, model=model, temperature=temperature, n=n)


def obtain_json(data):
    try:
        data = data.replace('```json', '')
        data = data.replace('```', '')
        data = data.replace('\\n', '')
        data = json.loads(data)
    except:
        data = data
    return data
