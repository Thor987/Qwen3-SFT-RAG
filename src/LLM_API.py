# -*- coding: utf-8 -*-
from openai import OpenAI
def LLM(model_name, content):
    api_key = 'XX'
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.bianxie.ai/v1"
    )
    completion = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )
    return completion.choices[0].message
if __name__ == "__main__":
    model_name = "gemini-2.5-pro-preview-06-05"
    content = '''XXXXXXXXX'''

    response = LLM(model_name, content)
    print(response)

