
import os
from openai import OpenAI

api_key = "4e7c61ff-d803-4363-acb8-3a7206d0a111"
base_url = "https://ark-cn-beijing.bytedance.net/api/v3"
model = "ep-20251113195357-4gftp"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

print(f"Testing connection to {base_url} with model {model}...")

try:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
    )
    print("Connection successful!")
    print("Response:", completion.choices[0].message.content)
except Exception as e:
    print("Connection failed!")
    print("Error:", e)
