
import requests
import json

url = "https://ark-cn-beijing.bytedance.net/api/v3/embeddings/multimodal"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 4e7c61ff-d803-4363-acb8-3a7206d0a111"
}
payload = {
    "model": "ep-20260111190334-rst5s",
    "input": [
        {
            "type": "text",
            "text": "探测向量维度"
        }
    ]
}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    embedding = data['data'][0]['embedding']
    print(f"DIMENSION: {len(embedding)}")
except Exception as e:
    print(f"ERROR: {e}")
    if 'response' in locals():
        print(f"RESPONSE: {response.text}")
