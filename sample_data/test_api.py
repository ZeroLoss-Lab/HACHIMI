#!/usr/bin/env python3
"""测试API连接"""

import requests

API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Translate '你好' to English. Just return the translation."}
    ],
    "temperature": 0.3,
    "max_tokens": 100
}

try:
    print(f"Testing API at: {API_URL}")
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
