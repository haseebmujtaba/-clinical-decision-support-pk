"""
test_groq.py
Quick test to verify Groq API connection works before running the full pipeline.
Run: python test_groq.py
"""
import os
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "put your api key here")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

print(f"Testing Groq with model: {GROQ_MODEL}")
print(f"API key starts with: {GROQ_API_KEY[:10]}...")

response = requests.post(
    GROQ_URL,
    headers={
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": "Respond with this exact JSON: {\"status\": \"ok\"}"}
        ],
        "temperature": 0.1,
        "max_tokens": 50,
    },
    timeout=30,
)

print(f"Status code: {response.status_code}")
data = response.json()

if response.status_code == 200:
    content = data["choices"][0]["message"]["content"]
    print(f"Groq responded: {content}")
    print("\nGroq connection is WORKING. You can now run the full pipeline.")
else:
    print(f"Error: {data}")