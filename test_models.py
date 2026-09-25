from groq import Groq
from config.settings import settings
import json

client = Groq(api_key=settings.groq_api_key)

models_to_try = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
]

for model in models_to_try:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Always respond with valid JSON only."},
                {"role": "user", "content": 'Return this JSON: {"ok": true, "model": "working"}'},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        content = r.choices[0].message.content.strip()
        print(f"OK  {model}: {content[:80]}")
    except Exception as e:
        print(f"ERR {model}: {e}")
