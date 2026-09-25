from groq import Groq
from config.settings import settings
import time

client = Groq(api_key=settings.groq_api_key)

models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b"]

prompt = """Analyze this user post about Google Photos:
\"\"\"
I can't find my photos from my trip to Hawaii last year. I searched 'Hawaii' and 'beach' but nothing comes up. So frustrating!
\"\"\"

Extract the following JSON (use null for missing fields, empty list [] if none apply):
{
  "is_retrieval_related": true or false,
  "retrieval_problem": "<1-2 sentence summary of the retrieval difficulty, or null>",
  "memory_cues": ["<cue the user remembers about the photo>"],
  "forgotten_info": ["<what info the user lacks to find it>"],
  "search_attempts": ["<what the user tried in Google Photos>"],
  "outcome": "found" or "not_found" or "unknown",
  "frustration_level": <integer 1-5>
}"""

for model in models:
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        content = r.choices[0].message.content.strip()
        elapsed = time.time() - t0
        print(f"OK  {model} ({elapsed:.1f}s): {content[:100]}...")
    except Exception as e:
        print(f"ERR {model}: {e}")
