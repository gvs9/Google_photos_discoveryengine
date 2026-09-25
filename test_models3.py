from groq import Groq
from config.settings import settings
import time
import json

client = Groq(api_key=settings.groq_api_key)

model = "allam-2-7b"

BATCH_PROMPT_TEMPLATE = """Analyze these {n} user posts about Google Photos. For each post, extract the fields below.

Posts:
[0] \"\"\"I can't find my photos from my trip to Hawaii last year. I searched 'Hawaii' and 'beach' but nothing comes up. So frustrating!\"\"\"
[1] \"\"\"Google Photos is great, but it drains my battery.\"\"\"

Return a JSON object with this exact structure:
{{
  "extractions": [
    {{
      "index": 0,
      "is_retrieval_related": true or false,
      "retrieval_problem": "<1-2 sentence summary or null>",
      "memory_cues": ["<what they remember about the photo>"],
      "forgotten_info": ["<what info they lack>"],
      "search_attempts": ["<what they tried in Google Photos>"],
      "outcome": "found" or "not_found" or "unknown",
      "frustration_level": <integer 1-5>
    }}
  ]
}}

Include one entry per post, using the same index as given. Use [] for empty lists."""

try:
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Always respond with valid JSON only."},
            {"role": "user", "content": BATCH_PROMPT_TEMPLATE.format(n=2)},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    content = r.choices[0].message.content.strip()
    elapsed = time.time() - t0
    print(f"OK  {model} ({elapsed:.1f}s):\n{content}")
except Exception as e:
    print(f"ERR {model}: {e}")
