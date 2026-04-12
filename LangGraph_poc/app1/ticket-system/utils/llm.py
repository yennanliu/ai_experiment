"""OpenAI client and chat helper"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI()
MODEL = "gpt-4o-mini"


def chat(system: str, user: str) -> str:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
