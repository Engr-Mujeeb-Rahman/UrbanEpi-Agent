"""Gemini provider. The key comes from .env and is never logged."""
import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


def client_from_env() -> genai.Client:
    load_dotenv(".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    return genai.Client(api_key=key)


def make_provider_call(model: str, temperature: float, retries: int = 4, wait_seconds: float = 20.0):
    """Return a function that sends one prompt and returns JSON text, retrying on rate limits, outages, and dropped connections."""
    client = client_from_env()
    settings = types.GenerateContentConfig(temperature=temperature, response_mime_type="application/json")

    def call(prompt: str) -> str:
        for attempt in range(retries + 1):
            try:
                return client.models.generate_content(model=model, contents=prompt, config=settings).text or ""
            except (errors.APIError, httpx.TransportError) as error:
                temporary = isinstance(error, httpx.TransportError) or error.code in (429, 500, 503)
                if not temporary or attempt == retries:
                    raise
                time.sleep(min(wait_seconds * (attempt + 1), 180))
        raise RuntimeError("Unreachable")

    return call
