import json
from pathlib import Path

import requests

MODEL_API_URL = "https://routerai.ru/api/v1"
KEYS_PATH = "../keys/keys.json"


def load_api_key(keys_path=KEYS_PATH):
    path = Path(keys_path)
    if not path.exists():
        raise FileNotFoundError(f"Keys file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_key = data.get("routerai_api_key")
    if not api_key:
        raise ValueError(f"'routerai_api_key' not found in {path}")

    return api_key.strip()


def send_api_request(
        messages,
        model,
        keys_path=KEYS_PATH,
        base_url=MODEL_API_URL,
        temperature=0.2,
        max_tokens=4096,
        timeout=60
):
    api_key = load_api_key(keys_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()
    result = response.json()

    if "choices" not in result or not result["choices"]:
        raise ValueError(f"Unexpected API response: {result}")

    return result


def extract_text(result):
    message = result["choices"][0]["message"]
    content = message.get("content", "")
    return content


def ask_model_response(
        promt,
        model,
        system_prompt=None,
        temperature=0.25,
        max_tokens=4096,
        timeout=60
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": promt})

    result = send_api_request(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    text = extract_text(result)
    usage = result.get("usage", {})

    return {
        "text": text,
        "usage": usage,
        "raw": result,
    }
