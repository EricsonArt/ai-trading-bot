"""One door to every language model the brain and the researcher can use.

Choose with two settings (environment variables, a `.env` file in the project folder, or
GitHub repository variables/secrets for the cloud):

  BRAIN_PROVIDER=ollama     free local/open models (default): mimo2.6:9b on the PC, qwen3:4b in the cloud
  BRAIN_PROVIDER=anthropic  Claude, needs ANTHROPIC_API_KEY          e.g. BRAIN_MODEL=claude-sonnet-5
  BRAIN_PROVIDER=openai     any OpenAI-compatible API, needs OPENAI_API_KEY
                            (+ OPENAI_BASE_URL for OpenRouter, Groq, LM Studio, ...)  e.g. BRAIN_MODEL=gpt-5

Every call returns a dict that follows the given JSON schema.
"""

import json
import os

import requests

from . import config


def model_for(where):
    """The model name to use on the PC ('local') or in GitHub Actions ('cloud')."""
    if config.BRAIN_MODEL:
        return config.BRAIN_MODEL
    return config.LOCAL_BRAIN_MODEL if where == "local" else config.CLOUD_BRAIN_MODEL


def _extract(text):
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def _ollama(system, user, schema, model, timeout):
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": model, "stream": False, "think": False, "format": schema, "keep_alive": "2m",
        "options": {"temperature": 0.3, "num_ctx": 12288},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }, timeout=timeout)
    r.raise_for_status()
    return _extract(r.json()["message"]["content"])


def _anthropic(system, user, schema, model, timeout):
    # forcing a tool call makes Claude return arguments that match the schema
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, json={
        "model": model, "max_tokens": 4000, "temperature": 0.3, "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [{"name": "answer", "description": "Return your answer.", "input_schema": schema}],
        "tool_choice": {"type": "tool", "name": "answer"},
    }, timeout=timeout)
    r.raise_for_status()
    return next(b["input"] for b in r.json()["content"] if b["type"] == "tool_use")


def _openai(system, user, schema, model, timeout):
    base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    body = {"model": model, "temperature": 0.3,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    r = requests.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=timeout)
    if r.status_code == 400:  # some providers only support plain JSON mode
        body["response_format"] = {"type": "json_object"}
        body["messages"][0]["content"] += "\nReply with JSON matching this schema: " + json.dumps(schema)
        r = requests.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return _extract(r.json()["choices"][0]["message"]["content"])


PROVIDERS = {"ollama": _ollama, "anthropic": _anthropic, "openai": _openai}


def chat_json(system, user, schema, model, timeout=900):
    return PROVIDERS[config.BRAIN_PROVIDER](system, user, schema, model, timeout)
