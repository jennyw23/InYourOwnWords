"""
Routes hypothesaes' LLM and embedding calls to DeepInfra's OpenAI-compatible API.
hypothesaes calls OpenAI's Responses API, which DeepInfra doesn't serve, so completions are
translated to Chat Completions. Point DEEPINFRA_BASE_URL at a local server (e.g. vLLM) to
run the same models yourself.
"""

import logging
import os
import time

import numpy as np
import openai

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError, openai.APIConnectionError)
# Responses-API-only kwargs that chat.completions rejects
UNSUPPORTED_KWARGS = ("verbosity", "text", "reasoning", "tokenizer_kwargs", "llm_sampling_kwargs")

_client = None


def client():
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set.")
        _client = openai.OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0)
    return _client


def no_thinking(model):
    """extra_body that turns off a reasoning model's thinking pass (otherwise it can use the
    whole token budget and return nothing)."""
    return {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in model.lower() else {}


def _retry(fn, label, max_retries=3, base_wait=1.0, backoff=2.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except RETRYABLE as e:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * backoff**attempt
            logger.warning("%s error (%s); retrying in %.1fs", label, type(e).__name__, wait)
            time.sleep(wait)


def get_completion(prompt=None, *, messages=None, model, timeout=None, system=None, **kwargs):
    """Drop-in replacement for hypothesaes.llm_api.get_completion."""
    messages = list(messages) if messages is not None else [{"role": "user", "content": prompt}]
    if system is not None and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": system}] + messages

    max_tokens = kwargs.pop("max_output_tokens", None) or kwargs.pop("max_completion_tokens", None)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    for k in UNSUPPORTED_KWARGS:
        kwargs.pop(k, None)
    if timeout is not None:
        kwargs["timeout"] = timeout

    def call():
        response = client().chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content or ""

    return _retry(call, f"completion ({model})", base_wait=timeout or 1.0)


def patch_hypothesaes(cache_dir):
    """Send hypothesaes' interpreter/annotator calls here and keep their cache in cache_dir."""
    import hypothesaes.annotate as annotate_mod
    import hypothesaes.interpret_neurons as interp_mod

    os.makedirs(cache_dir, exist_ok=True)
    # each module imported get_completion by name, so patch both
    for mod in (annotate_mod, interp_mod):
        mod.get_completion = get_completion
        mod.CACHE_DIR = cache_dir


def embed(texts, model, batch_size=96):
    """Embed texts in order -> array [len(texts), dim]."""
    from tqdm.auto import tqdm

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {model}"):
        batch = texts[i : i + batch_size]
        response = _retry(lambda: client().embeddings.create(model=model, input=batch),
                          f"embeddings ({model})", base_wait=3.0, backoff=3.0)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)
