import os
import re

import backoff
import openai
from .pricing import OPENAI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)

# Hermes-style thinking: prepend a system prompt that activates <think> tags,
# then strip them from the response so ShinkaEvolve sees clean code output.
HERMES_THINKING_ENABLED = os.getenv("HERMES_THINKING", "false").lower() == "true"
HERMES_THINKING_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model.

    Uses the Chat Completions API (/v1/chat/completions) for broad
    compatibility with OpenAI-compatible endpoints (vLLM, Nous, etc.).
    """
    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Optionally prepend Hermes thinking instructions to the system message
    effective_system_msg = system_msg
    if HERMES_THINKING_ENABLED:
        effective_system_msg = HERMES_THINKING_PROMPT + "\n\n" + system_msg

    messages = [
        {"role": "system", "content": effective_system_msg},
        *new_msg_history,
    ]

    # Translate Responses API kwargs to Chat Completions API format
    if "max_output_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        # Strip Hermes <think>...</think> blocks so ShinkaEvolve sees clean output
        if HERMES_THINKING_ENABLED:
            content = _THINK_TAG_RE.sub("", content).strip()
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        # response_model is handled by the instructor library, which wraps
        # the OpenAI client via instructor.from_openai(). The patched
        # create() accepts response_model and returns a Pydantic object.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=output_model,
            **kwargs,
        )
        content = response
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})

    usage = getattr(response, 'usage', None)
    if usage is not None:
        in_tok = usage.prompt_tokens or 0
        out_tok = usage.completion_tokens or 0
    else:
        in_tok = 0
        out_tok = 0

    input_cost = OPENAI_MODELS[model]["input_price"] * in_tok
    output_cost = OPENAI_MODELS[model]["output_price"] * out_tok
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result
