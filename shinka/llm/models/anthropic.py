import backoff
import anthropic
from .pricing import CLAUDE_MODELS
from .result import QueryResult
import logging
import json

logger = logging.getLogger(__name__)


MAX_TRIES = 20
MAX_VALUE = 20

# JSON schema for structured diff output - guarantees Claude follows the format
DIFF_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "A shortened name summarizing the edit (lowercase, no spaces, underscores allowed)"
        },
        "description": {
            "type": "string",
            "description": "Description and argumentation process of the edit"
        },
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Exact code to find and replace (must match including indentation)"
                    },
                    "replace": {
                        "type": "string",
                        "description": "New replacement code"
                    }
                },
                "required": ["search", "replace"],
                "additionalProperties": False
            },
            "description": "List of SEARCH/REPLACE patches to apply"
        }
    },
    "required": ["name", "description", "patches"],
    "additionalProperties": False
}


def json_to_diff_format(json_response: dict) -> str:
    """Convert structured JSON response to standard DIFF format.

    This allows the rest of ShinkaEvolve to work unchanged - apply_diff.py
    will parse the standard format as expected.
    """
    parts = []

    # Add NAME section
    name = json_response.get("name", "unnamed_edit")
    parts.append(f"<NAME>\n{name}\n</NAME>")

    # Add DESCRIPTION section
    description = json_response.get("description", "")
    parts.append(f"\n<DESCRIPTION>\n{description}\n</DESCRIPTION>")

    # Add DIFF section with all patches
    patches = json_response.get("patches", [])
    diff_blocks = []
    for patch in patches:
        search = patch.get("search", "")
        replace = patch.get("replace", "")
        diff_blocks.append(f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")

    if diff_blocks:
        diff_content = "\n\n".join(diff_blocks)
        parts.append(f"\n<DIFF>\n{diff_content}\n\n</DIFF>")

    return "\n".join(parts)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    on_backoff=backoff_handler,
)
def query_anthropic(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model.

    Uses structured outputs with extended thinking to guarantee Claude follows
    the SEARCH/REPLACE format. The JSON output is converted back to standard
    DIFF format for compatibility with the rest of ShinkaEvolve.
    """
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]

    # Check if extended thinking is enabled
    has_thinking = "thinking" in kwargs

    if output_model is None and has_thinking:
        # Use structured outputs with extended thinking
        # This guarantees Claude follows the SEARCH/REPLACE format
        logger.debug("Using structured outputs with extended thinking")

        # Use beta endpoint with streaming for structured outputs
        # Streaming is required when max_tokens > 21,333 (we use 40,000)
        with client.beta.messages.stream(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            betas=["structured-outputs-2025-11-13"],
            output_format={
                "type": "json_schema",
                "schema": DIFF_OUTPUT_SCHEMA
            },
            **kwargs,
        ) as stream:
            response = stream.get_final_message()

        # Extract thinking and content from response
        thought = ""
        json_content = ""
        for block in response.content:
            if hasattr(block, "thinking"):
                thought = block.thinking
            elif hasattr(block, "text"):
                json_content = block.text

        # Parse JSON and convert to standard DIFF format
        try:
            json_response = json.loads(json_content)
            content = json_to_diff_format(json_response)
            logger.debug(f"Converted JSON to DIFF format: {len(content)} chars")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {json_content[:500]}...")
            content = json_content  # Fall back to raw content

    elif output_model is None:
        # Use streaming without structured outputs (no extended thinking)
        with client.messages.stream(
            model=model,
            system=system_msg,
            messages=new_msg_history,
            **kwargs,
        ) as stream:
            response = stream.get_final_message()
        # Separate thinking from non-thinking content
        if len(response.content) == 1:
            thought = ""
            content = response.content[0].text
        else:
            thought = response.content[0].thinking
            content = response.content[1].text
    else:
        raise NotImplementedError("Pydantic output_model not supported for Anthropic.")
    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    input_cost = CLAUDE_MODELS[model]["input_price"] * response.usage.input_tokens
    output_cost = CLAUDE_MODELS[model]["output_price"] * response.usage.output_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
