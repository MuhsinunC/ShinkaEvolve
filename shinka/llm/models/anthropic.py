import backoff
import anthropic
from .pricing import CLAUDE_MODELS
from .result import QueryResult
import logging
import json

logger = logging.getLogger(__name__)


# Infinite retries - user must manually stop if they want to abort
# Exponential backoff caps at 5 minutes between retries
MAX_BACKOFF_SECONDS = 300  # 5 minutes max wait between retries

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
    """Log each retry attempt with clear error and wait time."""
    exc = details.get("exception")
    wait_time = details.get("wait", 0)
    tries = details.get("tries", 0)

    # Format wait time nicely
    if wait_time >= 60:
        wait_str = f"{wait_time / 60:.1f} minutes"
    else:
        wait_str = f"{wait_time:.0f} seconds"

    logger.warning(
        f"API CALL FAILED (attempt {tries}): {exc}\n"
        f"    Retrying in {wait_str}..."
    )


def _is_client_error(exc):
    """Check if exception is a client error (4xx) that shouldn't be retried.

    Client errors indicate malformed requests that won't succeed on retry.
    Server errors (5xx) and rate limits (429) should be retried.
    """
    # BadRequestError (400) - malformed request, don't retry
    if isinstance(exc, anthropic.BadRequestError):
        logger.error(f"CLIENT ERROR (400) - not retrying: {exc}")
        return True
    # AuthenticationError (401) - bad credentials, don't retry
    if isinstance(exc, anthropic.AuthenticationError):
        logger.error(f"AUTH ERROR (401) - not retrying: {exc}")
        return True
    # PermissionDeniedError (403) - forbidden, don't retry
    if isinstance(exc, anthropic.PermissionDeniedError):
        logger.error(f"PERMISSION ERROR (403) - not retrying: {exc}")
        return True
    # NotFoundError (404) - resource not found, don't retry
    if isinstance(exc, anthropic.NotFoundError):
        logger.error(f"NOT FOUND ERROR (404) - not retrying: {exc}")
        return True
    # All other errors (429 rate limit, 5xx server errors) should be retried
    return False


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_value=MAX_BACKOFF_SECONDS,  # Cap backoff at 5 minutes
    # No max_tries = infinite retries until success or manual stop
    on_backoff=backoff_handler,
    giveup=_is_client_error,  # Stop retrying on 4xx client errors
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

    # Convert system_msg to cached format for prompt caching
    # Cache has 5-min TTL; cache read = 10% cost, cache write = 125% cost
    system_blocks = [
        {
            "type": "text",
            "text": system_msg,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    if output_model is None and has_thinking:
        # Use structured outputs with extended thinking
        # This guarantees Claude follows the SEARCH/REPLACE format
        logger.debug("Using structured outputs with extended thinking")

        # Use beta endpoint with streaming for structured outputs
        # Streaming is required when max_tokens > 21,333 (we use 40,000)
        with client.beta.messages.stream(
            model=model,
            system=system_blocks,  # Cached system prompt
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
            system=system_blocks,  # Cached system prompt
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

    # Only add assistant message if content is non-empty
    # Empty content would poison message history and cause 400 errors on subsequent requests
    if content and content.strip():
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
    else:
        logger.warning("Skipping empty assistant message - would poison message history")

    # Extract cache metrics (available in Anthropic API response)
    cache_creation_input_tokens = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
    cache_read_input_tokens = getattr(response.usage, 'cache_read_input_tokens', 0) or 0

    # Log cache activity
    if cache_read_input_tokens > 0:
        logger.debug(f"Cache HIT: {cache_read_input_tokens} tokens read from cache")
    if cache_creation_input_tokens > 0:
        logger.debug(f"Cache WRITE: {cache_creation_input_tokens} tokens written to cache")

    # Calculate cost with cache pricing
    # Regular input tokens (not cached)
    regular_input = response.usage.input_tokens - cache_creation_input_tokens - cache_read_input_tokens

    # Cache write = 1.25× input price, cache read = 0.10× input price
    input_price = CLAUDE_MODELS[model]["input_price"]
    input_cost = (
        input_price * regular_input +
        input_price * 1.25 * cache_creation_input_tokens +
        input_price * 0.10 * cache_read_input_tokens
    )
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
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )
    return result
