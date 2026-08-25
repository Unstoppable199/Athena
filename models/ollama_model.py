"""
Ollama model wrapper.
"""

from ollama import Client
import time
from models.base import BaseModel
import re
from config import OLLAMA_TIMEOUT_SECONDS, RESPONSE_MAX_TOKENS


# A local generation should never hold the entire interface forever.
# Three minutes is deliberately generous for the 12B model on this
# machine, while still bounding rare Ollama stalls (one audit call took
# more than thirteen minutes to emit 68 tokens).
_CLIENT = Client(timeout=float(OLLAMA_TIMEOUT_SECONDS))


# Above this many prompt tokens per second, the prompt was not really
# read - Ollama reused the key/value cache from an identical prefix and
# only counted them.
#
# Measured on this machine: a cold 2,430-token prompt evaluated in
# 1.14s (~2,100/s); the same prompt again took 0.04s (~60,000/s). The
# gap is more than an order of magnitude, so anything in between is
# still clearly one side or the other.
CACHED_TOKENS_PER_SECOND = 10000


class UsageTracker:
    """Accumulates token counts across every model call in one request.

    A single reply can involve the router, the planner and the
    response model, so per-call numbers are not what the user wants to
    see - the interesting figure is what the whole turn cost.

    It also separates the tokens that were actually computed from the
    ones that were only counted. Ollama caches the prompt prefix, so a
    planner prompt resent on the second planning round costs almost
    nothing - but prompt_eval_count reports all 5,295 of them again.
    A reply showing "9,137 tokens" was mostly reporting work that never
    happened, which made the most prominent number on screen the least
    meaningful one.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.calls = 0

    def record(self, response):
        prompt = response.get("prompt_eval_count") or 0
        seconds = (response.get("prompt_eval_duration") or 0) / 1e9

        self.prompt_tokens += prompt
        self.output_tokens += response.get("eval_count") or 0
        self.calls += 1

        if not prompt:
            return

        # No measurable time for thousands of tokens means they were
        # served from the cache, not read.
        if seconds <= 0 or prompt / seconds > CACHED_TOKENS_PER_SECOND:
            self.cached_tokens += prompt

    @property
    def total(self):
        return self.prompt_tokens + self.output_tokens

    @property
    def prompt_read(self):
        """Prompt tokens the model actually read, cache excluded."""

        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def computed(self):
        """Everything the model really did: read plus written."""

        return self.prompt_read + self.output_tokens


USAGE = UsageTracker()


# How much the model is allowed to read in one call.
#
# Ollama's default is 4096, and it is not a soft limit: anything longer
# is cut from the FRONT and the model is told nothing about it. The
# planner's instructions alone are 4,886 tokens, so at the default they
# never arrived whole - roughly half of them, including the opening
# line, were dropped before the model saw a word of the actual
# question. Asked to repeat its own first instruction it quoted a line
# from the middle, because that was genuinely the start of what it
# received.
#
# 8192 was enough when the planner's instructions were 4,886 tokens.
# They are 5,295 now - every rule added to teach it something costs
# room here - and a real planning call was measured at 7,444 of 8,192
# with an EMPTY conversation and no file open. A few turns and an open
# document would have gone over, and the overflow comes off the front,
# taking the planner's opening instructions with it.
#
# 16384 restores the headroom. Measured cost on this card: qwen3:8b
# goes 5.6 -> 6.3 GB and still sits entirely on the GPU; gemma3:12b
# goes 8.7 -> 8.8 GB and spills 40% to the CPU instead of 39%. That is
# close to free, and far cheaper than a prompt that silently arrives
# with its first half missing.
#
# One value for every call on purpose. Ollama reallocates the context
# when this changes, which means a full model reload - measured at 3.4s
# for the 8b and far worse for the 12b. Varying it per role would pay
# that on nearly every call.
NUM_CTX = 16384

# Individual model replies are shortened only in the copy sent back to
# the model. The complete transcript remains on disk. Older exchanges
# are removed from the prompt only after Agent has represented them in
# its rolling summary; an arbitrary message-count slice would create a
# blind zone whenever summarisation was delayed or failed.
MAX_ASSISTANT_PROMPT_CHARS = 2400


def _history_for_prompt(state):
    """Copy every message not already represented by the summary."""

    summarized = getattr(state, "summarized_upto", 0) or 0
    messages = []

    for stored in state.messages[summarized:]:
        item = dict(stored)
        content = str(item.get("content") or "")

        if (
            item.get("role") == "assistant"
            and len(content) > MAX_ASSISTANT_PROMPT_CHARS
        ):
            item["content"] = (
                content[:MAX_ASSISTANT_PROMPT_CHARS].rstrip() + " [...]"
            )

        messages.append(item)

    return messages


def _warn_if_truncated(response, label):
    """Say so when a prompt came close to the context limit.

    Truncation is silent - the model simply answers with less than it
    was given, and the only visible symptom is that it starts ignoring
    instructions it appears to have been told. Worth a line in the log
    rather than another afternoon spent rewording a prompt that was
    never being read.
    """

    used = response.get("prompt_eval_count") or 0

    if used >= NUM_CTX * 0.9:
        print(
            f"[CONTEXT] {label}: {used} of {NUM_CTX} tokens - "
            "close to the limit, older content may have been dropped"
        )


class OllamaModel(BaseModel):

    def __init__(self, model_name, keep_alive=-1):
        self.model_name = model_name
        self.keep_alive = keep_alive

    def chat(
        self,
        state,
        message,
        system_prompt=None,
        images=None,
        num_predict=RESPONSE_MAX_TOKENS,
    ):

        # Every message not already represented by the rolling summary
        # is sent. Under normal operation the summary bounds this list;
        # if maintenance fails, keeping the unsummarised turns is safer
        # than silently forgetting them.
        messages = _history_for_prompt(state)

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        user_message = {"role": "user", "content": message}

        if images:
            user_message["images"] = images

        messages.append(user_message)

        options = {"num_ctx": NUM_CTX}

        # `is not None` rather than a plain truth test: -1 means "no
        # limit" and 0 means "generate nothing", and both are real
        # settings that a truth test would quietly drop.
        if num_predict is not None:
            options["num_predict"] = num_predict

        response = _CLIENT.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            keep_alive=self.keep_alive,
            think=False
        )

        USAGE.record(response)
        _warn_if_truncated(response, "chat")

        assistant = response["message"]["content"]
        assistant = re.sub(r"<think>.*?</think>", "", assistant, flags=re.DOTALL).strip()

        state.messages.append({"role": "user", "content": message})
        state.messages.append({"role": "assistant", "content": assistant})

        return assistant

    def complete(
        self,
        system_prompt,
        message,
        schema=None,
        num_predict=RESPONSE_MAX_TOKENS,
        think=None,
    ):

        options = {"num_ctx": NUM_CTX}

        # `is not None` rather than a plain truth test: -1 means "no
        # limit" and 0 means "generate nothing", and both are real
        # settings that a truth test would quietly drop.
        if num_predict is not None:
            options["num_predict"] = num_predict

        kwargs = {}

        if schema:
            kwargs["format"] = schema

        if think is not None:
            kwargs["think"] = think

        _call_start = time.perf_counter()

        response = _CLIENT.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            options=options,
            keep_alive=self.keep_alive,
            **kwargs
        )

        print(f"[OLLAMA CALL] {time.perf_counter() - _call_start:.2f}s")

        USAGE.record(response)
        _warn_if_truncated(response, "complete")

        thinking = response.get("message", {}).get("thinking") or ""

        print(
            f"[TOKENS] prompt_eval_count={response.get('prompt_eval_count')} "
            f"eval_count={response.get('eval_count')} "
            f"load_duration={(response.get('load_duration') or 0) / 1e9:.2f}s "
            f"prompt_eval_duration={(response.get('prompt_eval_duration') or 0) / 1e9:.2f}s "
            f"eval_duration={(response.get('eval_duration') or 0) / 1e9:.2f}s "
            f"thinking_chars={len(thinking)}"
        )
        
        return response["message"]["content"]
