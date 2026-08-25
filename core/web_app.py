"""
Athena Web Interface.

A minimal FastAPI server exposing Athena's Agent over HTTP,
paired with a single-page browser chat UI. Local-only by design -
bound to 127.0.0.1, not exposed on the network.
"""

import base64
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import time

from core import __version__
from core.agent import (
    Agent, PROGRESS, RESPONSE_MODEL, MODES, RETIRED_MODES, Stopped,
    resolve_mode,
)
from models.ollama_model import USAGE

# Normally main.py has already done this, but this module is also a
# direct entry point (`uvicorn core.web_app:app`), where redirected
# output would otherwise be cp1252 and crash on any non-ASCII print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

print("Loading model...")
agent = Agent()

# Name the model from the constant rather than a literal, so the
# startup line can't quietly claim the wrong model after a swap.
print(f"Loading {RESPONSE_MODEL}...      ✓")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):

    yield

    agent.shutdown()


app = FastAPI(
    title="Athena",
    version=__version__,
    description="Local, evidence-grounded assistant API.",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)
class ChatRequest(BaseModel):
    message: str
    image: str | None = None


class ChatResponse(BaseModel):
    response: str
    # Controlled application failures still use the normal response shape so
    # the local UI can always render them. This flag prevents a failure from
    # looking like a successfully completed answer in the pipeline.
    error: bool = False
    seconds: float = 0.0
    tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    # Tokens the model actually had to work through. The rest of
    # "tokens" was served from Ollama's prompt cache and only counted,
    # which is why the total on screen looked so much larger than the
    # work behind it.
    computed_tokens: int = 0
    # Split out so the interface can show three numbers that add up:
    # what was read, what was reused from the cache, what was written.
    read_tokens: int = 0
    cached_tokens: int = 0
    # True when the user stopped this reply. Lets the interface show it
    # as stopped rather than as an answer that happens to read oddly.
    stopped: bool = False
    sources: list[dict] = Field(default_factory=list)


class ModeRequest(BaseModel):
    mode: str


# One reply at a time. FastAPI runs a plain `def` endpoint on a
# threadpool, so two overlapping requests really do execute side by
# side - and they would share one Agent, one conversation history and
# one PROGRESS. Before the stop button that could not happen, because
# the browser kept the send button disabled until the reply arrived
# and the reply only arrived when the turn was over.
#
# Stopping breaks that: the page gives up waiting immediately, while
# the turn keeps going until it reaches a stage where it can stop. A
# message sent in that window would interleave its history with the
# turn being abandoned, and its start_turn() would clear the stop flag
# the old turn had not yet noticed - leaving a turn nobody can stop.
#
# Switching conversations takes it for the same reason: swapping the
# history out from under a turn that is still reading it would answer
# the new question against the old conversation.
_TURN = threading.Lock()

# Long enough to cover a stop landing mid-stage, short enough that the
# page is not left hanging if something is genuinely stuck.
_TURN_WAIT_SECONDS = 20
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_BASE64_CHARS = 14 * 1024 * 1024


def _try_turn_lock() -> bool:
    """One atomic gate for every operation that mutates the Agent."""

    return _TURN.acquire(blocking=False)


def _maintain_memory_when_idle():
    """Compact old history after a response, never over an active turn."""

    if not _try_turn_lock():
        return
    try:
        maintain = getattr(agent, "maintain_memory", None)
        if callable(maintain):
            maintain()
    finally:
        _TURN.release()


def _decode_image(value: str) -> tuple[bytes, str]:
    """Validate a browser image before writing a unique temporary file."""

    payload = str(value or "").split(",", 1)[-1]
    if not payload or len(payload) > _MAX_IMAGE_BASE64_CHARS:
        raise ValueError("The image is empty or larger than 10 MB.")

    try:
        image = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("The pasted image is not valid base64 data.") from error

    if not image or len(image) > _MAX_IMAGE_BYTES:
        raise ValueError("The image is empty or larger than 10 MB.")

    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    elif image.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    elif image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        suffix = ".webp"
    else:
        raise ValueError("Please paste a PNG, JPEG, or WebP image.")

    return image, suffix


@app.get("/status")
def status():
    """The stage the agent is currently in.

    Polled by the UI while a reply is in flight so it can show what is
    actually happening instead of a generic spinner.
    """

    # "key" names which step of the pipeline this is, so the flow
    # diagram can light up without matching against the sentence.
    # "tools" is every capability used this turn, in order.
    return {
        "stage": PROGRESS.stage,
        "key": PROGRESS.key,
        "tools": list(PROGRESS.tools),
        "mode": agent.mode,
    }


@app.get("/conversations")
def conversations():
    """Saved conversations, newest first."""

    return {
        "current": agent.state.conversation_id,
        "conversations": agent.store.list(),
    }


@app.post("/conversations/new")
def new_conversation():
    """Start a fresh conversation, leaving the current one saved."""

    if not _try_turn_lock():
        return {"ok": False, "error": "Wait for the current reply to finish."}

    try:
        agent.new_conversation()
    finally:
        _TURN.release()

    return {"ok": True}


@app.post("/conversations/{conversation_id}")
def open_conversation(conversation_id: str):
    """Reopen a saved conversation."""

    if not _try_turn_lock():
        return {"ok": False, "error": "Wait for the current reply to finish."}

    # Held for the same reason a mode switch is: swapping the history
    # out from under a turn that is still reading it would answer the
    # new question against the old conversation.
    try:
        found = agent.load_conversation(conversation_id)
    finally:
        _TURN.release()

    if not found:
        return {"ok": False, "error": "That conversation is no longer there."}

    return {
        "ok": True,
        "id": agent.state.conversation_id,
        "title": agent.state.title,
        "messages": agent.state.messages,
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Remove a saved conversation."""

    if not _try_turn_lock():
        return {"ok": False, "error": "Wait for the current reply to finish."}

    try:
        removed = agent.store.delete(conversation_id)

        # Deleting the open one leaves the window showing a conversation
        # that no longer exists anywhere, so it is closed as well.
        if removed and agent.state.conversation_id == conversation_id:
            agent.new_conversation()
    finally:
        _TURN.release()

    return {"ok": removed}


@app.post("/stop")
def stop():
    """Ask the running reply to give up.

    Returns immediately. The reply stops at its next stage, not at the
    instant this is called: a model call already sent to Ollama runs to
    completion whatever happens here, so what is actually promised is
    that nothing further starts. In practice that is the difference
    between waiting for one more model call and waiting for a web
    search, three planning rounds and a self-check.
    """

    armed = PROGRESS.stop()

    return {"ok": armed, "stage": PROGRESS.stage}


@app.get("/modes")
def modes():
    """The available modes, and which one is active."""

    return {
        "current": agent.mode,
        "modes": [
            {
                "name": name,
                "label": config["label"],
                "blurb": config["blurb"],
                "model": config["response"],
            }
            for name, config in MODES.items()
        ],
    }


@app.post("/mode")
def set_mode(request: ModeRequest):
    """Switch modes, which swaps the loaded model.

    Slow on purpose: the outgoing model is unloaded and the incoming
    one loaded cold, because both will not fit in this card at once.
    The UI shows this as a stage rather than appearing to hang.
    """

    # A retired name is followed to whatever replaced it, so a browser
    # holding "study" from before it became Max gets Max rather than
    # being told its own saved choice does not exist.
    #
    # Anything else is refused. resolve_mode() falls back to the
    # default, which is right for reading a config but wrong here: a
    # typo would switch modes, report success, and leave the caller
    # believing it got the mode it asked for.
    # Refused while a reply is in flight. Switching unloads the model
    # that is mid-answer, so the running turn would fail on its next
    # call - and the interface disables the picker anyway, which means
    # anything reaching here is a stale tab or a direct request, not a
    # user who needs the courtesy of being obeyed.
    if not _try_turn_lock():
        return {"ok": False, "error": "Wait for the current reply to finish.",
                "mode": agent.mode}

    try:
        if PROGRESS.busy:
            return {
                "ok": False,
                "error": "Wait for the current reply to finish.",
                "mode": agent.mode,
            }

        if request.mode not in MODES and request.mode not in RETIRED_MODES:
            return {"ok": False, "error": f"Unknown mode: {request.mode}",
                    "mode": agent.mode}

        wanted = resolve_mode(request.mode)

        if wanted == agent.mode:
            return {"ok": True, "mode": agent.mode, "seconds": 0.0}

        started = time.perf_counter()
        PROGRESS.set(f"Switching to {MODES[wanted]['label']}")

        try:
            agent.set_mode(wanted)

        except Exception:
            print("[ERROR] mode switch failed:")
            traceback.print_exc()
            return {"ok": False, "error": "That mode couldn't be loaded.",
                    "mode": agent.mode}

        finally:
            PROGRESS.clear()

        return {
            "ok": True,
            "mode": agent.mode,
            "seconds": round(time.perf_counter() - started, 2),
        }
    finally:
        _TURN.release()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks):

    if not _TURN.acquire(timeout=_TURN_WAIT_SECONDS):
        return ChatResponse(
            response=(
                "Athena is still finishing the previous reply. "
                "Give it a moment and send that again."
            )
        )

    try:
        return _answer(request, background_tasks)

    finally:
        _TURN.release()


@app.post("/retry", response_model=ChatResponse)
def retry(background_tasks: BackgroundTasks):
    """Answer the last question again, forgetting the first attempt.

    The first attempt is deleted rather than kept alongside the new
    one. A wrong answer left in the history is quoted back to the model
    every following turn as something already settled, and it builds on
    it - so "try that again" has to mean the earlier try is gone.
    """

    if not _TURN.acquire(timeout=_TURN_WAIT_SECONDS):
        return ChatResponse(
            response=(
                "Athena is still finishing the previous reply. "
                "Give it a moment and try again."
            )
        )

    try:
        message = agent.take_back_last_turn()

        if not message:
            return ChatResponse(response="There's nothing to redo yet.")

        return _answer(ChatRequest(message=message), background_tasks)

    finally:
        _TURN.release()


def _answer(
    request: ChatRequest,
    background_tasks: BackgroundTasks | None = None,
) -> ChatResponse:

    image_path = None

    USAGE.reset()
    PROGRESS.start_turn()
    started = time.perf_counter()

    try:

        if request.image:

            try:
                image_bytes, suffix = _decode_image(request.image)
            except ValueError as error:
                return ChatResponse(
                    response=str(error),
                    error=True,
                    seconds=round(time.perf_counter() - started, 2),
                )

            with tempfile.NamedTemporaryFile(
                prefix="athena_upload_", suffix=suffix, delete=False
            ) as temporary:
                temporary.write(image_bytes)
                image_path = temporary.name

        answer = agent.respond(request.message, image_path=image_path)

        if background_tasks is not None:
            background_tasks.add_task(_maintain_memory_when_idle)

        return ChatResponse(
            response=answer,
            seconds=round(time.perf_counter() - started, 2),
            tokens=USAGE.total,
            prompt_tokens=USAGE.prompt_tokens,
            output_tokens=USAGE.output_tokens,
            model_calls=USAGE.calls,
            computed_tokens=USAGE.computed,
            read_tokens=USAGE.prompt_read,
            cached_tokens=USAGE.cached_tokens,
            sources=list(
                getattr(getattr(agent, "state", None), "last_sources", []) or []
            ),
        )

    except Stopped:

        # Not an error - the user asked for this. Reported as a normal
        # reply so the tokens already spent are still counted, and so
        # the interface has something to put in the transcript rather
        # than an empty turn.
        print("[STOP] reply stopped by the user")

        return ChatResponse(
            response="Stopped.",
            stopped=True,
            seconds=round(time.perf_counter() - started, 2),
            tokens=USAGE.total,
            prompt_tokens=USAGE.prompt_tokens,
            output_tokens=USAGE.output_tokens,
            model_calls=USAGE.calls,
            computed_tokens=USAGE.computed,
            read_tokens=USAGE.prompt_read,
            cached_tokens=USAGE.cached_tokens,
        )

    except Exception:

        # The traceback stays local - it can contain absolute paths and
        # extracted document text, neither of which belongs in an HTTP
        # response. The user gets a stable message instead of a 500
        # that the UI would fail to parse as JSON.
        print("[ERROR] /chat failed:\n" + traceback.format_exc())

        return ChatResponse(
            response=(
                "Something went wrong while handling that. "
                "The details were logged locally - please try again."
            ),
            error=True,
            seconds=round(time.perf_counter() - started, 2),
            tokens=USAGE.total,
            prompt_tokens=USAGE.prompt_tokens,
            output_tokens=USAGE.output_tokens,
            model_calls=USAGE.calls,
            computed_tokens=USAGE.computed,
            read_tokens=USAGE.prompt_read,
            cached_tokens=USAGE.cached_tokens,
        )

    finally:

        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError as error:
                print(f"[UPLOAD] could not remove temporary image: {error}")

        # Always clear the stage, or the UI would keep showing the
        # last thing the agent was doing after the reply arrives.
        PROGRESS.clear()


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the page, never from the browser's cache.

    The file is read fresh on every request so an edit shows up on a
    refresh without restarting Athena - but that is pointless if the
    browser answers the refresh from its own cache instead of asking,
    which is exactly what it was doing. These headers make it ask.
    """

    return HTMLResponse(
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )
