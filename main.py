"""
Athena

Main entry point.
"""

import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from PIL import ImageGrab

from config import ATHENA_HOST, ATHENA_PORT
from core.agent import Agent


def _force_utf8_output():
    """Windows reports cp1252 for stdout as soon as it is redirected
    to a pipe or file, so any non-ASCII output raises
    UnicodeEncodeError and takes the process down. That covers more
    than the checkmarks below - Athena also prints extracted document
    text and model answers, so a rupee sign in a receipt or a curly
    quote in a reply is enough to do it. Interactive consoles are
    unaffected, which is why this only shows up when logging to a
    file.
    """

    for stream in (sys.stdout, sys.stderr):

        try:
            stream.reconfigure(encoding="utf-8", errors="replace")

        except (AttributeError, ValueError):
            # Already wrapped, or not a reconfigurable stream.
            pass


_force_utf8_output()


IMAGE_COMMAND = "/image"


def _stop_models():
    """Unload whatever Athena left in VRAM.

    Normally the server's own shutdown hook has already done this, and
    running `ollama stop` twice is harmless - stopping a model that is
    not loaded does nothing. It is repeated here because the cost of
    missing it is an 8 GB model sitting in the graphics card after
    Athena has closed, with nothing on screen to suggest it.
    """

    try:
        from core import web_app

        agent = getattr(web_app, "agent", None)

        if agent is not None:
            agent.shutdown()
            print("Model unloaded.          ✓")

    except Exception as error:
        print(f"Could not unload the model: {error}")
        print("Run 'ollama stop <model>' to free the memory.")


def read_message() -> str:

    lines = []

    while True:

        line = input("You: " if not lines else "")

        if not line:

            if lines:
                break

            continue

        lines.append(line)

    return "\n".join(lines)


def grab_clipboard_image():

    image = ImageGrab.grabclipboard()

    if image is None:
        return None

    path = Path(tempfile.gettempdir()) / "athena_clipboard.png"
    image.save(path, "PNG")

    return str(path)


def run_terminal():

    agent = Agent()

    try:

        while True:

            try:

                message = read_message().strip()

                if not message:
                    continue

                if message.lower() in {"exit", "quit"}:
                    break

                image_path = None

                if message.startswith(IMAGE_COMMAND):

                    image_path = grab_clipboard_image()

                    if image_path is None:
                        print("\nNo image found on the clipboard.")
                        continue

                    message = message[len(IMAGE_COMMAND):].strip()

                    if not message:
                        message = "Describe this image."

                response = agent.respond(message, image_path=image_path)

                print("\nAthena:", response)

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f"\nError: {e}")

    finally:

        print("\nShutting down...")
        agent.shutdown()


def run_web():

    print("=" * 49)
    print("Athena")
    print("=" * 49)
    print()

    import uvicorn

    import socket

    address = f"http://{ATHENA_HOST}:{ATHENA_PORT}"
    print(f"\nStarting Athena at {address} ...")

    # The server runs on the MAIN thread, and the browser is opened
    # from a helper thread instead. It used to be the other way round,
    # which quietly broke shutdown: uvicorn only installs its Ctrl+C
    # handling when it is on the main thread - its own source says
    # "Signals can only be listened to from the main thread" and skips
    # the setup otherwise. So Ctrl+C interrupted the join() here, the
    # process exited, and the daemon thread running the server was
    # killed outright. Its shutdown hook never ran, which meant
    # agent.shutdown() never ran, which meant `ollama stop` never ran -
    # and because every call asks for keep_alive=-1, the model stayed
    # loaded in VRAM after Athena had apparently closed.
    def open_browser_when_ready():

        while True:
            try:
                with socket.create_connection((ATHENA_HOST, ATHENA_PORT), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)

        print("Opening browser...       ✓")
        webbrowser.open(address)

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    print("Starting server...       ✓")

    try:
        uvicorn.run(
            "core.web_app:app",
            host=ATHENA_HOST,
            port=ATHENA_PORT
        )

    except KeyboardInterrupt:
        # Uvicorn normally handles this itself and unloads the model on
        # the way out. Caught here as well so an interrupt arriving
        # before the server is listening cannot skip the unload.
        pass

    finally:
        print("\nShutting down Athena...")
        _stop_models()


def main():
    run_web()


if __name__ == "__main__":
    main()
