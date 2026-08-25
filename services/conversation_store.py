"""
Conversation storage.

Conversations lived only in memory until now, on one global agent, and
closing Athena threw them away. That is fine for a demo and wrong for
something people are meant to use: the thing you asked yesterday is
often exactly the thing you want to look at today.

One JSON file per conversation, in a folder next to the code. No
database, because there is nothing here a database would do better -
the files are readable, greppable, easy to back up, and easy to delete
one of. If this ever needs full-text search across thousands of
conversations, that is the point to reach for SQLite, not before.
"""

import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import CONVERSATION_DIR

STORE_DIR = CONVERSATION_DIR
INDEX_NAME = "_index.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,159}$")

# What a conversation is called before anything has been said in it.
UNTITLED = "New conversation"

# How much of the first message becomes the title.
TITLE_LIMIT = 60


def _slug(text: str) -> str:
    """A filename-safe fragment, for readability in the folder."""

    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)

    return text[:40].strip("-")


def title_from(message: str) -> str:
    """Name a conversation after the first thing asked in it."""

    text = " ".join((message or "").split())

    if not text:
        return UNTITLED

    if len(text) <= TITLE_LIMIT:
        return text

    return text[:TITLE_LIMIT].rstrip() + "..."


@dataclass
class Conversation:
    """One saved conversation.

    Carries the summary alongside the messages because the two only
    make sense together - a summary describes exactly the messages that
    were folded into it, and reloading one without the other would
    either repeat what the summary already says or lose it.
    """

    id: str
    title: str = UNTITLED
    created_at: float = 0.0
    updated_at: float = 0.0
    messages: list = field(default_factory=list)

    # Older messages, folded into prose once there are enough of them.
    summary: str = ""

    # How many of `messages` the summary already covers, so the same
    # exchange is never summarised twice.
    summarized_upto: int = 0

    # Operational context must travel with the transcript. Otherwise a
    # reopened conversation can display a selected document while the
    # agent has forgotten which document follow-up questions refer to.
    last_file_path: str = None
    last_generated_path: str = None
    pending_file_paths: list = field(default_factory=list)
    pending_file_request: str = None
    pending_lookup: str = None
    user_profile: dict = field(default_factory=dict)
    last_capabilities: list = field(default_factory=list)
    last_capability_steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "summary": self.summary,
            "summarized_upto": self.summarized_upto,
            "last_file_path": self.last_file_path,
            "last_generated_path": self.last_generated_path,
            "pending_file_paths": self.pending_file_paths,
            "pending_file_request": self.pending_file_request,
            "pending_lookup": self.pending_lookup,
            "user_profile": self.user_profile,
            "last_capabilities": self.last_capabilities,
            "last_capability_steps": self.last_capability_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            id=str(data.get("id") or ""),
            title=data.get("title") or UNTITLED,
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            messages=list(data.get("messages") or []),
            summary=data.get("summary") or "",
            summarized_upto=int(data.get("summarized_upto") or 0),
            last_file_path=data.get("last_file_path") or None,
            last_generated_path=data.get("last_generated_path") or None,
            pending_file_paths=list(data.get("pending_file_paths") or []),
            pending_file_request=data.get("pending_file_request") or None,
            pending_lookup=data.get("pending_lookup") or None,
            user_profile=dict(data.get("user_profile") or {}),
            last_capabilities=list(data.get("last_capabilities") or []),
            last_capability_steps=list(data.get("last_capability_steps") or []),
        )


class ConversationStore:

    def __init__(self, directory: Path = None):
        self.directory = Path(directory or STORE_DIR)

    def _ensure_dir(self):
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        conversation_id = str(conversation_id or "")

        if not _SAFE_ID.fullmatch(conversation_id):
            raise ValueError("Invalid conversation id.")

        directory = self.directory.resolve()
        target = (directory / f"{conversation_id}.json").resolve()

        if target.parent != directory:
            raise ValueError("Invalid conversation path.")

        return target

    def _index_path(self) -> Path:
        return self.directory / INDEX_NAME

    def _write_index(self, entries: list):
        """Atomically write only the metadata needed by the sidebar."""

        self._ensure_dir()
        target = self._index_path()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _rebuild_index(self) -> list:
        """One-time migration for conversation files saved before the index."""

        entries = []

        for path in self.directory.glob("*.json"):
            if path.name == INDEX_NAME:
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            entries.append({
                "id": data.get("id") or path.stem,
                "title": data.get("title") or UNTITLED,
                "updated_at": float(data.get("updated_at") or 0.0),
                "messages": len(data.get("messages") or []),
            })

        entries.sort(key=lambda entry: entry["updated_at"], reverse=True)

        try:
            self._write_index(entries)
        except OSError as error:
            print(f"[STORE] could not rebuild conversation index: {error}")

        return entries

    def _read_index(self) -> list:
        self._ensure_dir()
        path = self._index_path()

        if not path.is_file():
            return self._rebuild_index()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else self._rebuild_index()
        except (OSError, ValueError):
            return self._rebuild_index()

    def new_id(self, first_message: str = "") -> str:
        """A sortable id, with a readable tail where there is one.

        The timestamp leads so the folder sorts chronologically in any
        file browser; the slug is there so a human scanning it can tell
        which conversation is which without opening them.
        """

        # Microseconds plus a random suffix make collisions impossible
        # even when two tabs create a conversation in the same instant.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stamp += f"-{time.time_ns() % 1_000_000:06d}-{secrets.token_hex(2)}"
        slug = _slug(first_message)

        return f"{stamp}-{slug}" if slug else stamp

    def save(self, conversation: Conversation):
        """Write a conversation, replacing any previous version.

        Written to a temporary file and moved into place. A crash
        halfway through a direct write leaves a truncated file that
        will not parse, and losing a conversation to a bad shutdown is
        exactly what this module exists to prevent.
        """

        self._ensure_dir()

        conversation.updated_at = time.time()

        if not conversation.created_at:
            conversation.created_at = conversation.updated_at

        target = self._path(conversation.id)
        temporary = target.with_suffix(".json.tmp")

        try:
            temporary.write_text(
                json.dumps(conversation.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, target)

            entries = [
                item for item in self._read_index()
                if item.get("id") != conversation.id
            ]
            entries.append({
                "id": conversation.id,
                "title": conversation.title or UNTITLED,
                "updated_at": conversation.updated_at,
                "messages": len(conversation.messages),
            })
            entries.sort(key=lambda entry: entry["updated_at"], reverse=True)
            self._write_index(entries)

        except OSError as error:
            print(f"[STORE] could not save {conversation.id}: {error}")

            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self, conversation_id: str):
        """Read one conversation, or None if it is missing or broken."""

        try:
            path = self._path(conversation_id)
        except ValueError:
            return None

        if not path.is_file():
            return None

        try:
            return Conversation.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )

        except (OSError, ValueError) as error:
            # A corrupt file should cost that one conversation, not the
            # ability to open Athena.
            print(f"[STORE] could not read {conversation_id}: {error}")
            return None

    def list(self, limit: int = 50) -> list:
        """Summaries of the saved conversations, newest first.

        Deliberately does not return the messages. The list is drawn on
        every page load, and reading every conversation in full to show
        a column of titles would get slower with every one saved.
        """

        entries = self._read_index()
        entries.sort(key=lambda entry: entry.get("updated_at", 0.0), reverse=True)
        return entries[:limit]

    def delete(self, conversation_id: str) -> bool:
        """Remove one conversation. True if there was one to remove."""

        try:
            self._path(conversation_id).unlink()

            entries = [
                item for item in self._read_index()
                if item.get("id") != conversation_id
            ]
            self._write_index(entries)
            return True

        except (OSError, ValueError):
            return False

    def most_recent_id(self):
        """The conversation to reopen on startup, if there is one."""

        entries = self.list(limit=1)

        return entries[0]["id"] if entries else None
