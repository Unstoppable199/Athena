"""
Semantic search over document contents.

Filename search answers "is there a file called hostel fees". It cannot
answer "which file has my hostel payment in it", and that is the more
natural question - people remember what a document said far better than
what they named it. Half the files on a real machine are called
`Scan_20241203.pdf`.

So documents are read, split into chunks, embedded, and searched by
meaning. "How much did I pay for accommodation" finds a receipt that
says "hostel fee" and never uses either word.

Deliberately modest in scope:

- Only text-bearing documents, and only ones small enough to be worth
  reading. Indexing a whole disk is a different project.
- The index is built when asked for, not by a background watcher. A
  daemon quietly reading every file on someone's computer is a lot to
  install by accident.
- Similarity is computed in plain Python. The maths is a dot product,
  the corpus is a few thousand chunks at most, and adding numpy as a
  dependency to avoid a loop that takes 20ms is not a good trade.
"""

import json
import math
import os
from pathlib import Path

import ollama
from config import EMBED_MODEL, WORKSPACE_DIR


# Small, fast, and made for this. 274 MB, which matters on a card that
# is already full - it is asked for with a short keep_alive so it does
# not sit in VRAM next to the model actually answering questions.
# Unloaded five minutes after the last use. The chat model asks for
# keep_alive=-1 and holds the card; this one is a guest.
EMBED_KEEP_ALIVE = "5m"

INDEX_PATH = WORKSPACE_DIR / "semantic_index.json"

# Documents worth reading. Images are absent: they need OCR per page,
# which is slow enough to make indexing feel broken.
INDEXABLE = {".pdf", ".docx", ".txt", ".md", ".csv", ".pptx", ".xlsx"}

# Characters per chunk, and how much each overlaps the last.
#
# Chunks are what get compared, so they have to be small enough to be
# about one thing - a whole document averages out to meaning nothing in
# particular. The overlap stops a sentence that straddles a boundary
# from being lost to both halves.
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

# Files larger than this are skipped. A 40 MB PDF is a book, and
# embedding it costs minutes for a result nobody is waiting for.
MAX_FILE_BYTES = 8 * 1024 * 1024

# Chunks per file. A long document would otherwise dominate the index
# purely by having more chances to match.
MAX_CHUNKS_PER_FILE = 40

# How similar a chunk must be before it counts as a match.
#
# Measured against 12 real documents on this machine, with the task
# prefixes in use:
#
#   "reversing a linked list"        0.662  (the lecture notes)
#   "hostel accommodation payment"   0.573  (the remittance form)
#   "how much did I pay"             0.546  (a transaction record)
#   "quantum chromodynamics on mars" 0.508  (nothing - a marksheet)
#   "recipe for chocolate cake"      0.468  (nothing - a DSA tutorial)
#
# 0.52 sits in the gap. It is not a wide gap: embeddings put unrelated
# text closer together than intuition suggests, and a query that is
# vague rather than nonsense ("how much did I pay") lands nearer the
# floor than a specific one. Erring high is deliberate - a document
# that is not returned costs another search, while one returned wrongly
# becomes evidence for an answer about something it does not mention.
SIMILARITY_FLOOR = 0.52


def _cosine(a, b) -> float:
    """Similarity between two vectors, -1 to 1.

    Ollama returns normalised vectors, so the magnitudes are already
    1 and the dot product alone would do - but dividing by them costs
    nothing measurable and means this still works if that changes.
    """

    dot = sum(x * y for x, y in zip(a, b))

    if not dot:
        return 0.0

    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))

    if not left or not right:
        return 0.0

    return dot / (left * right)


def _chunk(text: str) -> list:
    """Split text into overlapping pieces."""

    text = " ".join((text or "").split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_FILE:

        end = start + CHUNK_CHARS
        piece = text[start:end]

        # Prefer to end on a sentence, so a chunk reads as something
        # rather than stopping mid-clause.
        if end < len(text):
            stop = max(piece.rfind(". "), piece.rfind("! "), piece.rfind("? "))

            if stop > CHUNK_CHARS * 0.5:
                piece = piece[:stop + 1]
                end = start + stop + 1

        piece = piece.strip()

        if piece:
            chunks.append(piece)

        start = end - CHUNK_OVERLAP

        if start <= 0:
            break

    return chunks


# nomic-embed-text is trained with these prefixes and expects them: a
# stored passage is a "search_document", the thing being looked for is
# a "search_query". They are not decoration. Without them everything
# scores in a narrow band - measured here, a real match landed at 0.553
# and deliberate nonsense at 0.488, which is not enough of a gap to set
# a threshold in. With them the two separate properly.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embed(texts: list, prefix: str = DOCUMENT_PREFIX) -> list:
    """Vectors for a list of strings, or [] if the model is missing."""

    if not texts:
        return []

    try:
        response = ollama.embed(
            model=EMBED_MODEL,
            input=[prefix + t for t in texts],
            keep_alive=EMBED_KEEP_ALIVE,
        )
        return response["embeddings"]

    except Exception as error:
        print(f"[SEMANTIC] could not embed: {error}")
        return []


def is_available() -> bool:
    """Whether the embedding model is installed.

    Checked rather than assumed: this is the one feature with an extra
    setup step, and failing with "model not found" halfway through a
    search is worse than saying so up front.
    """

    try:
        ollama.embed(model=EMBED_MODEL, input=["ping"],
                     keep_alive=EMBED_KEEP_ALIVE)
        return True

    except Exception:
        return False


class SemanticIndex:

    def __init__(self, filesystem, path: Path = None):
        # Text extraction is the filesystem service's job and it
        # already handles every format Athena reads, including OCR for
        # scanned pages. Duplicating it here would mean two answers to
        # "what does this PDF say".
        self.filesystem = filesystem
        self.path = Path(path or INDEX_PATH)
        self._entries = None

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    def _load(self) -> list:

        if self._entries is not None:
            return self._entries

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = data.get("entries") or []

        except (OSError, ValueError):
            self._entries = []

        return self._entries

    def _save(self):

        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self.path.with_suffix(".json.tmp")

        try:
            temporary.write_text(
                json.dumps({"model": EMBED_MODEL,
                            "entries": self._entries or []}),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)

        except OSError as error:
            print(f"[SEMANTIC] could not save the index: {error}")

    # ------------------------------------------------------------
    # Building
    # ------------------------------------------------------------

    def _text_of(self, path: Path) -> str:
        """Extract a document's text, or "" if it cannot be read."""

        try:
            result = self.filesystem.read(str(path))

        except Exception:
            return ""

        if not isinstance(result, dict) or not result.get("success", True):
            return ""

        # filesystem.read puts the extracted text in "data".
        text = result.get("data")

        return text if isinstance(text, str) else ""

    def index(self, roots=None, limit_files: int = 400) -> dict:
        """Read and embed documents under the given folders.

        Incremental: a file whose size and modification time match what
        was stored last time is left alone. Re-embedding an unchanged
        document is the slowest possible way to get the same numbers.
        """

        roots = roots or default_roots()

        entries = self._load()
        known = {e["path"]: e for e in entries}

        seen = set()
        added = 0
        skipped_unchanged = 0
        scanned = 0
        limited = False
        active_roots = []

        for root in roots:

            root = Path(root)

            if not root.is_dir():
                continue

            try:
                active_roots.append(root.resolve())
            except OSError:
                continue

            for folder, dirs, files in os.walk(root):

                dirs[:] = [d for d in dirs
                           if d not in self.filesystem.SKIP_DIRS]

                for name in files:

                    if scanned >= limit_files:
                        limited = True
                        break

                    path = Path(folder) / name

                    if path.suffix.lower() not in INDEXABLE:
                        continue

                    try:
                        stat = path.stat()
                    except OSError:
                        continue

                    if stat.st_size > MAX_FILE_BYTES:
                        continue

                    scanned += 1
                    key = str(path)
                    seen.add(key)

                    previous = known.get(key)

                    if (previous
                            and previous.get("mtime") == stat.st_mtime
                            and previous.get("size") == stat.st_size):
                        skipped_unchanged += 1
                        continue

                    text = self._text_of(path)
                    chunks = _chunk(text)

                    if not chunks:
                        continue

                    vectors = embed(chunks)

                    if not vectors:
                        # The model is missing or failed. Stopping is
                        # better than walking the whole disk producing
                        # nothing.
                        return {
                            "success": False,
                            "error": "The embedding model is not available. "
                                     f"Run: ollama pull {EMBED_MODEL}",
                        }

                    known[key] = {
                        "path": key,
                        "name": path.name,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "chunks": chunks,
                        "vectors": vectors,
                    }
                    added += 1

                if scanned >= limit_files:
                    limited = True
                    break

            if limited:
                break

        # Only a complete scan proves that an unseen file was removed.
        # Reaching the file limit means "not visited", not "deleted".
        # Also prune only within roots that were actually requested;
        # indexing Documents must not erase entries previously created
        # from Desktop.
        old_paths = {e["path"] for e in entries}

        def under_active_root(value: str) -> bool:
            try:
                candidate = Path(value).resolve()
                return any(
                    candidate == root or root in candidate.parents
                    for root in active_roots
                )
            except OSError:
                return False

        removed = [] if limited else [
            key for key in known
            if key not in seen and key in old_paths and under_active_root(key)
        ]

        for key in removed:
            known.pop(key, None)

        self._entries = list(known.values())
        self._save()

        return {
            "success": True,
            "indexed": added,
            "unchanged": skipped_unchanged,
            "removed": len(removed),
            "total": len(self._entries),
            "complete_scan": not limited,
        }

    # ------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------

    def search(self, query: str, limit: int = 5,
               threshold: float = SIMILARITY_FLOOR) -> dict:
        """Find documents whose contents match the query's meaning.

        The threshold matters more than the ranking. Without one the
        closest chunk is always returned, however unrelated - and a
        confident answer drawn from an irrelevant document is worse
        than "nothing matched".
        """

        if not query or not query.strip():
            return {"success": False, "error": "What should I look for?"}

        entries = self._load()

        if not entries:
            return {
                "success": False,
                "error": "Nothing has been indexed yet.",
                "needs_index": True,
            }

        vectors = embed([query], prefix=QUERY_PREFIX)

        if not vectors:
            return {
                "success": False,
                "error": "The embedding model is not available. "
                         f"Run: ollama pull {EMBED_MODEL}",
            }

        wanted = vectors[0]
        scored = []

        for entry in entries:

            best = 0.0
            best_chunk = ""

            for chunk, vector in zip(entry.get("chunks") or [],
                                     entry.get("vectors") or []):
                score = _cosine(wanted, vector)

                if score > best:
                    best = score
                    best_chunk = chunk

            if best >= threshold:
                scored.append({
                    "path": entry["path"],
                    "name": entry["name"],
                    "score": round(best, 3),
                    # The passage that matched, so the answer can quote
                    # the document rather than the filename.
                    "excerpt": best_chunk[:400],
                })

        scored.sort(key=lambda e: e["score"], reverse=True)

        if not scored:
            return {
                "success": False,
                "error": f"No document matched '{query}'.",
            }

        return {
            "success": True,
            "query": query,
            "matches": scored[:limit],
            "searched": len(entries),
        }

    def stats(self) -> dict:

        entries = self._load()

        return {
            "documents": len(entries),
            "chunks": sum(len(e.get("chunks") or []) for e in entries),
            "model": EMBED_MODEL,
            "path": str(self.path),
        }


def default_roots() -> list:
    """Where to look, when nobody says.

    The obvious document folders rather than the whole home directory -
    walking everything picks up caches, build output and application
    data, none of which anyone means by "my files".
    """

    home = Path.home()

    candidates = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Documents",
    ]

    return [c for c in candidates if c.is_dir()]
