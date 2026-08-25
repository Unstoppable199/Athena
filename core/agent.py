"""
Athena Agent.

Main orchestration pipeline.
"""
import time
import json
import re
import subprocess
import threading
from difflib import get_close_matches
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit
import copy
from config import (
    BALANCED_MODEL,
    CODE_REPAIR_ATTEMPTS,
    FAST_MODEL,
    FAST_VISION_MODEL,
    RESPONSE_MAX_TOKENS,
    WORKSPACE_DIR,
)
from core.conversation_state import ConversationState
from services.conversation_store import (
    Conversation, ConversationStore, UNTITLED, title_from,
)
from core.router import Router
from core.planner import Planner
from core.execution_manager import ExecutionManager
from core.prompt_builder import PromptBuilder
from core.router import (
    asks_current_datetime,
    challenges_last_lookup,
    is_active_file_request,
    is_recency_request,
    looks_arithmetic,
    missing_subject_capability,
    missing_subject_question,
    names_a_new_subject,
    resolve_pending_file_selection,
    selection_is_pending,
)
from core.grounded_prompt import (
    GROUNDED_SCHEMA,
    GROUNDED_WEB_SYSTEM_PROMPT,
    GROUNDED_FILESYSTEM_SYSTEM_PROMPT,
    GROUNDED_CODE_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_DATETIME_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
)
from core.chat_prompt import CHAT_SYSTEM_PROMPT
from core.image_prompt import IMAGE_SYSTEM_PROMPT

from models.ollama_model import OllamaModel

DEFAULT_BUDGET = 250
DETAIL_BUDGET = 900
MAX_PLAN_STEPS = 4

# All three roles run on one model. gemma3:12b is multimodal, so it
# handles images as well as text - which removes the reason the vision
# role used to need a model of its own (qwen3:8b had no vision
# capability and failed outright with a 400 on any image).
#
# One model instead of two also means only one set of weights has to
# sit in VRAM. The old pairing was 5.2 GB + 3.3 GB = 8.5 GB against an
# 8 GB card, so sending an image evicted the text model and the next
# text reply had to reload it.
#
# The three names are kept separate on purpose: splitting a role back
# out later (a smaller model for routing, say) is then a one-line
# change rather than a refactor.
PLANNER_MODEL = BALANCED_MODEL
RESPONSE_MODEL = BALANCED_MODEL
VISION_MODEL = BALANCED_MODEL


# ----------------------------------------------------------------
# Modes
# ----------------------------------------------------------------
#
# One model is loaded at a time and swapped when the mode changes.
#
# Two models cannot both sit in 8 GB of VRAM - gemma3:12b alone
# already spills about a third of itself onto the CPU - so running a
# second one alongside would make both slower. Swapping instead means
# whichever model is loaded gets the whole card. The cost is a cold
# load on the first message after a switch, which is paid when the
# user deliberately changes mode rather than on every request.
#
# "think" is a qwen3-only toggle. gemma3 rejects it outright with a
# 400 rather than ignoring it, so it is recorded per mode.
# Beyond the models, a mode may turn the accuracy work up or down.
# These are the knobs, with the values used when a mode says nothing:
#
#   search_results  how many web results a lookup pulls back
#   min_sources     how many independent sources a fact needs before
#                   it is stated plainly rather than hedged
#   force_compute   send arithmetic through real code instead of
#                   letting the model do it in its head
#   self_check      re-read the finished answer against the evidence
#                   before showing it
#
# Each one costs something - more results is more to read, forcing
# computation is a second round trip, self-checking is a third. They
# are set per mode so the cost lands where it was asked for.
MODE_DEFAULTS = {
    "search_results": 4,
    "min_sources": 1,
    "force_compute": False,
    "self_check": False,
}

MODES = {

    "fast": {
        "label": "Fast",
        "blurb": "Quicker replies from a smaller model.",
        # 5.2 GB fits the card entirely, where the 12b does not, which
        # is where the speed comes from. It is measurably worse at
        # staying inside its evidence, hence the blurb.
        "planner": FAST_MODEL,
        "response": FAST_MODEL,
        # qwen3 has no vision at all - it returns a 400 for any image -
        # so images fall to the small gemma, loaded only when one
        # actually arrives.
        "vision": FAST_VISION_MODEL,
        "code": FAST_MODEL,
        "style": None,
        # Nothing turned up. This mode exists to be quick, and every
        # knob above costs time.
    },

    "balanced": {
        # Names describe what you get, not a claim about the others.
        # "Accurate" was worse: it implied the remaining modes were
        # inaccurate, which is a promise this cannot keep - every mode
        # runs the same grounding checks.
        "label": "Balanced",
        "blurb": "The default. Best answers, a little slower.",
        "planner": BALANCED_MODEL,
        "response": BALANCED_MODEL,
        "vision": BALANCED_MODEL,
        "code": BALANCED_MODEL,
        "style": None,
        # min_sources is the one upgrade that is genuinely free: the
        # corroboration count is already worked out for every web
        # answer, so requiring two sources costs no extra call, no
        # extra search and no extra tokens. It only changes how an
        # already-known number is acted on. The rest stay off here
        # because they all cost a round trip.
        "min_sources": 2,
    },

    "max": {
        "label": "Max",
        "blurb": "Most detailed answers and broader research. Slowest.",
        "planner": BALANCED_MODEL,
        "response": BALANCED_MODEL,
        "vision": BALANCED_MODEL,
        "code": BALANCED_MODEL,
        # Search and computation are turned up. The former model-based
        # self-check is deliberately off: the release evaluation spent
        # 99 seconds on 36 checks, caught no real error, and produced
        # eight false or useless warnings. Deterministic grounding still
        # runs in every mode and is the check that actually caught bad
        # numbers and identifiers.
        "search_results": 8,
        "min_sources": 2,
        "force_compute": True,
        "self_check": False,
        "style": """

----------------------------------------
SHOW YOUR WORKING
----------------------------------------

The reader wants to see how the answer was reached, not just what it
is.

- Give the answer, then show how it was reached, in order.
- Name the rule or formula being applied at each step.
- Define any term the first time it appears.
- Where a figure was calculated, show the numbers that produced it.
- Do not skip a step because it seems obvious.

Never invent a step that the evidence does not support. Showing
working means showing the working that actually happened.
""",
    },
}

DEFAULT_MODE = "balanced"

# Modes that used to exist, pointed at whatever replaced them. Without
# this a saved mode from an older version silently becomes the default,
# so someone who had picked "Explain" would quietly get plain answers
# and no indication of why.
RETIRED_MODES = {
    "accurate": "balanced",   # renamed
    "study": "max",           # its step-by-step style lives on in Max
}


def resolve_mode(name: str) -> str:
    """The current name for a mode, following any rename."""

    if name in MODES:
        return name

    return RETIRED_MODES.get(name, DEFAULT_MODE)


def _mode_config(name: str) -> dict:
    """Settings for a mode name, with the unset knobs filled in.

    Renamed modes are followed rather than dropped, and anything a
    mode does not mention takes the shared default - so a mode entry
    only has to list what it actually changes.
    """

    config = dict(MODE_DEFAULTS)
    config.update(MODES[resolve_mode(name)])

    return config


def _supports_thinking(model_name: str) -> bool:
    """Whether Ollama's think toggle may be sent to this model."""

    return str(model_name).startswith("qwen3")


class Stopped(Exception):
    """Raised when the user stops a reply that is already running."""


class Progress:
    """Publishes the pipeline stage the agent is currently in.

    Responses aren't streamed, so without this the UI can only show a
    generic spinner. Reporting the real stage means the label is
    honest - "Searching the web" appears because a web search is
    actually running, not because a timer decided it was time to say
    something.

    It also carries the stop flag, because the stages are exactly the
    points where stopping is safe.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.stage = "idle"
        self.key = "idle"

        # Capabilities used so far this turn, in order. Reported to the
        # interface so it can show what the request actually needed -
        # a weather lookup and a file read leave visibly different
        # trails, and until now that was only visible in the log.
        self.tools = []

        self._stopping = False

    def start_turn(self):
        """Begin a new request, forgetting the previous turn's tools."""

        with self._lock:
            self.tools = []
            self._stopping = False

    def stop(self) -> bool:
        """Ask the running turn to give up at its next stage.

        Does nothing when nothing is running, and says so. Arming the
        flag regardless would leave it set with no turn to consume it,
        and the next thing to announce a stage - a mode switch, say -
        would die on a stop that was never meant for it.
        """

        with self._lock:
            if self.stage == "idle":
                return False

            self._stopping = True
            return True

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    @property
    def busy(self) -> bool:
        """Whether a reply is currently being worked on."""

        with self._lock:
            return self.stage != "idle"

    def used(self, tool: str):
        """Record a capability as it starts running."""

        with self._lock:
            if tool and (not self.tools or self.tools[-1] != tool):
                self.tools.append(tool)

    def set(self, stage: str, key: str = "run"):
        """Publish the stage, with a key naming which part of the
        pipeline it belongs to.

        The sentence is for the user and changes with every new tool;
        the key is for the interface, which lights up a step of the
        flow diagram. Matching the diagram against the sentence would
        break silently the first time a capability was added with
        wording nobody thought to match, so the two are kept separate.

        "run" is the default because most stages are a tool running.

        This also raises Stopped if the user has asked to stop. Doing
        it here rather than with a check before each stage is
        deliberate: every stage already announces itself through this
        one method, so cancellation reaches all of them and cannot be
        forgotten when a new capability is added. Between stages is
        also the only safe place to stop - a model call already sent
        cannot be recalled, so the most that can be promised is that
        nothing FURTHER runs.
        """

        with self._lock:
            if self._stopping:
                raise Stopped()

            self.stage = stage
            self.key = key
        print(f"[STAGE] {stage}")

    def finish_turn(self):
        """Atomically reject a late stop or mark the turn complete."""

        with self._lock:
            if self._stopping:
                raise Stopped()
            self.stage = "idle"
            self.key = "idle"

    def clear(self):
        with self._lock:
            self.stage = "idle"
            self.key = "idle"
            # Cleared too, so a stop that arrived as a turn was finishing
            # cannot sit around and kill the next one before it starts.
            self._stopping = False


PROGRESS = Progress()


# Human-readable stage labels per capability.
_STEP_STAGES = {
    "web.search": "Searching the web",
    "weather.current": "Checking the weather",
    "finance.quote": "Checking the share price",
    "finance.exchange": "Checking the exchange rate",
    "filesystem.search": "Looking for the file",
    "filesystem.semantic_search": "Searching your documents",
    "filesystem.read": "Reading the file",
    "filesystem.list": "Listing the folder",
    "filesystem.exists": "Checking the path",
    "filesystem.info": "Checking the file",
    "python.run": "Running the script",
    "code.generate": "Writing the code",
    "code.run": "Running the code",
    "system.datetime": "Checking the time",
}


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _fix_stale_query_year(query: str) -> str:
    """
    The planner is instructed to include today's real date in
    time-sensitive queries, but repeatedly substitutes a year
    recalled from its own training data instead (e.g. "2022") -
    locking search results to a stale year regardless of how the
    prompt words the instruction. Enforced here structurally: any
    query for a recency-marked request that doesn't contain the
    current year gets its year replaced with today's actual date.
    """
    today = datetime.now()
    current_year = str(today.year)

    if current_year in query:
        return query

    query = _YEAR_RE.sub("", query).strip()
    query = re.sub(r"\s+", " ", query)

    return f"{query} {today.strftime('%d %B %Y')}".strip()


def _prepare_search_step(step: dict, message: str) -> dict:
    """Keep recency-marked searches anchored to today's real date.

    The planner once turned "last World Cup" into a 2022-only query
    even after the 2026 tournament had finished. The correction helper
    already existed but was no longer connected to execution, so this
    applies it to every query in the actual step before the search runs.
    Historical questions without recency wording are left untouched.
    """

    if step.get("type") != "web.search":
        return step

    prepared = copy.deepcopy(step)
    args = prepared.setdefault("args", {})

    # Category is a retrieval hint, not something the planner is
    # allowed to invent arbitrarily. A software-version lookup was once
    # labelled as finance, which changed source ranking for no reason.
    # Correct only clearly incompatible labels; ambiguous searches stay
    # general rather than being forced into a specialist category.
    category_context = [str(message or ""), str(args.get("query") or "")]
    if isinstance(args.get("queries"), list):
        category_context.extend(str(query) for query in args["queries"])
    text = " ".join(category_context).casefold()
    hints = {
        "weather": r"\b(?:weather|forecast|temperature|rain|snow|humidity)\b",
        "finance": r"\b(?:stock|share|ticker|market|exchange rate|currency|forex)\b",
        "sports": r"\b(?:sport|match|game|tournament|cup|league|fifa|uefa|nba|nfl|cricket|football)\b",
    }
    hinted = next(
        (name for name, pattern in hints.items() if re.search(pattern, text)),
        None,
    )
    category = str(args.get("category") or "general").casefold()
    if hinted:
        args["category"] = hinted
    elif category in hints:
        args["category"] = "general"

    if _asks_latest_stable_version(message):
        subject = _version_query_subject(message)
        if subject:
            variants = args.get("queries", [])
            if isinstance(variants, str):
                variants = [variants]
            elif not isinstance(variants, list):
                variants = []
            focused = f"{subject} official latest stable release"
            if focused.casefold() not in {
                str(query).casefold() for query in variants
            }:
                variants.append(focused)
            args["queries"] = variants

    # Result questions need result-bearing pages, not tournament overviews.
    # The generic query returned four Wikipedia pages for the World Cup but
    # only one snippet actually stated the final score; one model then used a
    # historical meeting from inside another page as the answer. Add focused
    # result variants for any sports winner/final request, and prefer the
    # governing body's domain when the competition names it.
    if (
        args.get("category") == "sports"
        and re.search(r"\b(?:won|winner|champion|final|result|score)\b", text)
    ):
        variants = args.get("queries", [])
        if isinstance(variants, str):
            variants = [variants]
        elif not isinstance(variants, list):
            variants = []

        primary = str(args.get("query") or message or "").strip()
        focused = f"{primary} official final result champion".strip()
        candidates = [focused]
        if re.search(r"\bfifa\b", text):
            candidates.append(f"site:fifa.com {primary} final result")

        existing = {str(query).casefold() for query in variants}
        for candidate in candidates:
            if candidate.casefold() not in existing:
                variants.append(candidate)
                existing.add(candidate.casefold())
        args["queries"] = variants[:2]

    if not is_recency_request(message):
        return prepared

    if isinstance(args.get("query"), str):
        args["query"] = _fix_stale_query_year(args["query"])

    if isinstance(args.get("queries"), list):
        args["queries"] = [
            _fix_stale_query_year(query) if isinstance(query, str) else query
            for query in args["queries"]
        ]

    return prepared


def _wants_detail(message: str) -> bool:

    keywords = [
        "detail", "explain", "elaborate", "in depth", "indepth",
        "more info", "everything", "full", "comprehensive",
        "walk me through", "breakdown", "long answer"
    ]

    lowered = message.lower()

    return any(k in lowered for k in keywords)

_BROAD_QUESTION = re.compile(r"^\s*(who is|what is|tell me about|describe)\b", re.IGNORECASE)


# ----------------------------------------------------------------
# Folding old messages into a summary
# ----------------------------------------------------------------
#
# Every turn resends recent history, and history only grows. Trimming
# to the last N messages bounds the prompt but forgets - twenty
# messages into a conversation the model no longer knows your name.
#
# So the old messages are folded into prose instead of dropped. The
# summary is written once, kept, and extended when enough new messages
# have built up again, which means the cost is one model call every
# SUMMARIZE_EVERY messages rather than a prompt that grows forever.

# Summarising is deliberately rare. It is bookkeeping, not part of an
# answer, and should not add a model call to normal conversation.
# Thirty exchanges is enough material to justify one compact update.
SUMMARIZE_EVERY = 24

# How many recent messages are always sent verbatim. Summaries lose
# detail, and the last few exchanges are where detail matters most -
# "the first one" has to be able to find what it refers to.
KEEP_VERBATIM = 8

SUMMARY_SYSTEM_PROMPT = """You are keeping extremely compact context notes on a conversation so it can be remembered after old messages stop being sent.

Write a brief factual record of what was said. Keep:

- who the user is, and anything they told you about themselves
- the topics and unresolved tasks the user may continue later
- decisions, preferences and corrections
- user-provided names, numbers, files and places that still matter

Existing notes are context, not an authority for outside facts.
Preserve durable user identity, preferences and the newest thing the
user explicitly said about themselves even when new messages are about
something else. Never replace a corrected user detail with an older one.

Do not preserve Athena's old general-knowledge answers, calculations,
current web facts, guesses or tool output as established truth. Those
can become stale or have been wrong and must be answered or verified
again when asked. Record a correction from the user as something the
user claimed, not automatically as a verified fact.

Drop pleasantries, repetition and your own phrasing. Do not inventory
generated scratch scripts, tool calls, search results or every file in
a directory; retain a path only when the user is still working with it.

Use at most 80 words. Do not infer intent or classify what capability
should run. Preserve unresolved questions and corrections exactly.
File selection, active paths and capability state are stored separately,
so do not try to reconstruct them.

Write plain sentences in the third person: "The user said their name is
Alex. They asked about..." Do not address the user. Do not add
anything that was not said."""


def _messages_as_text(messages: list) -> str:
    """Flatten messages into something a model can read as a record."""

    lines = []

    for message in messages:
        role = "User" if message.get("role") == "user" else "Athena"
        content = " ".join(str(message.get("content") or "").split())

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ----------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------
#
# A second read of the finished answer, by the same model that wrote
# it, against the same evidence. The checks above are mechanical: they
# compare words and numbers. This one is asked to notice what they
# cannot - a claim built entirely from words that appear in the
# evidence, saying something the evidence never said.
#
# It only ever flags. Letting it rewrite would put a fresh, unchecked
# answer on screen at exactly the point the pipeline had finished
# checking the old one.

SELF_CHECK_SYSTEM_PROMPT = """You are checking whether an answer is supported by the evidence it was based on.

You are not rewriting the answer, improving it, or answering the
question yourself. You only report what the evidence does not back up.

Only the text under ANSWER is being checked. QUESTION is given for
context only - it is what the USER said, not part of what Athena
answered, and it is never itself a claim to flag. Asked "no im pretty
sure its 6" after Athena correctly answered "2 + 3 equals 5", the
right result is an empty list: the disagreement is the user's, and
there is nothing unsupported in an answer that only states a computed
number. Flagging the question text back as an "unsupported claim"
quotes the user's own message at them as if Athena had said it.

A claim is UNSUPPORTED when the ANSWER states something the evidence
does not say. That includes a number that does not appear, a name the
evidence never mentions, a cause the evidence does not state, and a
certainty the evidence does not have ("always", "the only", "never").

A claim is SUPPORTED when the evidence says it, even in different
words.

Do not flag:
- ordinary wording, phrasing or summarising
- something the evidence plainly implies
- the answer saying it could not find something
- anything from QUESTION rather than ANSWER

List only the specific claims FROM THE ANSWER that are not supported.
If everything in the answer is supported, return an empty list."""

SELF_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported": {
            "type": "array",
            "items": {"type": "string"},
            # Capped rather than open: without a limit the model pads
            # the list to look thorough, and a check that flags
            # everything is as useless as one that flags nothing.
            "maxItems": 3,
        }
    },
    "required": ["unsupported"],
}


def _grounded_schema_for(message: str) -> dict:
    """Evidence array size enforced by the schema (constrained
    decoding), not requested in prose - 'cite enough for a broad
    question' and 'don't dump the whole page' are both guarantees
    this way, not requests the model can drift from under load."""
    # Evidence quality is checked in code after generation. Requiring a
    # fixed number here made the schema contradict the prompt's valid
    # "nothing found" response and forced broad questions to cite four
    # items even when only one or two useful facts existed.
    return copy.deepcopy(GROUNDED_SCHEMA)


# Two limits, because the two kinds of reply are not alike.
#
# A capability answer can be thousands of characters of scraped page
# text, none of it worth remembering - so it is cut hard.
#
# A chat reply is Athena's own prose, and the conversation depends on
# it: what was explained, what was suggested, what was agreed. Cutting
# that to 700 characters would drop the end of most real explanations,
# and the follow-up question is usually about the end. It still needs a
# ceiling - it used to have none at all, which is how a long
# conversation grew until it pushed the system prompt out of the
# context window - but a far more generous one.
_HISTORY_ANSWER_LIMIT = 700
_HISTORY_CHAT_LIMIT = 2400


def _for_history(answer: str, limit: int = _HISTORY_ANSWER_LIMIT) -> str:
    """Keep a compact record of a very long reply.

    What the user sees is never shortened - only what later turns are
    reminded of. The evidence fallback can emit thousands of characters
    of scraped page text, and that lands in the history every following
    prompt is built from. Asked about the weather and then "is there a
    file named hostel fees", the planner had six thousand characters of
    forecast prose ahead of the actual question and answered it with
    filesystem.info on a bare name instead of a search. History is meant
    to recall what was discussed, not to re-deliver the source material.
    """

    if len(answer) <= limit:
        return answer

    # Cut at a sentence end where there is one nearby, so the record
    # stops on a finished thought rather than mid-word.
    cut = answer[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))

    if stop > limit * 0.6:
        cut = cut[:stop + 1]

    return cut.rstrip() + " [...]"


PREAMBLE_PATTERNS = [
    r"^(okay|ok|sure|alright|got it)[,!.\s-]*",
    r"^here'?s (a |an |the )?(consolidated |quick |brief )?(summary|breakdown|overview)[^:]*:?\s*",
    r"^based on (the |all the )?(provided |available )?(text|information|snippets|results)[^:]*:?\s*",
]


def _strip_preamble(text: str) -> str:

    for pattern in PREAMBLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

_DATE_TOKEN_RE = re.compile(
    r"\b\d{1,4}(?:[-/:](?:[A-Za-z]{3,9}|\d{1,4})){1,3}\b"
)
_IDENTIFIER_RE = re.compile(
    r"\b(?=[A-Za-z]*\d)(?=[0-9]*[A-Za-z])[A-Za-z0-9]{3,}\b"
)
_DOTTED_ABBREVIATION_RE = re.compile(
    r"\b[A-Za-z](?:\.[A-Za-z]{1,5})+\b"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])"
)

_WORD_EQUIVALENTS = {
    "jan": "january", "feb": "february", "mar": "march",
    "apr": "april", "jun": "june", "jul": "july",
    "aug": "august", "sep": "september", "sept": "september",
    "oct": "october", "nov": "november", "dec": "december",
    "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "thu": "thursday", "thur": "thursday",
    "thurs": "thursday", "fri": "friday", "sat": "saturday",
    "sun": "sunday",
    # A model may expand the ISO label printed on a receipt without
    # changing the currency. Only the adjective is needed here because
    # lowercase "rupees" is not treated as a specific claim. This keeps
    # "INR 21,000" and "21,000 Indian rupees" equivalent while the exact
    # number still has to match digit for digit.
    "inr": "indian", "indian": "indian",
}


def _key_terms(text: str):
    """Proper nouns and numbers/scores - the words that actually carry
    the specific meaning of a sentence (who, what score, which date).
    Used to check that 'answer' doesn't state anything beyond what
    verified evidence actually contains.

    Sentence-initial words are lowercased first, so ordinary words
    that only happen to start a sentence ("Currently", "There") are
    not mistaken for proper nouns."""
    text = re.sub(r"(^|[.!?]\s+)([A-Z])", lambda m: m.group(1) + m.group(2).lower(), text)
    pattern = re.compile(
        _DATE_TOKEN_RE.pattern
        + "|" + _DOTTED_ABBREVIATION_RE.pattern
        + "|" + _IDENTIFIER_RE.pattern
        + r"|\b[A-Z][a-zA-Z]*\b"
        + "|" + _NUMBER_RE.pattern
    )
    return set(pattern.findall(text))


def _normal_number(value: str):
    """Comparable numeric value, without allowing partial matches."""

    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


_OUTCOME_CLAIM_RE = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})\s+"
    r"(?:(?:has|had)\s+)?(?:won|wins|win|winning|beat|beats|defeated|defeats)\b"
    r"(?P<scope>[^.!?\r\n]{0,140})"
)

_OUTCOME_SCOPE_WORDS = {
    "cup", "final", "title", "champion", "championship", "tournament",
    "match", "game", "election", "race", "award", "league", "fifa",
    "uefa", "world",
}

_OUTCOME_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}"
    r")\b[^.!?\r\n]{0,110}\b(?:"
    r"won|wins|winner|winners|champion|champions|"
    r"crowned|claim(?:s|ed|ing)?|victory\s+over"
    r")\b",
    re.IGNORECASE,
)

_OUTCOME_QUESTION_RE = re.compile(
    r"\b(?:who\s+won|winner|champion|who\s+beat|final\s+result)\b",
    re.IGNORECASE,
)


def _contains_outcome_relationship(text: str) -> bool:
    """Whether text directly states an outcome rather than background.

    This deliberately accepts the common language used by official sports
    sources: "crowned ... winners", "claiming ... crown", and "victory over"
    are just as explicit as the shorter "won" form.
    """

    return bool(_OUTCOME_CLAIM_RE.search(text or "") or _OUTCOME_EVIDENCE_RE.search(text or ""))


def _outcome_answer_missing(request: str, answer: str) -> bool:
    """True when an outcome question received only background information."""

    return bool(
        _OUTCOME_QUESTION_RE.search(request or "")
        and not _contains_outcome_relationship(answer)
    )


def _outcome_relationships_supported(answer: str, evidence: list) -> bool:
    """Reject a winner/loser inversion even when all names are present.

    The ordinary evidence check verifies names and numbers, but that cannot
    distinguish "Spain beat Argentina" from "Argentina beat Spain". For an
    explicit outcome claim, require one evidence item to attach the outcome
    verb to the same subject and, when supplied, the same competition scope.
    """

    claims = list(_OUTCOME_CLAIM_RE.finditer(answer or ""))
    if not claims:
        return True

    for claim in claims:
        subject = claim.group("subject").strip()
        scope_text = claim.group("scope").casefold()
        scope = {
            token for token in re.findall(r"[a-z0-9]+", scope_text)
            if token in _OUTCOME_SCOPE_WORDS or re.fullmatch(r"(?:19|20)\d{2}", token)
        }
        subject_re = re.escape(subject)
        supported = False

        for item in evidence:
            sentence = str(item or "")
            relation = bool(
                re.search(
                    rf"\b{subject_re}\b[^:;.!?\r\n]{{0,24}}\b"
                    r"(?:won|wins|win|winning|beat|beats|defeated|defeats|"
                    r"winner|winners|champion|champions)\b",
                    sentence,
                    re.IGNORECASE,
                )
                or re.search(
                    rf"\b{subject_re}\b[^:;.!?\r\n]{{0,110}}\b"
                    r"(?:crowned\b[^:;.!?\r\n]{0,70}\b(?:winner|winners|champion|champions)|"
                    r"claim(?:s|ed|ing)?\b[^:;.!?\r\n]{0,70}\b(?:crown|title|trophy)|"
                    r"victory\s+over\b)",
                    sentence,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\b(?:winner|champion)\b[^.!?\r\n]{0,35}"
                    rf"\b{subject_re}\b",
                    sentence,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\bwon\s+by\b[^.!?\r\n]{0,25}"
                    rf"\b{subject_re}\b",
                    sentence,
                    re.IGNORECASE,
                )
            )
            if not relation:
                continue

            # A bare historical line such as "Argentina winning 2-1" does
            # not support "Argentina won the 2026 FIFA World Cup final".
            # Require the competition words or year from the claim to occur
            # in that same evidence item. Two anchors avoid accepting a lone
            # generic word such as "final" from an unrelated result.
            if scope:
                evidence_tokens = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
                if len(scope & evidence_tokens) < min(2, len(scope)):
                    continue

            supported = True
            break

        if not supported:
            print(
                "[GROUNDING] outcome relationship is not supported for "
                f"{subject!r}"
            )
            return False

    return True


def _outcome_web_evidence(
    sentence_map: dict,
    sentence_origin: dict,
    search_urls: set,
    results: list,
    queries: list,
) -> list:
    """Best retrieved sentences that directly answer an outcome question.

    A grounded draft may cite tournament background instead of the official
    result.  This recovery remains deterministic: it can only return tagged
    retrieval text, ranks official/trusted pages first, and requires an
    explicit outcome relationship in the sentence itself.
    """

    page_meta = {}
    for result in results or []:
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            continue
        for page in data:
            if isinstance(page, dict) and page.get("url"):
                page_meta[str(page["url"])] = page

    ranked = []
    for position, (sid, text) in enumerate(sentence_map.items()):
        origin = str(sentence_origin.get(sid) or "")
        if origin not in search_urls or not _contains_outcome_relationship(text):
            continue
        relevance = _evidence_relevance_score([text], queries)
        if not relevance:
            continue
        page = page_meta.get(origin, {})
        authority = 2 if page.get("official") else 1 if page.get("trusted") else 0
        ranked.append((-authority, -relevance, position, text))

    ranked.sort()
    selected = []
    seen = set()
    for _, _, _, text in ranked:
        normalized = _normalize_ws(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(text)
        if len(selected) == 4:
            break
    return selected

def _answer_within_evidence(answer: str, evidence: list) -> bool:
    """True when every specific claim in the composed answer is also
    present in the verified evidence.

    Only proper nouns and numbers are checked - those are what carry
    the falsifiable content (which city, what temperature, whose
    name). Ordinary connecting words are what makes a composed answer
    readable in the first place, so requiring those to match too
    would reject every well-written answer.
    """

    evidence_text = _normalize_ws(" ".join(evidence))

    if not evidence_text:
        return False

    terms = _key_terms(answer)

    if not terms:
        # Nothing specific is being asserted, so there is nothing that
        # could go beyond the evidence.
        return True

    evidence_words = set(re.findall(r"[a-z]+", evidence_text))
    evidence_word_shapes = {
        re.sub(r"[^a-z]", "", word.casefold())
        for word in re.findall(r"[A-Za-z.]+", " ".join(evidence))
        if re.sub(r"[^a-z]", "", word.casefold())
    }
    evidence_ids = {match.casefold() for match in _IDENTIFIER_RE.findall(evidence_text)}
    evidence_dates = {
        re.sub(r"\s+", "", match).casefold()
        for match in _DATE_TOKEN_RE.findall(evidence_text)
    }
    evidence_numbers = {
        number for number in
        (_normal_number(match) for match in _NUMBER_RE.findall(evidence_text))
        if number is not None
    }
    fraction_components = set()

    # A model may spell executable ``0.5`` as the equivalent ``1/2``
    # when it explains a formula. The value is still grounded, but the
    # numerator and denominator otherwise look like two invented
    # numbers. Permit those components only when their exact quotient
    # is present in the evidence.
    fractions = re.findall(r"(?<!\d)([-+]?\d+)\s*/\s*([-+]?\d+)(?!\d)", answer)
    fractions += re.findall(
        r"\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}",
        answer,
    )
    for numerator, denominator in fractions:
        try:
            top = Decimal(numerator)
            bottom = Decimal(denominator)
            if bottom and top / bottom in evidence_numbers:
                fraction_components.update({top, bottom})
        except (InvalidOperation, ZeroDivisionError):
            continue

    def supported(term: str) -> bool:

        lowered = term.lower()

        if _DOTTED_ABBREVIATION_RE.fullmatch(term):
            shape = re.sub(r"[^a-z]", "", lowered)

            if shape in evidence_word_shapes:
                return True

            # OCR can join a following fragment to an abbreviation:
            # "M.Sc.exis" is the noisy source form of "M.Sc" followed
            # by more text. Accept the punctuation-delimited prefix,
            # but never a plain alphabetic prefix such as "Ann" in
            # "Anna", which would weaken name verification.
            return bool(re.search(
                rf"(?<![A-Za-z]){re.escape(term)}(?:\.|[^A-Za-z]|$)",
                " ".join(evidence),
                re.IGNORECASE,
            ))

        if _IDENTIFIER_RE.fullmatch(term):
            return lowered in evidence_ids

        if _DATE_TOKEN_RE.fullmatch(term):
            return re.sub(r"\s+", "", lowered) in evidence_dates

        if _NUMBER_RE.fullmatch(term):
            number = _normal_number(term)
            return (
                number is not None
                and (number in evidence_numbers or number in fraction_components)
            )

        if term.isupper():
            shape = re.sub(r"[^a-z]", "", lowered)

            # Punctuation is not meaning: UK and U.K. identify the same
            # scope.  Also accept a small set of standard expansions
            # commonly used in answers and source titles. This never
            # applies to mixed letter-number identifiers, which were
            # handled strictly above.
            if shape in evidence_word_shapes:
                return True

            phrase = {
                "uk": "united kingdom",
                "us": "united states",
                "pm": "prime minister",
            }.get(shape)
            if phrase and phrase in evidence_text:
                return True

        if lowered in evidence_words:
            return True

        expanded = _WORD_EQUIVALENTS.get(lowered, lowered)
        return expanded in {
            _WORD_EQUIVALENTS.get(word, word) for word in evidence_words
        }

    unsupported = [t for t in terms if not supported(t)]

    if unsupported:
        print(f"[GROUNDING] answer contains unsupported term(s): {unsupported}")

    return not unsupported and _outcome_relationships_supported(answer, evidence)


def _supported_answer_subset(answer: str, evidence: list) -> str:
    """Keep independently grounded sentences from a mixed-quality draft.

    A useful first sentence should not be replaced by several paragraphs
    of raw source text just because a later sentence adds one unsupported
    detail. Every retained sentence is checked with the same strict
    identifier/number/name verifier as the full answer. Sentences with
    no concrete terms are omitted so generic filler cannot masquerade as
    a verified answer.
    """

    supported = []

    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(answer or "")):
        sentence = sentence.strip()

        if (
            sentence
            and _key_terms(sentence)
            and _answer_within_evidence(sentence, evidence)
        ):
            supported.append(sentence)

    return " ".join(supported).strip()


_GENERIC_QUERY_TERMS = {
    "current", "currently", "today", "tonight", "now", "latest",
    "recent", "weather", "forecast", "news", "what", "whats", "the",
    "who", "where", "when", "why", "how", "is", "are", "was", "were",
    "in", "on", "at", "of", "for", "to", "and", "or", "a", "an",
    "please", "tell", "me", "about", "right",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _query_terms(text: str) -> set:
    """Distinctive terms from a search query.

    Deliberately not _key_terms: that lowercases sentence-initial
    words to avoid mistaking "Currently" for a proper noun, which is
    right for prose but loses the subject of a query like "London
    weather today" - the one word that matters most here.
    """

    terms = set(re.findall(r"\b[a-z0-9][a-z0-9._-]*\b", text.casefold()))
    return {
        term for term in terms
        if term not in _GENERIC_QUERY_TERMS and not term.isdigit()
    }


def _evidence_relevance_score(evidence: list, queries: list) -> int:
    """Count meaningful query terms present in an evidence block.

    Search queries often include today's date. A result mentioning only
    that date or only the country is not enough to answer a two-part
    subject such as "capital of Australia". Requiring two distinctive
    terms when available prevents loosely related snippets from becoming
    the fallback answer, while a focused query such as "London weather"
    still works because generic words such as "weather" are removed.
    """

    query_terms = set()
    for query in queries:
        query_terms |= {t.lower() for t in _query_terms(str(query))}

    if not query_terms:
        return 0

    evidence_terms = set(re.findall(
        r"\b[a-z0-9][a-z0-9._-]*\b",
        _normalize_ws(" ".join(evidence)),
    ))
    matched = set(query_terms & evidence_terms)
    for term in query_terms - matched:
        if len(term) < 4:
            continue
        if get_close_matches(term, evidence_terms, n=1, cutoff=0.84):
            matched.add(term)

    score = len(matched)
    required = 1 if len(query_terms) == 1 else 2
    return score if score >= required else 0


def _evidence_is_relevant(evidence: list, queries: list) -> bool:
    """True when the evidence is actually about what was searched for.

    Falling back to verbatim evidence is only safe if the evidence is
    on topic. When the model cites badly, the fallback would otherwise
    state page furniture as though it were the answer - "This shows
    the pollen level for the region this location is in" in response
    to a question about London's weather. That is confidently wrong,
    which is worse than admitting the lookup failed.
    """

    return _evidence_relevance_score(evidence, queries) > 0


def _concise_web_fallback(evidence: list, queries: list) -> str:
    """Turn verified web evidence into a short last-resort answer.

    This does not ask a model to reinterpret anything. It ranks one-
    and two-sentence windows from already-verified evidence by overlap
    with the search subject, then returns at most two compact windows.
    The old fallback joined every selected block and could dump page
    furniture, navigation and several unrelated paragraphs into chat.
    """

    query_terms = set()
    for query in queries or []:
        query_terms |= _query_terms(str(query))

    candidates = []

    for source_index, item in enumerate(evidence or []):
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue

        fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[.!?])\s+", text)
            if fragment.strip()
        ] or [text]

        for start in range(len(fragments)):
            for width in (1, 2):
                window = " ".join(fragments[start:start + width]).strip()
                if not window:
                    continue

                words = set(re.findall(
                    r"\b[a-z0-9][a-z0-9._-]*\b", window.casefold()
                ))
                overlap = len(words & query_terms)
                concrete = {
                    term.casefold() for term in _key_terms(window)
                }
                extra_specifics = len(concrete - query_terms)
                candidates.append((
                    -overlap,
                    -extra_specifics,
                    len(window),
                    source_index,
                    start,
                    window,
                ))

    if not candidates:
        return ""

    candidates.sort()
    selected = []
    total = 0

    for candidate in candidates:
        window = candidate[-1]
        normalized = _normalize_ws(window)

        if any(
            normalized in _normalize_ws(existing)
            or _normalize_ws(existing) in normalized
            for existing in selected
        ):
            continue

        remaining = 440 - total
        if remaining < 80:
            break

        if len(window) > remaining:
            window = window[:remaining].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

        selected.append(window)
        total += len(window) + 1

        if len(selected) == 2:
            break

    return "Verified evidence: " + " ".join(selected) if selected else ""


def _source_organization(url: str) -> str:
    """Stable organization key used for independent-source counts."""

    try:
        host = (urlsplit(str(url)).hostname or "").casefold().strip(".")
    except ValueError:
        return ""

    if host.startswith("www."):
        host = host[4:]

    labels = host.split(".") if host else []
    if len(labels) < 2:
        return host

    compound_suffixes = {
        "co.uk", "org.uk", "gov.uk", "com.au", "com.br", "co.in",
        "co.jp", "co.nz", "com.sg", "com.tr",
    }
    tail_two = ".".join(labels[-2:])
    size = 3 if tail_two in compound_suffixes and len(labels) >= 3 else 2
    return ".".join(labels[-size:])


def _is_web_origin(origin) -> bool:
    try:
        return urlsplit(str(origin)).scheme.casefold() in {"http", "https"}
    except ValueError:
        return False


def _source_records(origins: list, results: list) -> list:
    """User-facing source metadata, deduplicated without losing order."""

    pages = {}
    for result in results:
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            continue
        for page in data:
            if isinstance(page, dict) and page.get("url"):
                pages[str(page["url"])] = page

    records = []
    seen = set()
    for origin in origins:
        origin = str(origin or "").strip()
        if not origin or origin in seen:
            continue
        seen.add(origin)

        if _is_web_origin(origin):
            page = pages.get(origin, {})
            records.append({
                "label": page.get("title") or _source_organization(origin) or origin,
                "url": origin,
                "trusted": bool(page.get("trusted")),
                "kind": "web",
            })
        else:
            try:
                label = Path(origin).name or origin
            except Exception:
                label = origin
            records.append({
                "label": label,
                "path": origin,
                "trusted": True,
                "kind": "local",
            })

    return records


_LATEST_STABLE_VERSION_RE = re.compile(
    r"\b(?:latest|newest|current|most\s+recent)\b.{0,55}"
    r"\b(?:stable\s+)?(?:version|release)\b|"
    r"\b(?:version|release)\b.{0,55}\b(?:latest|newest|current)\b",
    re.IGNORECASE,
)
_SEMVER_RE = re.compile(
    r"(?<![\d.])v?(?P<major>\d{1,3})\."
    r"(?P<minor>\d{1,3})(?:\.(?P<patch>\d{1,4}))?(?![\d.])",
    re.IGNORECASE,
)
_UNSTABLE_VERSION_CONTEXT = re.compile(
    r"\b(?:alpha|beta|rc\d*|release\s+candidate|preview|pre-release|"
    r"nightly|development|dev\s+build|future|planned|schedule)\b",
    re.IGNORECASE,
)
_RELEASE_VERSION_CONTEXT = re.compile(
    r"\b(?:stable|release|released|available|download|maintenance|version)\b",
    re.IGNORECASE,
)


def _asks_latest_stable_version(message: str) -> bool:
    """A current software-release question, independent of product."""

    return bool(_LATEST_STABLE_VERSION_RE.search(message or ""))


def _version_query_subject(message: str) -> str:
    """Best product name available in a latest-version question."""

    words = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", message or "")
    ignored = {
        "gimme", "give", "tell", "show", "me", "the", "a", "an",
        "latest", "newest", "current", "most", "recent", "stable",
        "what", "whats", "is", "version", "release", "of", "for",
        "please", "pls", "right", "now",
    }
    useful = [word for word in words if word.casefold() not in ignored]
    # Product names such as "Visual Studio Code" need more than the
    # final word, while limiting the span avoids turning the whole
    # question into a search subject.
    return " ".join(useful[-3:]) if useful else ""


def _latest_stable_version_answer(message: str, results: list):
    """Extract the highest stable semantic version from official pages.

    Local models repeatedly chose a newly published security release of
    an *older* supported branch over the actual latest branch. This is
    ordering, not interpretation: once official release pages provide
    semantic versions, deterministic tuple comparison is both faster
    and more reliable. If authority or release wording is unclear, this
    helper declines and the normal grounded response path remains in
    control.
    """

    if not _asks_latest_stable_version(message):
        return None

    pages = [
        page
        for result in results
        if isinstance(result, dict)
        for page in (
            result.get("data") if isinstance(result.get("data"), list) else []
        )
        if isinstance(page, dict) and page.get("official") and page.get("url")
    ]
    if not pages:
        return None

    requested_product = _version_query_subject(message)
    product_words = re.findall(
        r"[A-Za-z0-9.+#-]+", requested_product.casefold()
    )
    candidates = []
    for page in pages:
        page_title = str(page.get("title") or "")
        # A release *schedule* can mention versions years newer than
        # anything currently available. It is useful research evidence,
        # but cannot establish the latest stable release.
        if re.search(r"\b(?:release\s+schedule|roadmap)\b", page_title,
                     re.IGNORECASE):
            continue

        text = "\n".join(
            str(page.get(key) or "")
            for key in ("title", "snippet", "content")
        )
        for match in _SEMVER_RE.finditer(text):
            start = max(0, match.start() - 90)
            end = min(len(text), match.end() + 120)
            context = text[start:end]

            if _UNSTABLE_VERSION_CONTEXT.search(context):
                continue
            if not _RELEASE_VERSION_CONTEXT.search(context):
                continue

            # Do not select an unrelated high version merely because it
            # appears somewhere on the requested product's official
            # domain. Each product word must be close to this exact
            # version (for example, "Python 3.14.7").
            local = context.casefold()
            if product_words and not all(word in local for word in product_words):
                continue
            if requested_product:
                paired = re.search(
                    rf"\b{re.escape(requested_product)}\b"
                    rf"(?:\s+(?:version|release))?\s+v?"
                    rf"{re.escape(match.group(0).lstrip('vV'))}\b",
                    context,
                    re.IGNORECASE,
                )
                if not paired:
                    continue

            version = tuple(
                int(value or 0)
                for value in (
                    match.group("major"),
                    match.group("minor"),
                    match.group("patch"),
                )
            )
            rendered = ".".join(
                part for part in (
                    match.group("major"), match.group("minor"),
                    match.group("patch"),
                ) if part is not None
            )
            candidates.append((version, rendered, page, context))

    if not candidates:
        return None

    _, version_text, page, context = max(candidates, key=lambda item: item[0])

    # The requested subject is safer than inferring a product from an
    # arbitrary preceding word such as "of" on the source page.
    product = requested_product or "software"

    product = product if any(ch.isupper() for ch in product) else product.title()
    return {
        "answer": (
            f"The latest stable {product} version shown on its official "
            f"release page is {product} {version_text}."
        ),
        "url": str(page["url"]),
        "version": version_text,
    }


def _search_queries_from(steps: list, results: list) -> list:
    """Collect the queries actually issued for web.search steps."""

    queries = []

    for step, result in zip(steps, results):

        if step.get("type") != "web.search":
            continue

        # The executor normalizes queries, so prefer what it reports
        # actually running over what the planner originally asked for.
        issued = result.get("queries")

        if isinstance(issued, list) and issued:
            queries.extend(issued)
            continue

        args = step.get("args", {})

        if isinstance(args.get("query"), str):
            queries.append(args["query"])

        if isinstance(args.get("queries"), list):
            queries.extend(a for a in args["queries"] if isinstance(a, str))

    return queries


def _count_corroborating_sources(evidence_sentence: str, results: list, origin_url: str = None) -> int:
    """
    Counts how many pages OTHER than the evidence's own origin page
    also contain this sentence's key facts. The origin page always
    counts separately and automatically - by construction, it's
    where the sentence was tagged from, so re-matching against it
    here is both redundant and risky: if the fuzzy match ever fails
    against the very page a sentence came from (encoding, symbols,
    formatting), it would wrongly discard a correct answer instead
    of just skipping a caveat. Only independent, other pages count
    toward this number.
    """

    terms = _key_terms(evidence_sentence)

    if not terms:
        return 0

    count = 0
    counted = set()
    origin_organization = _source_organization(origin_url)

    for result in results:

        data = result.get("data")

        if not isinstance(data, list):
            continue

        for page in data:

            organization = _source_organization(page.get("url"))

            if not organization or organization == origin_organization:
                continue

            if organization in counted:
                continue

            page_text = _normalize_ws(
                f"{page.get('title', '')} {page.get('snippet', '')} {page.get('content', '')}"
            )

            # Reuse the exact value/identifier matcher. Substring
            # corroboration made a page containing 117285 appear to
            # confirm an unrelated claim about 285.
            if _answer_within_evidence(evidence_sentence, [page_text]):
                count += 1
                counted.add(organization)

    return count

# The sentence ids the prompt uses to make evidence citable - [S1],
# [S2] and so on. They are scaffolding for the grounding check and are
# never meant to be read.
#
# They only started leaking once script output was tagged: a file's
# text gets paraphrased, so its tags rarely survive, but a computed
# result is copied out exactly as printed. Asked for the prefix of A+B,
# the reply came back "The prefix of A+B is [S1] + A B."
#
# Not preceded by a word character, so an index in code - "array[S1]" -
# is left alone. A tag always stands on its own.
_SENTENCE_TAG = re.compile(r"(?<!\w)\[S\d+\]\s*")


def _strip_tags(text: str) -> str:
    """Remove evidence markers from something about to be shown."""

    return _SENTENCE_TAG.sub("", text or "")


def _strip_markdown(text: str, flatten: bool = True) -> str:
    """Remove markdown the page cannot render.

    Replies are put on screen as text rather than HTML, so asterisks
    and hashes arrive exactly as written - asked what 2+2 was, Athena
    answered "* **Rule:** This is a basic addition problem", visibly.

    `flatten` folds everything into one paragraph and ends each line
    with a full stop, which suits a grounded answer assembled from
    fragments of a document. It is wrong for a chat reply: those have
    real structure, and lines that are code, a URL or a list item would
    each acquire a full stop that was never meant to be there.
    """

    # Emphasis markers sit tight against the text they emphasise, so a
    # star with a space on either side is a multiplication sign and is
    # left alone. Without this the postfix of "C*(D+E+F*G)/H" came back
    # as "C D E + F G  +  H /" - the two operators read as an emphasis
    # pair and deleted, quietly turning a correct answer into a wrong
    # one after every check upstream had passed.
    text = re.sub(r"\*\*(?!\s)(.*?)(?<!\s)\*\*", r"\1", text)
    text = re.sub(r"__(?!\s)(.*?)(?<!\s)__", r"\1", text)

    # Bold markers the paired pattern missed because the spacing was
    # uneven - "**Paths **" closes with a space before it and left a
    # stray asterisk at each end.
    #
    # Removed only where the pair touches a non-space on one side,
    # which is what makes it markup. "2 ** 8" has a space either side
    # and is exponentiation, so it survives.
    text = re.sub(r"(?<!\s)\*\*(?!\*)|(?<!\*)\*\*(?!\s)", "", text)

    # Italics. The asterisk must not sit against a word character or
    # another asterisk on either side, which is what keeps arithmetic
    # intact: "C*(D+E+F*G)/H" once came back as "C D E + F G  +  H /",
    # its operators read as an emphasis pair and deleted, turning a
    # correct answer into a wrong one after every upstream check had
    # passed. Excluding a neighbouring asterisk matters too - without
    # it "2 ** 8" was read as an empty italic and became "2  8".
    text = re.sub(r"(?<![\w*])\*(?![\s*])(.*?)(?<![\s*])\*(?![\w*])", r"\1", text)

    # Code fences and inline backticks. The page shows replies as text,
    # so a fence renders as three literal backticks introducing nothing.
    text = re.sub(r"^\s*```.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            if not flatten:
                # Kept, so paragraphs stay paragraphs.
                cleaned.append("")
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[\*\-\u2022]\s+", "", line)
        line = re.sub(r"^\d+[\.\)]\s+", "", line)

        if flatten and line and not line.endswith((".", "!", "?", ":")):
            line += "."

        cleaned.append(line)

    if not flatten:
        return "\n".join(cleaned).strip()

    return " ".join(cleaned)


def _grounding_line() -> str:

    now = datetime.now().astimezone()

    tz_name = now.tzname() or "local time"
    utc_offset = now.strftime("%z")

    return (
        f"Today's real date is {now.strftime('%A, %B %d, %Y')} "
        f"({now.strftime('%Y-%m-%d')}), and the current local time is "
        f"{now.strftime('%H:%M')} {tz_name} (UTC{utc_offset}). This is "
        f"accurate, current, ground truth - use it only for date/time "
        f"reasoning that is already directly supported by the evidence "
        f"(for example, converting a UTC timestamp in a source to this "
        f"local time to judge if something is 'today'), and trust it "
        f"over any date you might otherwise assume. Your training data is very old.\n"
    )


_FILESYSTEM_TYPES = {
    "filesystem.read", "filesystem.list",
    "filesystem.exists", "filesystem.info", "filesystem.search",
    "filesystem.semantic_search",
}
_CODE_TYPES = {"python.run", "code.run", "code.generate"}

# Structured lookups. Graded like the datetime prompt rather than the
# web one: the figures arrive already labelled from a single known
# source, so there is no page furniture to distrust and no
# corroboration to weigh - only whether the reply repeats the numbers
# it was given.
_LIVE_DATA_TYPES = {"weather.current", "finance.quote", "finance.exchange"}


# Requests where the code itself is the deliverable.
_CODE_IS_THE_POINT_RE = re.compile(
    r"\b(?:script|program|function|code|module|class|python\s+file)\b",
    re.IGNORECASE,
)

_DO_NOT_RUN_RE = re.compile(
    r"\b(?:do\s+not|don'?t|dont|without)\s+(?:also\s+)?(?:run|execute|launch)\b"
    r"|\b(?:just|only)\s+(?:write|save|create|generate)\b",
    re.IGNORECASE,
)


def _wants_built_document(message: str) -> bool:
    """Whether a generated script should be run to get the answer.

    Asked for a script, the file is the answer and running it uninvited
    would be wrong. Asked for anything else - a presentation, an
    acceleration, a postfix expression - the script is only the means,
    and stopping at it leaves the user holding a .py file instead of
    what they asked for.
    """

    text = message or ""

    if _DO_NOT_RUN_RE.search(text):
        return False

    return not _CODE_IS_THE_POINT_RE.search(text)


def _script_error(result: dict) -> str:
    """The error a script died with, or empty if it ran cleanly.

    A script can "succeed" as a step - the subprocess was launched and
    returned - while having produced nothing but a traceback, so the
    return code and stderr are what decide, not the success flag.
    """

    if not isinstance(result, dict):
        return ""

    data = result.get("data")

    if not isinstance(data, dict):
        return ""

    stderr = (data.get("stderr") or "").strip()

    if data.get("return_code") not in (0, None) and stderr:
        return stderr

    # Some failures exit zero but print a traceback anyway.
    if "Traceback (most recent call last)" in stderr:
        return stderr

    # A script that ran cleanly and computed nothing.
    #
    # Asked for the prefix of A+B, the generated program tokenised the
    # expression with a pattern that matched "A+B" as one token, then
    # emitted nothing at all - and printed "Prefix Expression:" with an
    # empty value after it. Nothing had failed by any measure the
    # checks above use, so that blank went on to be composed into an
    # answer: "The prefix expression is blank."
    #
    # An empty result is a failure. It is worth one repair attempt with
    # the output shown, which is exactly what a real error gets.
    stdout = data.get("stdout") or ""

    if _prints_nothing(stdout):
        return (
            "The program ran without error but produced no result. "
            f"Its entire output was:\n{stdout.strip() or '(nothing)'}\n"
            "Every value it was asked to print must actually be "
            "computed and printed."
        )

    # The program checked its own answer and the check failed.
    #
    # Asked to convert between representations of the same value -
    # notation, a number base, a unit - the generation prompt now
    # requires the script to verify its result with an independent
    # method before printing it: this is what catches an algorithm
    # that runs cleanly and produces the WRONG representation, such as
    # a prefix conversion that quietly skipped reversing its input and
    # printed the postfix answer instead. Nothing about a run like that
    # looks like a failure by any check above - return code 0, empty
    # stderr, real-looking output - which is exactly why the check has
    # to live inside the program.
    if "VERIFICATION FAILED" in stdout:
        return (
            "The program's own verification check failed - it computed "
            f"an answer that did not match on independent evaluation. "
            f"Its output was:\n{stdout.strip()}\n"
            "Find the mistake in the conversion, not in the check."
        )

    # A verification step that broke instead of disagreeing.
    #
    # The prompt asks for exactly one of two outputs - the result, or
    # VERIFICATION FAILED - but the first real run did neither: the
    # check's own eval() raised NameError on an undefined letter, the
    # script caught it, and printed "There was an error during
    # verification because the name 'A' is not defined" ALONGSIDE the
    # correct answer. That sentence matches no exact phrase, so it
    # would otherwise have gone straight into the reply next to the
    # result it was supposed to be checking - reporting success and
    # failure in the same breath.
    #
    # A phrase match rather than an exact string, because the prompt
    # cannot pin down the model's exact wording for a crash the way it
    # can for the deliberate failure line.
    if re.search(r"error (?:during|in) verif", stdout, re.IGNORECASE):
        return (
            "The verification step itself failed to run, and its error "
            f"was printed alongside the result instead of stopping the "
            f"program. Output was:\n{stdout.strip()}\n"
            "Wrap the verification in try/except and treat any "
            "exception as VERIFICATION FAILED - never print both an "
            "error and a result."
        )

    return ""


_SCALE_GROUP_RE = re.compile(
    r"\b(?:for|serves?|servings?\s*[:=]?)\s*"
    r"(?P<count>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<label>people|persons?|servings?|portions?|guests?|items?|units?)\b",
    re.IGNORECASE,
)
_NEW_SCALE_COUNT_RE = re.compile(
    r"\b(?:for|to|serves?|servings?\s*[:=]?)\s*"
    r"(?P<count>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?:people|persons?|servings?|portions?|guests?|items?|units?))?\b",
    re.IGNORECASE,
)
_SCALABLE_TOTAL_RE = re.compile(
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>grams?|g|kilograms?|kg|millilit(?:er|re)s?|ml|"
    r"lit(?:er|re)s?|l|cups?|tablespoons?|tbsp|teaspoons?|tsp|"
    r"ounces?|oz|pounds?|lb|met(?:er|re)s?|m|centimet(?:er|re)s?|cm)\b",
    re.IGNORECASE,
)


def _decimal_literal(value: Decimal) -> str:
    """A plain Python numeric literal without exponent notation."""

    return format(value, "f")


def _proportional_scaling_step(state, message: str):
    """Build a checked proportional calculation from the previous answer.

    This is deliberately general rather than recipe-specific. It applies only
    when the previous answer states a physical total for a labelled group and
    the new request changes that group size. The source quantity remains a
    total; it is never silently relabelled as a per-person value.
    """

    if not re.search(r"\b(?:instead|scale|scaled|serves?|for\s+\d)\b", message or "", re.I):
        return None

    previous = ""
    for item in reversed(getattr(state, "messages", []) or []):
        if item.get("role") == "assistant":
            previous = str(item.get("content") or "")
            break
    if not previous:
        return None

    old_group = _SCALE_GROUP_RE.search(previous)
    new_groups = list(_NEW_SCALE_COUNT_RE.finditer(message or ""))
    quantities = list(_SCALABLE_TOTAL_RE.finditer(previous))
    if not old_group or not new_groups or not quantities:
        return None

    try:
        old_count = Decimal(old_group.group("count").replace(",", ""))
        new_count = Decimal(new_groups[-1].group("count").replace(",", ""))
    except InvalidOperation:
        return None
    if old_count <= 0 or new_count <= 0 or old_count == new_count:
        return None

    # If more than one total appears, choose the one whose nearby label best
    # overlaps the user's words. This keeps "scale the flour" attached to the
    # flour amount rather than another ingredient in the same answer.
    request_terms = {
        token for token in re.findall(r"[a-z]{3,}", (message or "").casefold())
        if token not in {"how", "much", "need", "instead", "cooking", "make", "scale"}
    }
    ranked = []
    for position, quantity in enumerate(quantities):
        nearby = previous[max(0, quantity.start() - 70):quantity.end() + 70]
        # An explicitly per/each quantity is already a rate, not a total.
        before = previous[max(0, quantity.start() - 24):quantity.start()]
        after = previous[quantity.end():quantity.end() + 24]
        if (
            re.search(r"\b(?:per|each)\s*$", before, re.I)
            or re.match(r"\s*(?:per|each)\b", after, re.I)
        ):
            continue
        nearby_terms = set(re.findall(r"[a-z]{3,}", nearby.casefold()))
        ranked.append((-len(request_terms & nearby_terms), position, quantity))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    quantity = ranked[0][2]

    try:
        original_total = Decimal(quantity.group("amount").replace(",", ""))
    except InvalidOperation:
        return None
    if original_total <= 0:
        return None

    unit = quantity.group("unit")
    code = (
        f"original_group_size = {_decimal_literal(old_count)}\n"
        f"original_total = {_decimal_literal(original_total)}\n"
        f"new_group_size = {_decimal_literal(new_count)}\n"
        "rate_per_group_member = original_total / original_group_size\n"
        "new_total = rate_per_group_member * new_group_size\n"
        f"print('Original total: {{}} {unit}'.format(original_total))\n"
        "print('Original group size: {}'.format(original_group_size))\n"
        "print('Rate: original_total / original_group_size = {}'.format(rate_per_group_member))\n"
        "print('New group size: {}'.format(new_group_size))\n"
        f"print('New total: {{}} {unit}'.format(new_total))"
    )
    return {"type": "code.run", "args": {"code": code}}


# A line of output with a label and nothing after it - "Result:" with
# the result missing.
_EMPTY_LABEL = re.compile(r"^[^:]{0,60}:\s*$")


def _prints_nothing(stdout: str) -> bool:
    """Whether the output carries no computed value.

    Labels alone do not count. A program that prints "Prefix
    Expression:" and stops has told the reader what is missing, not
    what the answer is.

    The test is the LAST line rather than every line, because that is
    where the result goes. "Steps:" partway through is a heading with
    working underneath it and perfectly fine; the same line at the end
    is a result that never arrived. The real failure printed

        Infix Expression: A+B
        Prefix Expression:

    where only the second line is empty - so checking that every line
    is blank would have missed it entirely.
    """

    lines = [line.strip() for line in (stdout or "").splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return True

    return bool(_EMPTY_LABEL.match(lines[-1]))


def _generated_script_path(step: dict, result: dict):
    """Path of the script just generated, when it is runnable Python."""

    path = (result.get("data") or {}).get("path") if isinstance(result.get("data"), dict) else None
    path = path or step.get("args", {}).get("path")

    if isinstance(path, str) and path.lower().endswith(".py"):
        return path

    return None


# ----------------------------------------------------------------
# Requests whose capability is unambiguous
# ----------------------------------------------------------------

_FILE_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|\\\\)[^\"\r\n]*?"
    r"\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|md|py|json|log|png|jpe?g|env))"
    # A full stop commonly separates the filename from the next sentence:
    # "read notes.txt. what is the total?". It is punctuation, not part of
    # the path.
    r"(?=$|[\s,;.!?])",
    re.IGNORECASE,
)

_ARTIFACT_EXTENSIONS = {
    ".pptx": "PowerPoint presentation",
    ".docx": "Word document",
    ".xlsx": "Excel workbook",
}

_RUN_LAST_SCRIPT_RE = re.compile(
    r"^\s*(?:please\s+)?(?:run|execute|test)\s+"
    r"(?:(?:that|the|this)\s+)?(?:script|program|python\s+file)"
    r"(?:\s+now)?\s*[.!?]*$",
    re.IGNORECASE,
)

_DELETE_LOCAL_RE = re.compile(
    r"^\s*(?:please\s+)?(?:delete|remove|erase)\b.*"
    r"(?:\bfile\b|\bfolder\b|\bdirectory\b|[A-Za-z]:[\\/])",
    re.IGNORECASE,
)

_PUBLIC_INSTRUCTIONS_QUESTION = re.compile(
    r"\b(?:(?:your|athena(?:'s)?)\s+(?:system\s+|chat\s+|internal\s+)?"
    r"(?:prompt|instructions?)|(?:system|chat|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?))\b",
    re.IGNORECASE,
)

# Product boundary, not a subject-specific knowledge rule. Athena may
# help inspect a contract, but it must never turn a request for legal
# certainty into web research and then imply that snippets amount to a
# compliance guarantee.
_LEGAL_GUARANTEE_RE = re.compile(
    r"(?=.*\b(?:guarantee|certify|assure|confirm)\b)"
    r"(?=.*\b(?:legal(?:ly)?|law|laws|compliance|compliant|contract)\b)",
    re.IGNORECASE | re.DOTALL,
)

_FIRST_MESSAGE_QUERY = re.compile(
    r"\b(?:first\s+(?:thing|message).{0,40}(?:said|sent|wrote|asked)"
    r"|what.{0,30}first.{0,30}(?:said|sent|wrote|asked))\b",
    re.IGNORECASE,
)
_NAME_MEMORY_QUERY = re.compile(
    r"\b(?:what(?:'?s| is)\s+my\s+name|"
    r"what\s+should\s+(?:you|u)\s+call\s+me|"
    r"remember\s+my\s+name|what\s+name\b)\b",
    re.IGNORECASE,
)
_PREFERRED_NAME_QUERY = re.compile(
    r"\b(?:what\s+should\s+(?:you|u)\s+call\s+me|"
    r"what\s+do\s+(?:you|u)\s+call\s+me|"
    r"what\s+did\s+i\s+ask\s+(?:you|u)\s+to\s+call\s+me)\b",
    re.IGNORECASE,
)
_ORIGINAL_NAME_QUERY = re.compile(
    r"\b(?:at\s+the\s+(?:very\s+)?start|original(?:ly)?|"
    r"name.{0,24}(?:tell|told|said)|(?:tell|told|said).{0,24}name)\b",
    re.IGNORECASE,
)
_COURSE_MEMORY_QUERY = re.compile(
    r"\b(?:what\s+course|which\s+course|what\s+am\s+i\s+studying|"
    r"what.{0,20}(?:study|studying)|course\s+did\s+i)\b",
    re.IGNORECASE,
)
_CALL_ME = re.compile(
    r"\bcall\s+me\s+(?:as\s+)?(?P<name>[A-Za-z][A-Za-z'-]{0,30})",
    re.IGNORECASE,
)
_MY_NAME_IS = re.compile(
    r"\bmy\s+name\s+is\s+(?P<name>[A-Za-z][A-Za-z'-]{0,30})",
    re.IGNORECASE,
)
_I_AM_NAME = re.compile(
    r"\b(?:i\s+am|i'?m)\s+(?P<name>[A-Za-z][A-Za-z'-]{1,30})"
    r"(?=\s*(?:btw|by\s+the\s+way|[,!.]|$))",
    re.IGNORECASE,
)
_NOT_NAMES = {
    "fine", "good", "great", "okay", "ok", "well", "just", "trying",
    "happy", "sad", "tired", "here", "back", "sorry", "ready",
}
_COURSE_DECLARATION = re.compile(
    r"\b(?:studying|study|majoring\s+in|my\s+course\s+is|doing)\s+"
    r"(?P<course>.+?)(?=\s+at\s+[A-Za-z]|[,.;]|$)",
    re.IGNORECASE,
)
_COURSE_CORRECTION = re.compile(
    r"\b(?:switched|changed|moved)\s+to\s+(?P<course>[^,.;]+)",
    re.IGNORECASE,
)
_EDUCATION_HINT = re.compile(
    r"\b(?:course|degree|major|year|yr|engineering|science|sci|computer|"
    r"mechanical|electrical|civil|law|medicine|business|economics|arts|"
    r"physics|chemistry|math(?:s|ematics)?)\b",
    re.IGNORECASE,
)


def _asks_for_legal_guarantee(message: str) -> bool:
    return bool(_LEGAL_GUARANTEE_RE.search(message or ""))


def _asks_about_public_instructions(message: str) -> bool:
    """Whether a reply should expose only public static prompt context.

    Athena's source prompts are public, but its rolling conversation notes
    and document evidence are not part of those prompts. Recognising this
    request lets the same local response model answer transparently without
    receiving unrelated private runtime context.
    """

    return bool(_PUBLIC_INSTRUCTIONS_QUESTION.search(message or ""))


_CAPITULATION_SENTENCE = re.compile(
    r"\b(?:my apologies|i apologize|i apologise|my mistake|"
    r"i stand corrected|sorry about that|still learning)\b",
    re.IGNORECASE,
)


def _remove_unnecessary_capitulation(message: str, answer: str) -> str:
    """Drop apology-only tail sentences after a firm factual rejection.

    In long chats, a model can correctly say "No, X is the answer" and
    then apologize as though the user's false correction were right.
    Only trailing sentences are removed, only when the reply begins with
    an explicit rejection, so genuine corrections and ordinary apologies
    are left untouched.
    """

    if not re.search(
        r"\b(?:isn'?t|aren'?t|wasn'?t|weren'?t|u\s+sure|are\s+you\s+sure|"
        r"but\s+is|nah\b)",
        message or "",
        re.IGNORECASE,
    ):
        return answer
    if not re.match(r"\s*(?:no\b|not\s+quite\b|actually[, ]+no\b)",
                    answer or "", re.IGNORECASE):
        return answer

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    if len(sentences) < 2:
        return answer
    kept = [sentences[0]] + [
        sentence for sentence in sentences[1:]
        if not _CAPITULATION_SENTENCE.search(sentence)
    ]
    return " ".join(kept).strip()


def _avoidable_typo_clarification(message: str, answer: str) -> bool:
    """Whether a typo-tolerant retry is cheaper than failing needlessly.

    This is intentionally narrow: a second answer call is made only
    when the first word is a close misspelling of a question word and
    the draft either asks what the user meant or refuses a settled
    ordinary question solely because it was misspelled. Legitimate
    requests for missing details (such as weather with no city) are
    left alone.
    """

    if missing_subject_question(message):
        return False

    words = re.findall(r"[A-Za-z']+", message or "")
    if len(words) < 4:
        return False

    first = words[0].casefold().replace("'", "")
    question_words = ("what", "whats", "who", "where", "when", "why", "how", "which")
    if first in question_words or not get_close_matches(first, question_words, n=1, cutoff=0.72):
        return False

    clarification = re.search(
        r"\b(?:which .{0,60}(?:do you mean|are you asking)|"
        r"could you clarify|can you clarify|what do you mean|"
        r"which .{0,40}would you like)\b",
        answer or "",
        re.IGNORECASE,
    )
    avoidable_uncertainty = re.search(
        r"\b(?:i(?:'m| am) not sure|i (?:do not|don't) have current information|"
        r"i(?:'m| am) having trouble (?:providing|giving|finding)|"
        r"i(?:'m| am) struggling to (?:recall|answer|remember)|"
        r"i (?:seem|appear) to be having (?:difficulty|trouble)|"
        r"(?:would|will) need to (?:look|check|search)|"
        r"should be looked up|"
        r"best to (?:check|confirm)|need(?:s)? to be (?:checked|confirmed)|"
        r"cannot be certain)\b",
        answer or "",
        re.IGNORECASE,
    )
    echoed_question = str(answer or "").strip().endswith("?")
    return bool(clarification or avoidable_uncertainty or echoed_question)


def _clean_profile_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t,.;!?\"'")


def _update_user_profile(profile: dict, message: str) -> dict:
    """Keep only facts the user stated explicitly about themselves."""

    profile = dict(profile or {})
    text = message or ""
    match = _CALL_ME.search(text)

    if match:
        name = _clean_profile_value(match.group("name"))
        if name:
            profile["preferred_name"] = name

    match = _MY_NAME_IS.search(text) or _I_AM_NAME.search(text)

    if match:
        name = _clean_profile_value(match.group("name"))
        if name and name.casefold() not in _NOT_NAMES:
            profile.setdefault("name", name)

    correction = _COURSE_CORRECTION.search(text)
    declaration = _COURSE_DECLARATION.search(text)

    if correction and profile.get("course"):
        course = _clean_profile_value(correction.group("course"))
        if _EDUCATION_HINT.search(course):
            profile["course"] = course
    elif declaration:
        course = _clean_profile_value(declaration.group("course"))
        if _EDUCATION_HINT.search(course):
            profile["course"] = course

    return profile


def _profile_from_messages(messages: list) -> dict:
    profile = {}
    for item in messages or []:
        if item.get("role") == "user":
            profile = _update_user_profile(profile, item.get("content") or "")
    return profile


def _memory_reply(state: ConversationState, message: str):
    """Answer exact transcript/profile questions from stored state."""

    if _FIRST_MESSAGE_QUERY.search(message or ""):
        first = next(
            (
                str(item.get("content") or "")
                for item in state.messages
                if item.get("role") == "user" and str(item.get("content") or "").strip()
            ),
            "",
        )
        return f'You first said: "{first}"' if first else "You haven't said anything yet."

    wants_name = bool(_NAME_MEMORY_QUERY.search(message or ""))
    wants_course = bool(_COURSE_MEMORY_QUERY.search(message or ""))
    profile = state.user_profile or {}

    if not (wants_name or wants_course):
        return None

    asks_original_name = bool(_ORIGINAL_NAME_QUERY.search(message or ""))
    asks_preferred_name = bool(_PREFERRED_NAME_QUERY.search(message or ""))
    recorded_name = profile.get("name")
    preferred = profile.get("preferred_name")

    # A preferred form of address does not replace the person's name.
    # "What's my name?" therefore recalls the declared name, while
    # "what should you call me?" recalls the preference.  The previous
    # implementation used the preference for both and told Riya that
    # her name was RJ after she merely asked Athena to call her RJ.
    name = (
        (preferred or recorded_name)
        if asks_preferred_name and not asks_original_name
        else (recorded_name or preferred)
    )
    course = profile.get("course")

    if wants_name and wants_course and name and course:
        reply = f"You told me your name is {name}, and you're now studying {course}."
        if (
            preferred
            and recorded_name
            and preferred.casefold() != recorded_name.casefold()
            and not asks_preferred_name
        ):
            reply += f" You asked me to call you {preferred}."
        return reply
    if wants_name and name:
        if asks_preferred_name:
            return f"You asked me to call you {name}."
        reply = f"You told me your name is {name}."
        if (
            preferred
            and recorded_name
            and preferred.casefold() != recorded_name.casefold()
        ):
            reply += f" You asked me to call you {preferred}."
        return reply
    if wants_course and course:
        return f"You said you're now studying {course}."

    return None


def _pending_weather_step(message: str):
    """Turn a short answer to 'which city?' into an exact weather step."""

    if re.match(
        r"^\s*(?:who|what|why|how|when|where|can|could|would|should|"
        r"tell|write|explain|make|give|show|find|read|run|create)\b",
        message or "",
        re.IGNORECASE,
    ):
        return None

    text = re.sub(
        r"\b(?:please|pls|plz|now|rn|currently|today|yeah|yes|i\s+mean)\b",
        " ",
        message or "",
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*(?:in|at|for)\s+", "", text, flags=re.IGNORECASE)
    location = re.sub(r"[^\w\s.'-]", " ", text)
    location = re.sub(r"\s+", " ", location).strip()

    if location and len(location.split()) <= 5 and re.search(r"[A-Za-z]", location):
        return {"type": "weather.current", "args": {"location": location}}

    return None


def _path_from_message(message: str):
    """Return an explicit Windows file path without trailing prose."""

    match = _FILE_PATH_RE.search(message or "")
    return match.group("path").strip(" \t'\"") if match else None


def _directory_from_message(message: str):
    """Return a directory path used after a list/inside phrase."""

    match = re.search(
        r"\b(?:inside|contents?\s+of|list(?:\s+what'?s)?\s+in)\s+"
        r"(?P<path>(?:[A-Za-z]:[\\/]|\\\\)[^\r\n]+?)\s*[?.!]*$",
        message or "",
        re.IGNORECASE,
    )
    return match.group("path").strip(" \t'\"") if match else None


def _snippet_from_message(message: str):
    """Literal code supplied after 'run this snippet', if present."""

    text = message or ""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced and re.search(r"\b(?:run|execute|test)\b", text, re.IGNORECASE):
        return fenced.group(1).strip()

    inline = re.search(
        r"\b(?:run|execute|test)\s+(?:this\s+)?(?:python\s+)?"
        r"(?:snippet|code)\s*:\s*(?P<code>.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return inline.group("code").strip() if inline else None


def _direct_step_for(message: str, state: ConversationState):
    """Resolve capability contracts that require no model judgment.

    This is deliberately about generic operations, not school-subject
    functions: an exact path, literal code, or a content search maps to
    one existing capability regardless of what the file or code means.
    """

    # This machine's own clock is already deterministic. A named place
    # still goes through the planner so it can map natural language
    # ("Tokyo") to an IANA timezone ("Asia/Tokyo").
    if (
        asks_current_datetime(message)
        and not re.search(r"\b(?:in|at|for)\s+[A-Za-z]", message or "", re.IGNORECASE)
    ):
        return {"type": "system.datetime", "args": {}}

    # Explicitly current software versions must be refreshed even when
    # an answer to the same question appears in recent chat history.
    # Leaving this to the planner allowed it to return {"done": true}
    # without running a tool, silently reusing a potentially stale fact.
    if _asks_latest_stable_version(message):
        subject = _version_query_subject(message)
        query = " ".join(
            part for part in (subject, "latest stable version") if part
        )
        return {
            "type": "web.search",
            "args": {"query": query, "category": "general"},
        }

    snippet = _snippet_from_message(message)
    if snippet:
        return {"type": "code.run", "args": {"code": snippet}}

    if _RUN_LAST_SCRIPT_RE.match(message or ""):
        path = getattr(state, "last_generated_path", None)
        if path:
            return {"type": "python.run", "args": {"path": path}}

    if re.search(
        r"\b(?:which\s+(?:of\s+)?my\s+(?:docs?|documents?|files?)\s+mentions?|"
        r"where\s+(?:did|do|have)\s+i\s+(?:write|save|put|note)|"
        r"search\s+(?:my|through\s+my)\s+(?:files?|documents?|notes?))\b",
        message or "",
        re.IGNORECASE,
    ):
        return {
            "type": "filesystem.semantic_search",
            "args": {"query": (message or "").strip()},
        }

    path = _path_from_message(message)
    lowered = (message or "").casefold()

    if path:
        if re.search(r"\b(?:exist|exists|present|there)\b", lowered):
            return {"type": "filesystem.exists", "args": {"path": path}}
        if re.search(r"\b(?:size|modified|metadata|information|info)\b", lowered):
            return {"type": "filesystem.info", "args": {"path": path}}
        if re.search(r"\b(?:read|open|summari[sz]e|inspect|show|extract)\b", lowered):
            return {"type": "filesystem.read", "args": {"path": path}}

    directory = _directory_from_message(message)
    if directory:
        return {"type": "filesystem.list", "args": {"path": directory}}

    return None


def _request_needs_multiple_tools(message: str) -> bool:
    """Whether one request explicitly joins two capability domains."""

    text = (message or "").casefold()
    if not re.search(r"\b(?:then|after(?:\s+that)?|compare|using|and\s+then|use)\b", text):
        return False

    domains = 0
    if (
        _path_from_message(text)
        or re.search(
            r"\b(?:file|document|pdf|receipt|spreadsheet|my\s+notes?)\b"
            r"|\bmy\s+(?:policy|contract|report|record|itinerary|schedule|"
            r"budget|invoice)\b",
            text,
        )
    ):
        domains += 1
    if re.search(
        r"\b(?:current|today|latest|web|internet|law|regulation|weather|"
        r"stock|share|exchange|rate|usd|inr|eur|gbp)\b",
        text,
    ):
        domains += 1
    if re.search(r"\b(?:run|execute|code|script|program)\b", text):
        domains += 1

    return domains >= 2


def _artifact_path_from_message(message: str):
    """An exact requested output path for a generated office file."""

    match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/]|\\\\)[^\"\r\n]*?"
        r"\.(?:pptx|docx|xlsx))(?=$|[\s,;.!?])",
        message or "",
        re.IGNORECASE,
    )
    return Path(match.group("path").strip(" \t'\"")) if match else None


def _artifact_extension(message: str):
    text = (message or "").casefold()
    if re.search(r"\b(?:pptx?|powerpoint|presentation|slides?|deck)\b", text):
        return ".pptx"
    if re.search(r"\b(?:docx|word\s+document)\b", text):
        return ".docx"
    if re.search(r"\b(?:xlsx|excel|spreadsheet|workbook)\b", text):
        return ".xlsx"
    return None


def _prepare_generation_step(step: dict, message: str) -> dict:
    """Keep a builder script and its requested artifact distinct."""

    if step.get("type") != "code.generate" or not _wants_built_document(message):
        return step

    extension = _artifact_extension(message)
    if not extension:
        return step

    prepared = copy.deepcopy(step)
    args = prepared.setdefault("args", {})
    requested = _artifact_path_from_message(message)
    proposed = Path(str(args.get("path") or "build_artifact.py"))

    if requested is None and proposed.suffix.casefold() in _ARTIFACT_EXTENSIONS:
        requested = proposed if proposed.is_absolute() else WORKSPACE_DIR / proposed

    if proposed.suffix.casefold() != ".py":
        stem = (requested or proposed).stem or "artifact"
        proposed = WORKSPACE_DIR / f"build_{stem}.py"

    if requested is None:
        stem = re.sub(r"^build[_-]?", "", proposed.stem) or "artifact"
        requested = proposed.with_name(stem + extension)

    proposed = proposed if proposed.is_absolute() else WORKSPACE_DIR / proposed
    requested = requested.resolve()
    proposed = proposed.resolve()

    args["path"] = str(proposed)
    args["artifact_path"] = str(requested)
    args["spec"] = (
        f"{args.get('spec', '').strip()}\n\n"
        f"The finished {_ARTIFACT_EXTENSIONS[extension]} must be written "
        f"to this exact absolute path: {requested}\n"
        "Create its parent directory if needed. Do not save the finished "
        "file under any other name or relative path. Print the exact "
        "finished path after saving it."
    ).strip()
    return prepared


def _completed_artifact_answer(steps: list, results: list):
    """Return an exact success reply for a built office file, or None.

    The builder script and its subprocess already establish whether the file
    exists. Asking a response model to reinterpret that success produced the
    worst possible hand-off: a valid PowerPoint on disk accompanied by "the
    script didn't produce a usable result." The filesystem fact is stronger
    than another model opinion.
    """

    ran_builder = any(
        step.get("type") == "python.run" and result.get("success", True)
        for step, result in zip(steps, results)
    )
    if not ran_builder:
        return None

    for step in reversed(steps):
        if step.get("type") != "code.generate":
            continue
        path = step.get("args", {}).get("artifact_path")
        if not path or not Path(path).is_file():
            continue

        labels = {
            ".pptx": "presentation",
            ".docx": "document",
            ".xlsx": "workbook",
        }
        label = labels.get(Path(path).suffix.casefold(), "file")
        return f"The {label} was saved to {path}.", str(path)

    return None


_FILE_SUMMARY_REQUEST = re.compile(
    r"\b(?:summari[sz](?:e|ing)?|summary|sum\s+(?:it|this)\s+up|"
    r"overview|briefly|action\s+notes)\b",
    re.IGNORECASE,
)
_TRANSACTION_DOCUMENT = re.compile(
    r"\b(?:receipt|invoice|payment|transaction|statement|confirmation|"
    r"membership|booking)\b",
    re.IGNORECASE,
)
_IMPORTANT_IDENTIFIER_LINE = re.compile(
    r"^\s*(?P<label>[^:\r\n]{0,48}\b(?:reference(?:\s+(?:number|code))?|"
    r"ref(?:erence)?(?:\s+(?:number|code))?|transaction\s+(?:id|number|reference)|"
    r"booking\s+(?:id|reference|code)|receipt\s+(?:id|number)|"
    r"invoice\s+(?:id|number)))\s*:\s*(?P<value>[^\r\n]+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _append_exact_fact(answer: str, fact: str) -> str:
    """Append one evidence-derived fact without disturbing existing prose."""

    answer = (answer or "").strip()
    fact = (fact or "").strip()
    if not fact:
        return answer
    if answer and answer[-1] not in ".!?":
        answer += "."
    return f"{answer} {fact}".strip()


def _tabular_largest_row(text: str):
    """Return (label, value, unit) for a simple labelled numeric table."""

    lines = [line.strip() for line in (text or "").splitlines() if "\t" in line]
    if len(lines) < 2:
        return None

    header = [part.strip() for part in lines[0].split("\t")]
    unit_match = re.search(r"\(([^)]+)\)", header[1] if len(header) > 1 else "")
    unit = unit_match.group(1).strip() if unit_match else ""
    rows = []

    for line in lines[1:]:
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 2 or not parts[0]:
            continue
        label = parts[0]
        if label.casefold() in {"total", "subtotal", "grand total", "balance"}:
            continue
        number = re.search(r"-?\d[\d,]*(?:\.\d+)?", parts[1])
        if not number:
            continue
        try:
            value = Decimal(number.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        rows.append((value, label, number.group(0)))

    if not rows:
        return None
    _value, label, displayed = max(rows, key=lambda item: item[0])
    return label, displayed, unit


def _augment_local_file_answer(message: str, answer: str, steps: list, results: list) -> str:
    """Preserve exact high-value fields a grounded draft accidentally omits.

    This is deliberately limited to deterministic structure already present
    in a successfully-read local file. It does not infer a value, ask another
    model, or encode a receipt/template-specific answer.
    """

    documents = [
        result.get("data")
        for step, result in zip(steps, results)
        if step.get("type") == "filesystem.read"
        and result.get("success", True)
        and isinstance(result.get("data"), str)
    ]
    if not documents:
        return answer

    combined = "\n".join(documents)
    revised = answer

    # Brief transaction summaries still need exact identifiers. The model can
    # cite the right line and omit it while composing; append at most two exact
    # labelled values so a useful reference is not silently lost.
    if _FILE_SUMMARY_REQUEST.search(message or "") and _TRANSACTION_DOCUMENT.search(combined):
        appended = 0
        for match in _IMPORTANT_IDENTIFIER_LINE.finditer(combined):
            value = match.group("value").strip().strip(".,;")
            if not value or value.casefold() in revised.casefold():
                continue
            label = re.sub(r"\s+", " ", match.group("label")).strip()
            revised = _append_exact_fact(revised, f"{label}: {value}.")
            appended += 1
            if appended >= 2:
                break

    # A largest/biggest-table request is incomplete if it gives only a number.
    # Use the table's literal label and value rather than trusting another model
    # call to notice the omission.
    if re.search(
        r"\b(?:biggest|largest|highest|maximum)\s+"
        r"(?:expense|cost|amount|item|category)\b",
        message or "",
        re.IGNORECASE,
    ):
        largest = _tabular_largest_row(combined)
        if largest:
            label, value, unit = largest
            if label.casefold() not in revised.casefold():
                amount = f"{unit} {value}".strip()
                revised = _append_exact_fact(
                    revised,
                    f"The biggest expense is {label} at {amount}.",
                )

    return revised


def _structured_result_answer(message: str, step: dict, result: dict):
    """Format exact typed results without asking a model to copy them.

    These services already return labelled fields. A second model call
    can only omit a field or alter a number; it cannot add evidence.
    Mixed plans still use the grounded response model because combining
    sources genuinely requires interpretation.
    """

    if not result.get("success") or not isinstance(result.get("data"), (dict, list, bool)):
        return None

    tool = step.get("type")
    data = result["data"]
    text = (message or "").casefold()

    if tool == "system.datetime" and isinstance(data, dict):
        if data.get("timezone") and data.get("timezone") != "local":
            return (
                f"In {data['timezone']}, it is {data['time']} on "
                f"{data['day_of_week']}, {data['date']}."
            )
        if re.search(r"\byear\b", text) and not re.search(r"\bdate\b", text):
            return f"It is {data['date'][:4]}."
        if re.search(r"\btime\b|\bclock\b", text):
            return f"It is {data['time']} ({data['timezone']})."
        if re.search(r"\bday\b", text) and not re.search(r"\bdate\b", text):
            return f"Today is {data['day_of_week']}."
        return f"Today is {data['day_of_week']}, {data['date']}."

    if tool == "weather.current" and isinstance(data, dict):
        return (
            f"In {data['place']}, it is {data['conditions']} at "
            f"{data['temperature']}{data['temperature_unit']} "
            f"(feels like {data['feels_like']}{data['temperature_unit']}). "
            f"Humidity is {data['humidity']}{data['humidity_unit']}, "
            f"precipitation is {data['precipitation']}{data['precipitation_unit']}, "
            f"and wind is {data['wind_speed']} {data['wind_unit']}. "
            f"Observed at {data['observed_at']} ({data['timezone']})."
        )

    if tool == "finance.quote" and isinstance(data, dict):
        name = data.get("name") or data.get("symbol")
        answer = (
            f"{name} ({data.get('symbol')}) is {data.get('price')} "
            f"{data.get('currency')}."
        )
        if data.get("change") is not None:
            sign = "+" if data["change"] > 0 else ""
            answer += (
                f" Change from the previous close: {sign}{data['change']} "
                f"{data.get('currency')} ({sign}{data.get('change_percent')}%)."
            )
        return answer

    if tool == "finance.exchange" and isinstance(data, dict):
        return (
            f"{data['amount']} {data['base']} is approximately "
            f"{data['converted']} {data['target']} at a rate of "
            f"1 {data['base']} = {data['rate']} {data['target']}. "
            f"Rate date: {data['rates_published']}."
        )

    if tool == "filesystem.exists" and isinstance(data, bool):
        return "Yes, that path exists." if data else "No, that path does not exist."

    if tool == "filesystem.info" and isinstance(data, dict):
        kind = "file" if data.get("is_file") else "directory"
        return (
            f"{data.get('name')} is a {kind} at {data.get('path')}. "
            f"Size: {data.get('size')} bytes. Last modified: "
            f"{data.get('modified_at')}."
        )

    if tool == "filesystem.list" and isinstance(data, list):
        if not data:
            return "That directory is empty."
        lines = [
            f"{item.get('name')} ({item.get('type', 'item')})"
            for item in data if isinstance(item, dict)
        ]
        return "Directory contents:\n" + "\n".join(lines)

    if tool in {"code.run", "python.run"} and isinstance(data, dict):
        # Literal code supplied by the user already has an exact subprocess
        # result. Formatting stdout directly is both faster and safer than a
        # model call that may embellish it or reject its own valid evidence.
        literal_run = bool(
            re.search(
                r"\b(?:run|execute|test)\b[^\r\n]{0,40}"
                r"\b(?:snippet|code|script)\b",
                message or "",
                re.IGNORECASE,
            )
        )
        stdout = str(data.get("stdout") or "").strip()
        if literal_run and data.get("return_code") == 0 and stdout:
            if "\n" in stdout:
                return "The output is:\n" + stdout
            return f"The output is {stdout}."

    return None


def _select_grounded_prompt(steps: list) -> str:
    """
    Pick the narrowest grounded system prompt that fits every step
    type in the plan. Falls back to the full combined prompt if the
    plan mixes categories, so no rule is lost for mixed plans.
    """

    types = {step["type"] for step in steps}

    if types <= {"web.search"}:
        return GROUNDED_WEB_SYSTEM_PROMPT

    if types <= _FILESYSTEM_TYPES:
        return GROUNDED_FILESYSTEM_SYSTEM_PROMPT

    if types <= _CODE_TYPES:
        return GROUNDED_CODE_SYSTEM_PROMPT

    if types <= {"system.datetime"}:
        return GROUNDED_SYSTEM_DATETIME_PROMPT

    if types <= _LIVE_DATA_TYPES:
        return GROUNDED_SYSTEM_DATETIME_PROMPT

    return GROUNDED_SYSTEM_PROMPT


class Agent:

    def __init__(self, mode: str = DEFAULT_MODE, store=None):

        self.state = ConversationState()
        self.mode = mode if mode in MODES else DEFAULT_MODE

        self.store = store or ConversationStore()

        self._build_for_mode(self.mode)

        self._warm_up()

        # A restart should reopen what the user was working on. The
        # store may be a tiny fake in tests, so restoration is optional
        # when that interface is not present.
        recent = (
            self.store.most_recent_id()
            if hasattr(self.store, "most_recent_id") else None
        )
        if recent:
            self.load_conversation(recent)

    def _build_for_mode(self, mode: str):
        """Point every role at the models this mode uses.

        The wrappers hold no state beyond a model name, so switching is
        just rebuilding them - along with the router, planner and
        executor that were handed the old ones.
        """

        config = _mode_config(mode)

        self.planner_model = OllamaModel(config["planner"])
        self.response_model = OllamaModel(config["response"])

        # Fast mode needs a separate vision model. Keep it only briefly
        # after an image instead of pinning two sets of weights in 8 GB
        # VRAM indefinitely. Modes whose vision and response model are
        # the same keep the shared model resident as before.
        vision_keep_alive = (
            -1 if config["vision"] == config["response"] else "2m"
        )
        self.vision_model = OllamaModel(
            config["vision"], keep_alive=vision_keep_alive
        )
        self.code_model = OllamaModel(config["code"])

        self.response_supports_thinking = _supports_thinking(config["response"])
        self.response_style = config.get("style")

        # The accuracy knobs. Read straight off the config so a mode
        # only has to name the ones it changes.
        self.search_results = config["search_results"]
        self.min_sources = config["min_sources"]
        self.force_compute = config["force_compute"]
        self.self_check = config["self_check"]

        self.router = Router(self.planner_model)
        self.planner = Planner(self.planner_model)

        self.executor = ExecutionManager(self.response_model)
        self.executor.code.model = self.code_model
        self.executor.search_results = self.search_results

    def models_in_use(self) -> set:
        """Model names this mode has loaded."""

        return {
            self.planner_model.model_name,
            self.response_model.model_name,
            self.vision_model.model_name,
            self.code_model.model_name,
        }

    def set_mode(self, mode: str) -> str:
        """Switch modes, unloading any model the new one doesn't use.

        Unloading matters more than it looks: two models will not fit
        in this card at once, so leaving the old one resident would
        push the new one onto the CPU and make the switch a downgrade.
        The conversation is deliberately kept - changing how a question
        is answered should not erase what was already said.
        """

        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")

        if mode == self.mode:
            return self.mode

        previous = self.models_in_use()
        previous_mode = self.mode
        runtime_keys = {
            "planner_model", "response_model", "vision_model", "code_model",
            "response_supports_thinking", "response_style", "search_results",
            "min_sources", "force_compute", "self_check", "router", "planner",
            "executor",
        }
        previous_runtime = {
            key: self.__dict__[key] for key in runtime_keys
            if key in self.__dict__
        }

        self._build_for_mode(mode)

        try:
            self._warm_up()
        except Exception:
            attempted = self.models_in_use()
            self.__dict__.update(previous_runtime)
            self.mode = previous_mode

            for name in attempted - previous:
                try:
                    subprocess.run(["ollama", "stop", name], check=False)
                except Exception:
                    pass
            raise

        self.mode = mode

        for name in previous - self.models_in_use():
            try:
                print(f"[MODE] unloading {name}")
                subprocess.run(["ollama", "stop", name], check=False)
            except Exception as error:
                print(f"[MODE] could not unload {name}: {error}")

        print(f"[MODE] now {mode} ({', '.join(sorted(self.models_in_use()))})")

        return self.mode

    def _with_memory(self, system_prompt: str) -> str:
        """Add the notes on earlier messages, if there are any.

        This is what stops trimming from meaning forgetting. The old
        messages are no longer sent, so without this the model would
        genuinely not know your name twenty messages in.
        """

        notes = []

        if self.state.summary:
            notes.append(self.state.summary)

        if self.state.user_profile:
            facts = []
            labels = {
                "name": "The user stated their name is",
                "preferred_name": "The user asked to be called",
                "course": "The user's current course is",
            }
            for key, label in labels.items():
                value = self.state.user_profile.get(key)
                if value:
                    facts.append(f"{label} {value}.")
            if facts:
                notes.append("Explicit user facts:\n" + "\n".join(facts))

        if not notes:
            return system_prompt

        return system_prompt + f"""

----------------------------------------
EARLIER IN THIS CONVERSATION
----------------------------------------

{chr(10).join(notes)}

These are notes on messages that are no longer being sent in full.
Treat them as things the user already told you."""

    def _styled(self, system_prompt: str) -> str:
        """Append the current mode's instructions to a system prompt.

        Appended rather than replacing anything: a mode changes how an
        answer is presented, never the grounding rules that decide what
        may be said at all.
        """

        style = getattr(self, "response_style", None)

        if not style:
            return system_prompt

        return system_prompt + style

    # ------------------------------------------------------------
    # Saving and remembering
    # ------------------------------------------------------------

    def _persist(self):
        """Write the conversation to disk.

        Called after every turn rather than on shutdown. Shutdown is
        the one moment that is not guaranteed to happen - a crash, a
        killed process or a closed laptop all skip it, and those are
        exactly the times losing the conversation would sting.
        """

        if not self.state.messages:
            return

        if not self.state.conversation_id:
            first = next(
                (m.get("content") for m in self.state.messages
                 if m.get("role") == "user"),
                "",
            )
            self.state.conversation_id = self.store.new_id(first)
            self.state.title = title_from(first)

        self.store.save(Conversation(
            id=self.state.conversation_id,
            title=self.state.title or UNTITLED,
            messages=self.state.messages,
            summary=self.state.summary,
            summarized_upto=self.state.summarized_upto,
            last_file_path=self.state.last_file_path,
            last_generated_path=self.state.last_generated_path,
            pending_file_paths=list(self.state.pending_file_paths),
            pending_file_request=self.state.pending_file_request,
            pending_lookup=self.state.pending_lookup,
            user_profile=dict(self.state.user_profile),
            last_capabilities=list(self.state.last_capabilities),
            last_capability_steps=copy.deepcopy(
                self.state.last_capability_steps
            ),
        ))

    def load_conversation(self, conversation_id: str) -> bool:
        """Reopen a saved conversation. False if it is not there."""

        saved = self.store.load(conversation_id)

        if saved is None:
            return False

        self.state = ConversationState(
            messages=list(saved.messages),
            conversation_id=saved.id,
            title=saved.title,
            summary=saved.summary,
            summarized_upto=saved.summarized_upto,
            last_file_path=saved.last_file_path,
            last_generated_path=saved.last_generated_path,
            pending_file_paths=list(saved.pending_file_paths),
            pending_file_request=saved.pending_file_request,
            pending_lookup=saved.pending_lookup,
            user_profile=(
                dict(saved.user_profile)
                if saved.user_profile
                else _profile_from_messages(saved.messages)
            ),
            last_capabilities=list(saved.last_capabilities),
            last_capability_steps=copy.deepcopy(
                saved.last_capability_steps
            ),
        )

        print(f"[STORE] reopened {saved.id} ({len(saved.messages)} messages)")

        return True

    def new_conversation(self):
        """Start fresh, leaving the previous one saved."""

        self.state = ConversationState()
        print("[STORE] started a new conversation")

    def take_back_last_turn(self):
        """Remove the last exchange, returning what was asked.

        For asking again after a bad answer. The point is not that the
        user sees a second attempt - they could retype the question for
        that - but that the first attempt stops existing. A wrong answer
        left in the history is quoted back to the model on every
        following turn as something it already established, and it will
        cheerfully build on it.

        Returns None when there is nothing to take back.
        """

        messages = self.state.messages

        for index in range(len(messages) - 1, -1, -1):

            if messages[index].get("role") != "user":
                continue

            message = messages[index].get("content") or ""

            # Everything from that question onward, which is the
            # question and whatever it produced.
            del messages[index:]

            # The summary may already cover messages that are still
            # here, but it can never cover ones that are not.
            self.state.summarized_upto = min(
                self.state.summarized_upto, len(messages)
            )
            self.state.last_capabilities = []
            self.state.last_capability_steps = []
            self.state.pending_file_paths = []
            self.state.pending_file_request = None
            self.state.pending_lookup = None
            self.state.user_profile = _profile_from_messages(messages)

            print(f"[RETRY] took back the last turn ({len(messages)} left)")

            return message

        return None

    def _maybe_summarize(self):
        """Fold older messages into prose once enough have built up.

        Called by the web layer as idle maintenance after the response
        has been returned. The cost is one model call every
        SUMMARIZE_EVERY messages, never one call per turn.

        A failure here is deliberately silent beyond a log line: the
        conversation is still perfectly usable unsummarised, and there
        is nothing the user could do about it anyway.
        """

        pending = len(self.state.messages) - self.state.summarized_upto

        # KEEP_VERBATIM remains outside the summary, so triggering at
        # SUMMARIZE_EVERY alone used to fold only 16 new messages
        # (24 pending minus the 8-message tail). That made maintenance
        # run every eight user turns. Wait until a full 24 messages can
        # actually be folded, then preserve the recent tail.
        if pending < SUMMARIZE_EVERY + KEEP_VERBATIM:
            return

        # Everything except the recent messages, which stay verbatim.
        cutoff = len(self.state.messages) - KEEP_VERBATIM

        if cutoff <= self.state.summarized_upto:
            return

        fresh = self.state.messages[self.state.summarized_upto:cutoff]

        if not fresh:
            return

        existing = (
            f"NOTES SO FAR\n\n{self.state.summary}\n\n"
            if self.state.summary else ""
        )

        print(f"[SUMMARY] folding messages "
              f"{self.state.summarized_upto}-{cutoff} into notes")

        try:
            summary = self.response_model.complete(
                SUMMARY_SYSTEM_PROMPT,
                f"""{existing}NEW MESSAGES

{_messages_as_text(fresh)}

Write the updated notes.""",
                num_predict=192,
                # qwen3 can spend a small output allowance entirely on
                # hidden reasoning and return no notes. Summarisation is
                # compression, not a reasoning task.
                think=False,
            )

        except Exception as error:
            print(f"[SUMMARY] failed: {error}")
            return

        summary = (summary or "").strip()

        if not summary:
            return

        self.state.summary = summary
        self.state.summarized_upto = cutoff

        print(f"[SUMMARY] now covers {cutoff} messages "
              f"({len(summary)} chars)")

    def maintain_memory(self):
        """Run optional post-response memory maintenance and save it."""

        before = self.state.summarized_upto
        self._maybe_summarize()
        if self.state.summarized_upto != before:
            self._persist()

    def _run_self_check(self, message: str, answer: str, evidence: list) -> str:
        """Re-read the answer against its evidence and hedge what is
        not supported.

        Returns the answer, with a note appended if anything was
        flagged. The answer itself is never edited: the mechanical
        checks have already passed it, and silently rewriting it here
        would discard that.

        Any failure returns the answer untouched. A check that cannot
        run is not a reason to withhold an answer that already passed
        everything else.
        """

        if not answer or not evidence:
            return answer

        PROGRESS.set("Checking its own answer", "verify")

        joined = "\n".join(f"- {line}" for line in evidence)

        try:
            raw = self.response_model.complete(
                SELF_CHECK_SYSTEM_PROMPT,
                f"""QUESTION
{message}

EVIDENCE
{joined}

ANSWER
{answer}""",
                schema=SELF_CHECK_SCHEMA,
                num_predict=RESPONSE_MAX_TOKENS,
            )

            flagged = [
                str(claim).strip()
                for claim in (json.loads(raw).get("unsupported") or [])
                if str(claim).strip()
            ]

        except Exception as error:
            print(f"[SELF CHECK] could not run: {error}")
            return answer

        # A flagged claim has to share real words with the answer it
        # is supposedly quoting. Without this, the check invented "The
        # program printed 6." as an unsupported claim inside an answer
        # that read "The program printed 5." - a number that appears
        # nowhere the check was given, hedging an answer against a
        # sentence Athena never wrote.
        answer_words = set(re.findall(r"[a-z0-9]+", answer.lower()))

        def _grounded_in_answer(claim: str) -> bool:
            claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
            meaningful = {w for w in claim_words if len(w) > 2}
            return bool(meaningful) and meaningful <= answer_words

        # A claim that is nearly the WHOLE answer is not a claim, it is
        # the checker restating what it was given because it could not
        # find anything specific wrong. "The program printed 5." was
        # checked and came back flagged as "The program printed 5.." -
        # the same four words, not a discrepancy. The result was a
        # correct answer followed by a hedge that quoted the answer at
        # itself: "the sources don't clearly back this up: The program
        # printed 5.." A claim has to leave OUT some of the answer to
        # be pointing at a specific part of it.
        def _is_the_whole_answer(claim: str) -> bool:
            claim_words = {w for w in re.findall(r"[a-z0-9]+", claim.lower())
                           if len(w) > 2}
            meaningful_answer = {w for w in answer_words if len(w) > 2}

            if not meaningful_answer:
                return False

            overlap = len(claim_words & meaningful_answer) / len(meaningful_answer)
            return overlap >= 0.85

        real = [c for c in flagged
                if _grounded_in_answer(c) and not _is_the_whole_answer(c)]

        if len(real) < len(flagged):
            print(f"[SELF CHECK] discarded {len(flagged) - len(real)} "
                  f"claim(s) not actually present in the answer")

        flagged = real

        if not flagged:
            print("[SELF CHECK] nothing flagged")
            return answer

        print(f"[SELF CHECK] flagged {len(flagged)}: {flagged}")

        listed = "; ".join(flagged)

        return (
            f"{answer}\n\n"
            f"On a second read, the sources don't clearly back this up: {listed}. "
            "Worth confirming before relying on it."
        )

    def _warm_up(self):
        """Load the selected model, surfacing failure to the caller."""

        self.planner_model.complete("You are a test.", "ping", num_predict=50)

    def _update_file_state(self, step: dict, result: dict, message: str = None):
        """Keep file-selection state in sync with what actually ran.

        Note a search that finds nothing deliberately leaves the
        current selection alone - failing to find Y is not a reason to
        forget that the user was already working with X.
        """

        step_type = step["type"]

        if step_type == "filesystem.read":
            data = result.get("data")
            if (result.get("success") and isinstance(data, str)
                    and data.strip()):
                self.state.last_file_path = step["args"].get("path")
                self.state.pending_file_paths = []
                self.state.pending_file_request = None

        elif step_type == "filesystem.search":

            # A new search replaces any older unanswered choice even
            # when this search finds nothing.
            self.state.pending_file_paths = []
            self.state.pending_file_request = None

            matches = result.get("matches") or []

            if matches:
                # Ambiguous: nothing is selected yet, but remember what
                # was offered so an ordinal reply can resolve it, and
                # what was asked so the reply still knows the job.
                self.state.pending_file_paths = matches
                self.state.pending_file_request = message
                self.state.last_file_path = None

            elif result.get("resolved_path"):
                self.state.last_file_path = result["resolved_path"]

        elif step_type == "filesystem.semantic_search":

            # Same bookkeeping as a filename search, so "read the first
            # one" and "open it" work the same way afterwards. Without
            # this a search that found the right document by meaning
            # would leave nothing selected, and the follow-up would go
            # looking all over again.
            found = [m["path"] for m in (result.get("matches") or [])
                     if m.get("path")]

            if len(found) == 1:
                self.state.last_file_path = found[0]
                self.state.pending_file_paths = []
                self.state.pending_file_request = None

            elif found:
                self.state.pending_file_paths = found
                self.state.pending_file_request = message
                self.state.last_file_path = None

        elif step_type == "code.generate" and result.get("success"):
            # Generated code is remembered, but never promoted to the
            # selected document. Those are different kinds of context:
            # "run that script" should find this path, while "summarize
            # it" after reading a receipt should keep referring to the
            # receipt rather than Athena's scratch Python file.
            data = result.get("data")
            if isinstance(data, dict):
                self.state.last_generated_path = data.get("path") or None

    def respond(self, message: str, image_path: str = None) -> str:
        """Answer transactionally, committing only a completed turn.

        Model wrappers append to the state they receive. Running a turn
        against a copy means Stop, an exception, or a late cancellation
        cannot leave an invisible user message or answer influencing the
        next request.
        """

        original = self.state
        working = copy.deepcopy(original)
        working.last_sources = []
        tool_offset = len(PROGRESS.tools)
        self.state = working

        try:
            answer = self._respond(message, image_path=image_path)

            # A stop may arrive while the final model call is already in
            # flight. No new stage would otherwise observe it.
            PROGRESS.finish_turn()

            working.last_capabilities = list(PROGRESS.tools[tool_offset:])
            self.state = working

            try:
                self._persist()
            except Exception as error:
                print(f"[STORE] could not save the conversation: {error}")

            return answer

        except Exception:
            self.state = original
            raise

    def _respond(self, message: str, image_path: str = None) -> str:

        if _asks_for_legal_guarantee(message):
            reply = (
                "I can't guarantee that a contract is legally compliant, "
                "especially without reviewing it. I can help identify clauses "
                "and compare them with public sources, but a qualified lawyer "
                "must make the final assessment."
            )
            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": reply})
            self.state.last_capabilities = []
            self.state.last_capability_steps = []
            self.state.pending_lookup = None
            print("[SAFETY] legal guarantee refused without a lookup")
            return reply

        if image_path:

            PROGRESS.used("vision")
            PROGRESS.set("Looking at the image")

            try:
                return self.vision_model.chat(
                    self.state,
                    message,
                    system_prompt=IMAGE_SYSTEM_PROMPT,
                    images=[image_path]
                )

            except Exception as error:

                # Most likely the vision model isn't pulled locally.
                # Say so concretely - "something went wrong" would
                # leave the user with no idea that one `ollama pull`
                # fixes it.
                print(f"[VISION ERROR] {error}")

                failure = (
                    f"I couldn't analyse that image - the vision model "
                    f"'{self.vision_model.model_name}' didn't respond. If it "
                    f"isn't installed yet, running "
                    f"'ollama pull {self.vision_model.model_name}' should fix it."
                )

                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append({"role": "assistant", "content": failure})

                return failure

        # Explicit user facts are cheap and safe to keep structurally.
        # This runs before answering so "call me RJ" is already true in
        # the acknowledgement, while the transaction wrapper still
        # rolls it back if the turn is cancelled or fails.
        self.state.user_profile = _update_user_profile(
            self.state.user_profile,
            message,
        )

        remembered = _memory_reply(self.state, message)

        if remembered:
            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": remembered})
            self.state.last_capabilities = []
            self.state.last_capability_steps = []
            self.state.pending_lookup = None
            print("[MEMORY] answered from stored conversation state")
            return remembered

        # -----------------------------
        # Step 0 - Deterministic file selection
        # -----------------------------
        # Which file "the first one" or "summarize it" refers to is
        # state we already hold. Deciding it here rather than routing
        # and planning for it removes two model calls from the loop
        # and, more importantly, removes the chance of the planner
        # re-deriving the wrong path (or none at all) from prose.

        forced_step = None

        # What the grounded model is asked to answer. Normally the
        # message itself, but "the first one" is an answer to Athena's
        # question, not a request in its own right - handing that to a
        # fact-extraction prompt as the thing to answer gave it nothing
        # to look for, and it replied that the information wasn't in
        # the file. The original request is restored here instead.
        grounding_request = message

        if _DELETE_LOCAL_RE.search(message or ""):
            reply = (
                "I can't delete local files. Athena deliberately has no "
                "file-deletion capability, so nothing was changed."
            )
            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": reply})
            self.state.last_capabilities = []
            self.state.last_capability_steps = []
            self.state.pending_file_paths = []
            self.state.pending_file_request = None
            self.state.pending_lookup = None
            return reply

        selected_path = resolve_pending_file_selection(self.state, message)
        direct_step = _direct_step_for(message, self.state)
        pending_route = False

        if self.state.pending_lookup:
            pending_type = self.state.pending_lookup

            if direct_step or names_a_new_subject(message):
                print("[LOOKUP] new subject -> clearing pending clarification")
                self.state.pending_lookup = None
            elif pending_type == "weather.current":
                pending_step = _pending_weather_step(message)
                if pending_step:
                    direct_step = pending_step
                    self.state.pending_lookup = None
                    print(
                        "[LOOKUP] weather clarification resolved -> "
                        f"{pending_step['args']['location']}"
                    )
            elif len((message or "").split()) <= 6:
                # Company and currency names still need planner
                # interpretation, but the capability domain no longer
                # needs to be guessed from the short reply alone.
                pending_route = True
                self.state.pending_lookup = None

        if not direct_step and challenges_last_lookup(self.state, message):
            for previous_step in reversed(self.state.last_capability_steps):
                if previous_step.get("type") in {
                    "weather.current", "finance.quote", "finance.exchange",
                    "web.search", "system.datetime", "code.run",
                }:
                    direct_step = copy.deepcopy(previous_step)
                    print(
                        "[DIRECT] challenge -> repeating "
                        f"{direct_step['type']} with the same arguments"
                    )
                    break

        if (
            selection_is_pending(self.state)
            and not selected_path
            and (names_a_new_subject(message) or direct_step)
        ):
            print("[SELECTION] new topic -> clearing pending file choice")
            self.state.pending_file_paths = []
            self.state.pending_file_request = None

        if selected_path:
            print(f"[SELECTION] pending match resolved -> {selected_path}")

            original_request = getattr(self.state, "pending_file_request", None)

            if original_request:
                grounding_request = original_request
                print(f"[SELECTION] answering the original request: {original_request!r}")

            forced_step = {
                "type": "filesystem.read",
                "args": {"path": selected_path},
            }

        elif direct_step:
            print(f"[DIRECT] {direct_step['type']} selected from explicit request")
            forced_step = direct_step

        elif (
            selection_is_pending(self.state)
            and len(message.split()) <= 5
            and not names_a_new_subject(message)
        ):

            # We asked which file was meant, and this short reply didn't
            # name one. Asking again is the only safe move: routed
            # onward, "frist one" was classified as ordinary chat and
            # answered from nothing at all - a confident summary of room
            # categories, caution deposits and late-fee rules for a
            # document that is a single payment receipt. The chat path
            # has no evidence and no grounding check behind it, so a
            # failed selection must never reach it. Anything longer than
            # a few words is treated as a genuine change of subject and
            # falls through to normal routing.
            print("[SELECTION] pending choice unresolved -> asking again")

            options = "\n".join(self.state.pending_file_paths)

            reply = (
                "I'm not sure which one you meant. Could you pick one of "
                f"these?\n{options}"
            )

            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": reply})

            return reply

        elif is_active_file_request(self.state, message):
            print(f"[SELECTION] active file request -> {self.state.last_file_path}")
            forced_step = {
                "type": "filesystem.read",
                "args": {"path": self.state.last_file_path},
            }

        # -----------------------------
        # Step 1 - Route
        # -----------------------------

        must_calculate = False

        if forced_step or pending_route:
            route = "capability"

        else:
            PROGRESS.set("Working out what you need", "route")
            _t0 = time.perf_counter()
            route = self.router.route(self.state, message)
            print(f"\n[TIMING] router: {time.perf_counter() - _t0:.2f}s")

            if route == "calculate":
                # Carried to the planner rather than discarded - it is
                # the whole reason this didn't go to chat.
                must_calculate = True
                route = "capability"

            elif self.force_compute and route == "chat" and looks_arithmetic(message):
                # The router judged this answerable from memory. In a
                # mode that promises computed answers that is not good
                # enough: a model asked to do arithmetic in its head
                # will produce a confident wrong number, and nothing
                # downstream can tell that from a right one. Overriding
                # here rather than rewording the router prompt because
                # a rule the router can decline to apply is not a
                # guarantee.
                print("[MODE] force_compute -> calculate")
                must_calculate = True
                route = "capability"

            if route == "file":
                # The keyword rules above missed it, but the router
                # recognised the question as being about the open file.
                print(f"[SELECTION] router says active file -> {self.state.last_file_path}")
                forced_step = {
                    "type": "filesystem.read",
                    "args": {"path": self.state.last_file_path},
                }
                route = "capability"

        # A lookup question with nothing to look up. Asked back rather
        # than run at all - regardless of whether the router sent it to
        # chat or to a capability, since both are unsafe here.
        #
        # Chat has no evidence and invents an answer: "what's the
        # weather", after Delhi and Mumbai had been looked up, was
        # answered "overcast, 28.6 degrees, humidity 88%", belonging to
        # nowhere.
        #
        # The capability route is not automatically safer. "Live topic
        # with nothing named" is deliberately left to the router's own
        # classifier rather than forced to chat, and that classifier
        # can decide it needs a lookup anyway - at which point the
        # planner is handed a message with no city in it and infers one
        # from conversation history. Told earlier that the user studies
        # near Rupnagar, it answered for that city - confidently
        # reporting the weather for a place that was
        # never asked about, without a word of it being a guess. Only
        # gating this on route == "chat" caught the first failure and
        # missed the second, because both come from the same gap: the
        # message names no city, whichever way it was classified.
        if not forced_step:

            missing = missing_subject_question(message)

            if missing:
                print(f"[ROUTER] lookup with nothing named ({route}) -> asking back")

                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append(
                    {"role": "assistant", "content": missing}
                )
                self.state.last_capabilities = []
                self.state.last_capability_steps = []
                self.state.pending_lookup = missing_subject_capability(message)

                return missing

        if route == "chat":

            # A complete new chat turn supersedes an unanswered live
            # clarification. Without this, "tell me a joke" after
            # "which city?" could make a later bare noun look like a
            # city several turns after the topic had changed.
            self.state.pending_lookup = None

            PROGRESS.set("Thinking", "compose")

            if _asks_about_public_instructions(message):
                # The static prompt is public repository content; answer
                # with the configured response model. Use complete() rather
                # than chat() so rolling memory, prior messages and private
                # document evidence are not included in this transparency
                # request.
                answer = self.response_model.complete(
                    self._styled(CHAT_SYSTEM_PROMPT),
                    f"""Answer this question about Athena's public static
instructions using only the system instructions supplied with this call:

{message}

Do not mention or infer any user identity, conversation history, rolling
memory, local document, credential, file path or previous tool result.
If asked for the prompt text, it is public and may be quoted.""",
                    num_predict=RESPONSE_MAX_TOKENS,
                    think=False,
                )
                answer = _strip_markdown(_strip_tags(answer), flatten=False)
                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append({"role": "assistant", "content": answer})
                self.state.last_capabilities = []
                self.state.last_capability_steps = []
                print("[CHAT] public prompt answered without private runtime context")
                return answer

            grounded_chat_message = f"""{message}

(Background context only, not something to respond to unless relevant: {_grounding_line()})"""

            _t1 = time.perf_counter()
            answer = self.response_model.chat(
                self.state,
                grounded_chat_message,
                system_prompt=self._with_memory(self._styled(CHAT_SYSTEM_PROMPT))
            )
            print(f"[TIMING] chat response: {time.perf_counter() - _t1:.2f}s")
            

            # chat() appended both messages itself. The user's is
            # replaced because what went to the model had the grounding
            # note stapled to it, and the history should hold what was
            # actually typed.
            self.state.messages[-2] = {"role": "user", "content": message}

            # Markdown is stripped here as well as on the grounded
            # path. The page puts replies on screen as text, not HTML,
            # so asterisks arrive as asterisks - asked what 2+2 is,
            # Athena answered "* **Rule:** This is a basic addition
            # problem", visibly.
            #
            # Not flattened, unlike a grounded answer: a chat reply has
            # real paragraphs, and collapsing them into one block turns
            # a readable explanation into a wall.
            answer = _strip_markdown(_strip_tags(answer), flatten=False)
            answer = _remove_unnecessary_capitulation(message, answer)

            if _avoidable_typo_clarification(message, answer):
                # The request is specific but the draft treated a close
                # spelling error as missing information. Retry only this
                # rare shape, without Max's extra explanation style; a
                # blanket second response call would double ordinary
                # chat latency and usually repeat the same answer.
                PROGRESS.set("Re-reading the typo", "compose")
                try:
                    revised = self.response_model.complete(
                        # Deliberately omit rolling notes here. This is
                        # a fresh reading of a standalone typo, and an
                        # old assistant answer in a summary must not
                        # make a settled question appear disputed.
                        CHAT_SYSTEM_PROMPT,
                        f"""User request:

{message}

Draft reply:

{answer}

The first word appears misspelled and the request may still have a clear,
ordinary meaning. Re-read it typo-tolerantly. Answer the likely request
directly if it is clear; otherwise keep the clarification. Return only the
reply.""",
                        num_predict=RESPONSE_MAX_TOKENS,
                        think=False,
                    )
                    revised = _strip_markdown(
                        _strip_tags(revised), flatten=False
                    )

                    # Some models correctly decipher the typo but echo
                    # the corrected question instead of answering it.
                    # One final standalone call is cheaper and cleaner
                    # than launching the planner and web search for an
                    # ordinary knowledge question. This is general typo
                    # handling; no subject-specific answer is encoded.
                    if (
                        revised
                        and revised.rstrip().endswith("?")
                        and _avoidable_typo_clarification(message, revised)
                    ):
                        revised = self.response_model.complete(
                            CHAT_SYSTEM_PROMPT,
                            f"""Answer this corrected user request directly:

{revised}

Do not repeat or rephrase the question. Do not ask a question.
Return only the answer.""",
                            num_predict=RESPONSE_MAX_TOKENS,
                            think=False,
                        )
                        revised = _strip_markdown(
                            _strip_tags(revised), flatten=False
                        )

                    if revised and not _avoidable_typo_clarification(message, revised):
                        answer = revised
                        print("[CHAT REPAIR] replaced an avoidable typo clarification")
                except Exception as error:
                    print(f"[CHAT REPAIR] retry failed: {error}")

            if _avoidable_typo_clarification(message, answer):
                # The fresh retry still could not answer a fully formed
                # typo-heavy factual question. Remove the unhelpful
                # draft that chat() appended and continue through the
                # ordinary planner/evidence path. Search engines are
                # typo tolerant; returning evidence is safer than
                # persisting uncertainty or an echoed question.
                del self.state.messages[-2:]
                self.state.last_capabilities = []
                self.state.last_capability_steps = []
                route = "capability"
                print("[CHAT REPAIR] unresolved typo -> capability lookup")

            else:
                # The reply is capped in the history for the same reason
                # capability answers are - it is resent on every following
                # turn - but far less harshly, since this is Athena's own
                # prose and the conversation is built on it.
                self.state.messages[-1] = {
                    "role": "assistant",
                    "content": answer,
                }

                # Nothing ran, so there is nothing for a follow-up to
                # continue. Left stale, a weather lookup from two turns ago
                # would keep pulling ordinary conversation back to it.
                self.state.last_capabilities = []
                self.state.last_capability_steps = []

                return answer

        # -----------------------------
        # Step 2 - Sequential plan + execute
        # -----------------------------

        executed = []
        steps = []
        results = []
        seen = set()

        # Step types that produce a complete, self-contained result and
        # do not need another planning call for a single-job request.
        # A web.search step can already carry multiple query variants;
        # a genuinely mixed request may still continue into a different
        # capability because the condition below checks for that.
        # code.generate is deliberately absent. Asked for a PowerPoint,
        # the planner correctly wrote a script that builds one - and the
        # loop then stopped, because generating code counted as a
        # finished job. The script was never run, so no presentation
        # existed. Writing code is sometimes the whole request and
        # sometimes only the means to a file, and which one it is comes
        # from the request rather than the step type, so the planner is
        # asked instead of being cut off.
        ONE_SHOT_TYPES = {
            "system.datetime", "filesystem.read", "filesystem.list",
            "filesystem.exists", "filesystem.info", "filesystem.search",
            "python.run", "code.run", "web.search",
        }

        if forced_step:

            forced_step = _prepare_search_step(forced_step, grounding_request)

            PROGRESS.used(forced_step["type"])
            PROGRESS.set(_STEP_STAGES.get(forced_step["type"], "Working"))

            _t2 = time.perf_counter()
            result = self.executor.execute({"steps": [forced_step]})[0]
            print(f"[TIMING] step ({forced_step['type']}): {time.perf_counter() - _t2:.2f}s")

            steps.append(forced_step)
            results.append(result)
            executed.append({"step": forced_step, "result": result})
            seen.add((
                forced_step["type"],
                json.dumps(forced_step.get("args", {}), sort_keys=True, default=str),
            ))

            self._update_file_state(forced_step, result, grounding_request)

        continue_after_forced = bool(
            forced_step
            and results[-1].get("success", True)
            and _request_needs_multiple_tools(grounding_request)
        )
        remaining_steps = (
            max(0, MAX_PLAN_STEPS - len(executed))
            if (not forced_step or continue_after_forced)
            else 0
        )

        for _ in range(remaining_steps):

            PROGRESS.set("Planning" if not executed else "Deciding what's next", "plan")

            decision = self.planner.plan_step(
                self.state, grounding_request, executed,
                must_calculate=must_calculate,
            )

            if decision.get("error"):
                # A planner failure is not the same as the planner
                # deciding it has enough - don't let it masquerade as a
                # finished plan.
                print(f"[PLANNER] step planning failed: {decision['error']}")
                break

            if decision.get("done"):
                break

            step = decision.get("step")

            if not step:
                break

            step = _prepare_generation_step(step, grounding_request)
            step = _prepare_search_step(step, grounding_request)

            if must_calculate and step.get("type") == "code.run":
                proportional_step = _proportional_scaling_step(
                    self.state,
                    grounding_request,
                )
                if proportional_step:
                    step = proportional_step
                    print(
                        "[CALCULATION] preserved the source total while "
                        "scaling the group size"
                    )

            if (
                must_calculate
                and step.get("type") == "code.run"
                and step.get("args", {}).get("code")
                and getattr(
                    getattr(self.executor, "code", None),
                    "repair_generated_syntax",
                    None,
                )
            ):
                step["args"]["code"] = self.executor.code.repair_generated_syntax(
                    step["args"]["code"]
                )

            fingerprint = (
                step["type"],
                json.dumps(
                    step.get("args", {}),
                    sort_keys=True,
                    default=str,
                ),
            )

            if fingerprint in seen:
                print(f"[PLANNER] duplicate step blocked: {step}")
                break

            # A read with no path to read. The planner emitted
            # {"type": "filesystem.read", "args": {}} straight after a
            # successful search, and the resulting "Path is not a file."
            # went on to become the whole answer. Where a file is
            # already selected the intent is obvious and the path is
            # filled in; where none is, the step is dropped rather than
            # run on an empty string.
            if step["type"] == "filesystem.read" and not step.get("args", {}).get("path"):

                known = self.state.last_file_path

                if known:
                    print(f"[PLANNER] read with no path -> using {known}")
                    step.setdefault("args", {})["path"] = known
                else:
                    print("[PLANNER] read with no path and none open -> dropped")
                    break

            # A search step already accepts a list of query variants.
            # A second web.search in the same turn therefore repeats
            # the retrieval stage rather than adding a new capability.
            # Do not stop a mixed plan after its first search, though:
            # it may still continue into a file read, calculation, etc.
            if (
                step["type"] == "web.search"
                and any(
                    item.get("step", {}).get("type") == "web.search"
                    for item in executed
                )
            ):
                print(f"[PLANNER] repeated web search blocked: {step}")
                break

            seen.add(fingerprint)

            PROGRESS.used(step["type"])
            PROGRESS.set(_STEP_STAGES.get(step["type"], "Working"))

            _t3 = time.perf_counter()
            result = self.executor.execute({"steps": [step]})[0]
            print(f"[TIMING] step ({step['type']}): {time.perf_counter() - _t3:.2f}s")

            if step["type"] in {"code.run", "python.run"}:
                execution_error = _script_error(result)

                # Calculations are generated as disposable in-memory
                # snippets. If one crashes, prints nothing, or reports
                # that its own verification failed, give the code model
                # exactly one chance to repair it. Literal code supplied
                # by the user does not set must_calculate and is never
                # silently rewritten.
                if (
                    execution_error
                    and must_calculate
                    and step["type"] == "code.run"
                    and step.get("args", {}).get("code")
                    and getattr(getattr(self.executor, "code", None),
                                "repair_snippet", None)
                ):
                    print(
                        "[REPAIR] calculation failed -> "
                        f"{execution_error.splitlines()[-1][:90]}"
                    )
                    PROGRESS.set("Fixing the calculation")

                    try:
                        repaired_code = self.executor.code.repair_snippet(
                            grounding_request,
                            step["args"]["code"],
                            execution_error,
                        )
                    except Exception as repair_error:
                        print(f"[REPAIR] calculation repair failed: {repair_error}")
                        repaired_code = ""

                    if repaired_code and repaired_code.strip():
                        repaired_step = copy.deepcopy(step)
                        repaired_step["args"]["code"] = repaired_code.strip()
                        _t_retry = time.perf_counter()
                        repaired_result = self.executor.execute(
                            {"steps": [repaired_step]}
                        )[0]
                        print(
                            "[TIMING] repaired step (code.run): "
                            f"{time.perf_counter() - _t_retry:.2f}s"
                        )
                        step = repaired_step
                        result = repaired_result
                        execution_error = _script_error(result)

                if execution_error:
                    result = {
                        "success": False,
                        "error": execution_error,
                        "data": result.get("data"),
                    }

            executed.append({"step": step, "result": result})
            steps.append(step)
            results.append(result)

            self._update_file_state(step, result, grounding_request)

            if not result.get("success", True):
                break

            # A script written to build a document gets run, without
            # asking the planner whether to.
            #
            # Asked for a PowerPoint, it wrote a correct python-pptx
            # script and then answered "done" - leaving the user a .py
            # file and no presentation. Told about both cases by
            # example, it still stopped: "write me a script" and "make
            # me a deck" both end in code.generate, and only the request
            # says which was wanted. That is already known here, so it
            # is decided rather than asked.
            if step["type"] == "code.generate" and _wants_built_document(grounding_request):

                built = _generated_script_path(step, result)

                if built:
                    print(f"[BUILD] document requested -> running {built}")
                    PROGRESS.used("python.run")
                    PROGRESS.set("Building the file")

                    run_step = {"type": "python.run", "args": {"path": built}}

                    _t_run = time.perf_counter()
                    run_result = self.executor.execute({"steps": [run_step]})[0]
                    print(f"[TIMING] step (python.run): {time.perf_counter() - _t_run:.2f}s")

                    # A script that crashed gets one repair attempt with
                    # the error shown to the model that wrote it.
                    #
                    # Generated code fails in ways it can obviously fix
                    # once it sees the traceback - a bad character range
                    # in a regex, a name used before assignment - but
                    # nothing was reading the traceback, so the whole
                    # request came back as "I couldn't find that" while
                    # the actual fault sat in stderr. One retry only: if
                    # the second attempt fails too, the error is not the
                    # kind that reading it solves.
                    repair_attempts_used = 0
                    for repair_number in range(1, CODE_REPAIR_ATTEMPTS + 1):
                        error_text = _script_error(run_result)
                        if not error_text and not run_result.get("success", True):
                            error_text = str(
                                run_result.get("error") or "The script failed."
                            )
                        if not error_text:
                            break

                        repair_attempts_used = repair_number

                        print(
                            f"[REPAIR {repair_number}/{CODE_REPAIR_ATTEMPTS}] "
                            f"script failed -> {error_text.splitlines()[-1][:90]}"
                        )
                        PROGRESS.used("code.generate")
                        PROGRESS.set("Fixing the script")

                        repair = {
                            "type": "code.generate",
                            "args": {
                                "path": step["args"].get("path"),
                                "spec": (
                                    f"{step['args'].get('spec', '')}\n\n"
                                    f"Repair attempt {repair_number}. The current "
                                    f"program failed with this error. Correct the "
                                    f"specific failing line and return the whole "
                                    f"working program:\n{error_text}"
                                ),
                                "overwrite": True,
                            },
                        }

                        repair_result = self.executor.execute({"steps": [repair]})[0]
                        if not repair_result.get("success", True):
                            run_result = repair_result
                            break

                        run_result = self.executor.execute({"steps": [run_step]})[0]
                        print(
                            f"[REPAIR {repair_number}/{CODE_REPAIR_ATTEMPTS}] "
                            "re-ran the corrected script"
                        )

                    expected_artifact = step.get("args", {}).get("artifact_path")
                    if (
                        not _script_error(run_result)
                        and expected_artifact
                        and not Path(expected_artifact).is_file()
                    ):
                        run_result = {
                            "success": False,
                            "error": (
                                "The build script ran, but it did not create "
                                f"the requested file at '{expected_artifact}'."
                            ),
                            "data": run_result.get("data"),
                        }

                    if (
                        expected_artifact
                        and run_result.get("success", True)
                        and Path(expected_artifact).is_file()
                    ):
                        self.state.last_file_path = expected_artifact
                        if isinstance(run_result.get("data"), dict):
                            run_result["data"]["artifact_path"] = expected_artifact

                    final_build_error = _script_error(run_result)
                    if not final_build_error and not run_result.get("success", True):
                        final_build_error = str(
                            run_result.get("error") or "The build script failed."
                        )
                    if final_build_error:
                        # A builder script is only an implementation
                        # detail. Saving that script is not partial
                        # success when the requested presentation,
                        # document or workbook was not created.
                        if str(final_build_error).startswith(
                            "The build script ran, but it did not create"
                        ):
                            safe_build_error = final_build_error
                        elif repair_attempts_used:
                            safe_build_error = (
                                "I couldn't create the requested file because "
                                "the generated build script still failed after "
                                f"{repair_attempts_used} repair attempts."
                            )
                        else:
                            safe_build_error = (
                                "I couldn't create the requested file because "
                                "the generated build script failed."
                            )
                        run_result = {
                            "success": False,
                            "error": safe_build_error,
                            "data": run_result.get("data"),
                        }
                        results[-1] = {
                            "success": False,
                            "error": safe_build_error,
                            "data": result.get("data"),
                        }

                    executed.append({"step": run_step, "result": run_result})
                    steps.append(run_step)
                    results.append(run_result)
                    break

            if (
                (
                    step["type"] in ONE_SHOT_TYPES
                    or (
                        step["type"] == "web.search"
                        and _asks_latest_stable_version(grounding_request)
                    )
                )
                and not _request_needs_multiple_tools(grounding_request)
            ):
                # No planner "done" call needed. One web.search step
                # can already carry several queries, so repeating the
                # same capability consumes time and prompt space rather
                # than adding a genuinely different capability.
                break

        plan = {"steps": steps}

        print("\nPLAN:")
        print(plan)

        if not steps:
            # Nothing ran, so there is no evidence to ground anything
            # in. Say that plainly rather than running the grounded
            # prompt over an empty result set, which would surface as a
            # misleading "I couldn't find that in what I looked up."
            fallback = (
                "I wasn't able to work out how to handle that one. "
                "Could you rephrase it?"
            )

            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": fallback})

            return fallback

        # -----------------------------
        # Step 3 - Check errors
        # -----------------------------

        # A failure only ends the turn when nothing else worked.
        #
        # This used to return the first error unconditionally, throwing
        # away every step that had succeeded. Asked which file held a
        # hostel payment, semantic search found it - and the planner
        # then added a filesystem.read with no path at all, which
        # failed, and "Path is not a file." became the entire answer.
        # The document had been found and read; the reply mentioned
        # neither.
        #
        # Steps that failed are dropped rather than described. A partial
        # answer from the evidence that exists is more use than an error
        # about the step that did not contribute any.
        failed = [r for r in results if not r.get("success", True)]

        if failed:

            kept = [(s, r) for s, r in zip(steps, results)
                    if r.get("success", True)]

            if not kept:
                error_message = failed[0]["error"]

                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

                return error_message

            print(f"[PLAN] {len(failed)} step(s) failed, "
                  f"answering from the {len(kept)} that worked")

            steps = [s for s, _ in kept]
            results = [r for _, r in kept]
            plan["steps"] = steps

        # "Latest stable version" is a generic ordering problem once an
        # official release page has supplied semantic versions. Resolve
        # it deterministically instead of asking a model to decide
        # whether a newer security release of an older branch outranks
        # the current branch.
        if (
            len(steps) == 1
            and steps[0].get("type") == "web.search"
            and len(results) == 1
        ):
            version_fact = _latest_stable_version_answer(
                grounding_request, results
            )
            if version_fact:
                answer = version_fact["answer"]
                self.state.last_sources = _source_records(
                    [version_fact["url"]], results
                )
                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                self.state.last_capabilities = ["web.search"]
                self.state.last_capability_steps = copy.deepcopy(steps)
                print(
                    "[ANSWER] selected the highest stable version from "
                    "official release evidence"
                )
                print(f"[STATE] messages now: {len(self.state.messages)} total")
                return answer

        completed_artifact = _completed_artifact_answer(steps, results)
        if completed_artifact:
            answer, artifact_path = completed_artifact
            self.state.last_sources = _source_records([artifact_path], results)
            self.state.messages.append({"role": "user", "content": message})
            self.state.messages.append({"role": "assistant", "content": answer})
            self.state.last_capabilities = [step["type"] for step in steps]
            self.state.last_capability_steps = copy.deepcopy(steps)
            print("[ANSWER] confirmed the generated artifact on disk")
            print(f"[STATE] messages now: {len(self.state.messages)} total")
            return answer

        # A single structured service result is already the final fact.
        # Formatting it in code saves a model call and prevents a model
        # from dropping conditions, changing a number, or replying with
        # Python's bare True/False representation.
        if len(steps) == 1 and len(results) == 1:
            direct_answer = _structured_result_answer(
                grounding_request, steps[0], results[0]
            )
            if direct_answer:
                source_by_tool = {
                    "weather.current": "https://open-meteo.com/",
                    "finance.quote": "https://finance.yahoo.com/",
                    "finance.exchange": "https://www.frankfurter.app/",
                }
                origin = source_by_tool.get(steps[0]["type"])
                if not origin and steps[0]["type"].startswith("filesystem."):
                    origin = steps[0].get("args", {}).get("path")

                self.state.last_sources = (
                    _source_records([origin], results) if origin else []
                )
                for source in self.state.last_sources:
                    if source.get("kind") == "web":
                        source["trusted"] = True

                self.state.messages.append({"role": "user", "content": message})
                self.state.messages.append(
                    {"role": "assistant", "content": direct_answer}
                )
                self.state.last_capabilities = [steps[0]["type"]]
                self.state.last_capability_steps = copy.deepcopy(steps)
                print("[ANSWER] formatted directly from structured data")
                print(f"[STATE] messages now: {len(self.state.messages)} total")
                return direct_answer

        # -----------------------------
        # Step 4 - Evidence-anchored final response
        # -----------------------------
        
        prompt, sentence_map, sentence_origin = PromptBuilder.build(
            self.state,
            grounding_request,
            plan,
            results
        )


        grounded_message = f"""{_grounding_line()}
Current User Request:

{grounding_request}

{prompt}
"""

        grounded_system_prompt = _select_grounded_prompt(plan["steps"])
        has_web_steps = any(
            step.get("type") == "web.search" for step in plan["steps"]
        )

        PROGRESS.set("Composing the answer", "compose")

        _t4 = time.perf_counter()
        raw_response = self.response_model.complete(
            self._styled(grounded_system_prompt),
            grounded_message,
            schema=(
                _grounded_schema_for(grounding_request)
                if has_web_steps else GROUNDED_SCHEMA
            ),
            num_predict=RESPONSE_MAX_TOKENS,
            think=getattr(self, "response_supports_thinking", False)
        )
        print(f"[TIMING] grounded response: {time.perf_counter() - _t4:.2f}s")

        raw_response = re.sub(
            r"<think>.*?</think>", "", raw_response,
            flags=re.DOTALL,
        ).strip()

        print("\n[GROUNDED RAW]:", repr(raw_response))
        try:
            parsed = json.loads(raw_response)
            evidence = parsed.get("evidence", [])
            answer = parsed.get("answer", "").strip()

            if isinstance(evidence, str):
                evidence = [evidence] if evidence.strip() else []

        except Exception:
            evidence = []
            answer = ""

        PROGRESS.set("Checking the sources", "verify")

        source_text = _normalize_ws(prompt)

        # Resolve evidence IDs the same way for every plan. Provenance,
        # not which prompt object happened to be selected, determines
        # whether an item is web or local evidence. This is essential
        # for plans that compare a private file with a public source.
        verified_evidence = []
        verified_origins = []
        seen_evidence = set()

        for item in evidence:
            evidence_ids = re.findall(r"S\d+", str(item))

            if evidence_ids:
                for sid in evidence_ids:
                    text = sentence_map.get(sid)
                    origin = sentence_origin.get(sid)
                    key = (text, origin)
                    if text and key not in seen_evidence:
                        seen_evidence.add(key)
                        verified_evidence.append(text)
                        verified_origins.append(origin)
                continue

            # Backward-compatible exact-copy validation for a tool
            # result that has not yet been sentence-tagged.
            sub_sentences = re.split(r'(?<=[.!?])\s+', str(item).strip())
            kept = [
                sentence for sentence in sub_sentences
                if (_normalize_ws(sentence)
                    and _normalize_ws(sentence) in source_text)
            ]
            if kept:
                text = " ".join(kept)
                key = (text, None)
                if key not in seen_evidence:
                    seen_evidence.add(key)
                    verified_evidence.append(text)
                    verified_origins.append(None)

        if len(verified_evidence) < len(evidence):
            print(
                f"[GROUNDING] {len(evidence)} evidence ref(s) resolved "
                f"to {len(verified_evidence)} valid item(s)"
            )

        evidence = verified_evidence

        search_urls = {
            str(page.get("url"))
            for result in results
            for page in (
                result.get("data") if isinstance(result.get("data"), list)
                else []
            )
            if isinstance(page, dict) and page.get("url")
        }
        web_pairs = [
            (text, origin)
            for text, origin in zip(evidence, verified_origins)
            if str(origin) in search_urls
        ]
        web_evidence = [text for text, _ in web_pairs]
        local_document = [
            text for sid, text in sentence_map.items()
            if str(sentence_origin.get(sid)) not in search_urls
        ]

        min_corroboration = None
        web_relevant = True
        web_answer_evidence = list(web_evidence)
        web_answer_origins = []
        outcome_fallback_evidence = []

        if has_web_steps:
            queries = _search_queries_from(plan["steps"], results)
            if _OUTCOME_QUESTION_RE.search(grounding_request or ""):
                outcome_fallback_evidence = _outcome_web_evidence(
                    sentence_map,
                    sentence_origin,
                    search_urls,
                    results,
                    queries,
                )
            web_relevant = bool(web_evidence) and _evidence_is_relevant(
                web_evidence, queries
            )

            # A model can cite the wrong IDs even when retrieval found a
            # concise, directly relevant sentence. Recover only the
            # highest-scoring on-topic sentence(s) from the tagged web
            # evidence. This keeps the fallback deterministic and
            # evidence-bound rather than accepting the model's prose or
            # unrelated snippets that happen to mention one place name.
            if not web_relevant:
                ranked = []
                for sid, text in sentence_map.items():
                    origin = sentence_origin.get(sid)
                    if str(origin) not in search_urls:
                        continue
                    score = _evidence_relevance_score([text], queries)
                    if score:
                        ranked.append((score, text, origin))

                if ranked:
                    best = max(item[0] for item in ranked)
                    recovered = []
                    recovered_seen = set()
                    for score, text, origin in ranked:
                        key = (text, origin)
                        if score == best and key not in recovered_seen:
                            recovered_seen.add(key)
                            recovered.append((text, origin))
                        if len(recovered) >= 2:
                            break
                    web_pairs = recovered
                    web_evidence = [text for text, _ in recovered]
                    web_answer_evidence = list(web_evidence)
                    web_relevant = True
                    print(
                        "[GROUNDING] recovered on-topic web evidence "
                        "after invalid citations"
                    )

            if web_evidence and web_relevant:
                counts = [
                    1 + _count_corroborating_sources(text, results, origin)
                    for text, origin in web_pairs
                ]
                min_corroboration = min(counts)
                print(
                    "[FACT CHECK] minimum corroboration across evidence: "
                    f"{min_corroboration} source(s)"
                )
            elif web_evidence:
                print("[GROUNDING] cited web evidence is off-topic")

            # Relevant citations can still be incomplete. A model once
            # cited a sentence saying Starmer's term had ended while the
            # next tagged result sentence named Burnham as his successor.
            # The citations were on-topic, so the invalid-citation recovery
            # above did not run; the direct answer was then discarded even
            # though retrieval did contain every claimed fact.
            #
            # Recover from other tagged web sentences only when the whole
            # composed answer verifies against them. Candidate lines must
            # either match the search subject or share at least two concrete
            # answer terms. An invented number or name absent from retrieval
            # therefore remains impossible to admit.
            if (
                answer
                and web_relevant
                and not _answer_within_evidence(answer, web_evidence)
            ):
                answer_terms = {
                    term.casefold() for term in _key_terms(answer)
                }
                expanded_pairs = list(web_pairs)
                expanded_seen = set(expanded_pairs)

                for sid, text in sentence_map.items():
                    origin = sentence_origin.get(sid)
                    pair = (text, origin)
                    if str(origin) not in search_urls or pair in expanded_seen:
                        continue

                    concrete = {
                        term.casefold() for term in _key_terms(text)
                    }
                    if (
                        _evidence_relevance_score([text], queries) > 0
                        or len(answer_terms & concrete) >= 2
                    ):
                        expanded_seen.add(pair)
                        expanded_pairs.append(pair)

                expanded_evidence = [text for text, _ in expanded_pairs]
                if (
                    len(expanded_pairs) > len(web_pairs)
                    and _answer_within_evidence(answer, expanded_evidence)
                ):
                    web_answer_evidence = expanded_evidence
                    web_answer_origins = [
                        origin for _, origin in expanded_pairs if origin
                    ]
                    print(
                        "[GROUNDING] recovered omitted supporting web "
                        "sentences from the retrieved results"
                    )

        # Web claims may use cited evidence plus any directly relevant
        # tagged sentence recovered above. Local document claims may use
        # the whole extracted document because summaries naturally draw
        # on more lines than they cite.
        safe_web = web_answer_evidence if web_relevant else []
        non_web_citations = [
            text for text, origin in zip(evidence, verified_origins)
            if str(origin) not in search_urls
        ]
        support_pool = safe_web + local_document + non_web_citations
        answer_verified_against_document = bool(
            answer and support_pool
            and _answer_within_evidence(answer, support_pool)
        )

        # Relevance alone is not completeness. A fully supported sentence
        # about the tournament format still does not answer "who won?".
        # When retrieval contains an explicit, authority-ranked outcome and
        # the composed answer contains none, replace it with that verified
        # result even though the background sentence was technically grounded.
        if (
            answer_verified_against_document
            and outcome_fallback_evidence
            and _outcome_answer_missing(grounding_request, answer)
        ):
            direct_outcome = _concise_web_fallback(
                outcome_fallback_evidence,
                queries,
            )
            if direct_outcome:
                print(
                    "[GROUNDING] grounded draft omitted the requested outcome "
                    "-> using direct result evidence"
                )
                answer = direct_outcome
                evidence = list(outcome_fallback_evidence)
                safe_web = list(outcome_fallback_evidence)
                outcome_keys = {
                    _normalize_ws(text) for text in outcome_fallback_evidence
                }
                web_pairs = [
                    (text, sentence_origin.get(sid))
                    for sid, text in sentence_map.items()
                    if _normalize_ws(text) in outcome_keys
                    and str(sentence_origin.get(sid)) in search_urls
                ]
                web_answer_origins = [
                    origin for _, origin in web_pairs if origin
                ]
                answer_verified_against_document = True

        if answer_verified_against_document:
            print("[GROUNDING] composed answer verified against typed evidence")
        elif evidence:
            fallback = safe_web + non_web_citations
            if fallback:
                pure_web_answer = bool(
                    safe_web and not non_web_citations and not local_document
                )

                if pure_web_answer:
                    supported_subset = _supported_answer_subset(
                        answer, support_pool
                    )

                    # An answer to "who won?" is not useful if strict
                    # sentence filtering leaves only tournament background.
                    # Prefer tagged, authority-ranked result evidence that
                    # directly states the outcome. This is still a verbatim
                    # evidence fallback, never permission to trust model prose.
                    if (
                        outcome_fallback_evidence
                        and not _contains_outcome_relationship(supported_subset)
                    ):
                        supported_subset = ""
                        print(
                            "[GROUNDING] supported subset omitted the outcome "
                            "-> using direct result evidence"
                        )

                    if supported_subset:
                        print(
                            "[GROUNDING] keeping supported sentences from "
                            "the composed answer"
                        )
                        answer = supported_subset
                    else:
                        print("[GROUNDING] using concise verified web evidence")
                        answer = _concise_web_fallback(
                            outcome_fallback_evidence or safe_web,
                            queries,
                        )
                else:
                    print("[GROUNDING] composed answer unsupported -> using verified evidence")
                    answer = (
                        "I couldn't put that together into a reliable answer, "
                        "so here is what the verified evidence says: "
                        + " ".join(fallback)
                    )
                answer_verified_against_document = True
                evidence = fallback
            else:
                answer = ""
                evidence = []
        else:
            print("[GROUNDING] answer has no verified support")
            answer = ""

        if (
            safe_web
            and min_corroboration is not None
            and min_corroboration < self.min_sources
            and answer
        ):
            answer += (
                f" - though this only turned up in {min_corroboration} "
                f"independent source{'' if min_corroboration == 1 else 's'}, "
                "so it's worth double-checking."
            )

        cited_origins = [
            origin for origin in verified_origins
            if origin and str(origin) not in search_urls
        ] + [
            origin for _, origin in web_pairs if origin and web_relevant
        ] + web_answer_origins
        if answer_verified_against_document and not cited_origins:
            cited_origins = [
                origin for origin in sentence_origin.values() if origin
            ]
        self.state.last_sources = _source_records(cited_origins, results)

        if not answer or (not evidence and not answer_verified_against_document):

            # Worded for what actually ran. "I couldn't find that in
            # what I looked up" is the right sentence for a failed
            # search and the wrong one for a failed calculation -
            # nothing was looked up, and a user who asked what 2+2 is
            # gets told their arithmetic could not be found.
            if {s["type"] for s in steps} <= _CODE_TYPES:
                answer = (
                    "I couldn't work that out - the script I wrote for it "
                    "didn't produce a usable result. Try rephrasing it?"
                )
            else:
                answer = "I couldn't find that in what I looked up."

        answer = _strip_tags(answer)
        answer = _strip_preamble(answer)
        answer = _strip_markdown(answer)

        # Last, so it reads what actually gets shown - including the
        # fallbacks above, which replace the answer outright. Checking
        # before them would have checked a draft that no longer exists.
        if self.self_check:
            answer = self._run_self_check(
                message,
                answer,
                support_pool or evidence,
            )

        answer = _augment_local_file_answer(
            grounding_request, answer, steps, results
        )

        # Answers used to be cut to the first four sentences here, which
        # silently truncated any reply that legitimately needed more -
        # a file summary covering several fields, for instance. Length
        # is now left to the model and its prompt.

        self.state.messages.append({"role": "user", "content": message})
        self.state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Remembered so the next turn can recognise a follow-up that
        # carries no subject of its own. "And in Mumbai?" is only a
        # weather question because the question before it was.
        self.state.last_capabilities = [s["type"] for s in steps]
        self.state.last_capability_steps = copy.deepcopy(steps)

        print(f"[STATE] messages now: {len(self.state.messages)} total")
        
        return answer

    def shutdown(self):

        stopped = set()

        for model in (
            self.planner_model,
            self.response_model,
            self.vision_model,
            self.code_model,
        ):

            if model.model_name in stopped:
                continue

            stopped.add(model.model_name)

            try:
                subprocess.run(["ollama", "stop", model.model_name])

            except Exception:
                pass
