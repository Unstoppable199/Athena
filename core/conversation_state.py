"""
Conversation state.
"""

from dataclasses import dataclass, field


@dataclass
class ConversationState:

    messages: list = field(default_factory=list)
    last_file_path: str = None

    # Code Athena generated is not the document the user is reading.
    # Keeping the paths separate prevents a later "summarize it" from
    # opening a scratch calculation script, while still allowing "run
    # that script" to work without asking for its path again.
    last_generated_path: str = None
    pending_file_paths: list = field(default_factory=list)

    # What the user actually asked when the ambiguous search ran, so a
    # later "the first one" still knows the job was "summarise it".
    pending_file_request: str = None

    # A live-data question can be incomplete on one turn and completed
    # on the next: "what's the weather like" -> "Delhi please".  The
    # second message does not contain the word weather, so remembering
    # what Athena just asked for is safer than asking a router to infer
    # it from a vague fragment.
    pending_lookup: str = None

    # Which saved conversation this is. None means nothing has been
    # said yet, so there is nothing worth writing to disk.
    conversation_id: str = None
    title: str = None

    # Older messages folded into prose, and how many of `messages` that
    # prose already covers. Kept here rather than recomputed because
    # summarising costs a model call - doing it twice for the same
    # exchange would be paying for the same work again.
    summary: str = ""
    summarized_upto: int = 0

    # A few explicit user facts are kept structurally as well as in the
    # prose summary.  Summaries are intentionally tiny and may compress
    # away a name or a corrected course after a long tool-heavy chat;
    # explicit statements such as "call me RJ" should not be lossy.
    user_profile: dict = field(default_factory=dict)

    # Which capabilities the previous turn actually ran.
    #
    # A short follow-up carries no subject of its own, so the router
    # cannot tell "and in Mumbai?" from ordinary conversation by
    # looking at the words. Asked the weather in Delhi and then that,
    # the small model sent it to chat - where there is no evidence and
    # no grounding check - and invented Mumbai's weather outright:
    # partly cloudy and 31.2 degrees against a real overcast 27.2, with
    # every other figure wrong too. Knowing what the last turn did is
    # what makes the follow-up recognisable.
    last_capabilities: list = field(default_factory=list)

    # The concrete steps from the previous completed turn. Types alone
    # identify a follow-up as live data; the arguments are what let a
    # challenge such as "u sure?" re-check the exact same city, ticker
    # or query without asking a planner to reconstruct it.
    last_capability_steps: list = field(default_factory=list)

    # Sources used by the most recently completed answer. This is
    # response metadata, not conversation memory, so it is deliberately
    # rebuilt each turn and is not sent back to the model.
    last_sources: list = field(default_factory=list)
