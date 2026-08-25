"""
Router.

Tiny first-stage decision: does this request need a live capability
(web search, filesystem, code execution, date/time), or can it be
answered directly in chat using fixed knowledge and reasoning?
"""

import json

import difflib
import re

from config import ROUTER_MAX_TOKENS


def _schema_for(active_file) -> dict:
    """ACTIVE_FILE is only offered as an answer when a file is actually
    open, so the model cannot pick it when there is nothing to read."""

    routes = ["SAFE", "CALCULATE", "NEEDS_LOOKUP"]

    if active_file:
        routes = routes + ["ACTIVE_FILE"]

    return {
        "type": "object",
        "properties": {"route": {"type": "string", "enum": routes}},
        "required": ["route"],
    }


# Appended to the router prompt only while a file is open.
#
# The keyword test below (is_active_file_request) catches the obvious
# phrasings for free, but it can only ever match words someone thought
# to list. A receipt has a reference number, an entry number, a branch,
# a payee, a date - asking about any of those reached no rule, so the
# question went to ordinary chat, where the model could only see the
# earlier summary and answered "not mentioned in the provided details"
# about text that was sitting in the file. Deciding this needs judgment
# about what the question refers to, so it is asked of the model rather
# than grown into a longer word list.
_ACTIVE_FILE_ROUTER_BLOCK = """

----------------------------------------

A file is currently open:

{path}

Return ACTIVE_FILE when the message asks about anything that would be
written in that file - any field, number, name, date, code or detail -
or asks to summarise, re-read, check or explain it.

Once a file is open, a plain question with no other obvious subject is
about that file. "What's the reference number", "when was it paid",
"who issued it", "does it mention a discount", "it starts with DUO" are
all ACTIVE_FILE, even though none of them say the word "file".

Use SAFE or NEEDS_LOOKUP only when the message has clearly moved on to
a different subject - a new topic to chat about, live information, or a
different file by name."""


# The prompt this replaced was ~2,230 tokens and was sent twice per
# message - roughly 4,500 tokens to answer a yes/no question. Measured
# over 24 messages, this short version reached the same decision on all
# 24 for about 82% fewer tokens, and the long one has now been deleted
# rather than left sitting behind a flag nobody turned.
#
# Worth knowing before trusting the score too far: this prompt was
# refined against those same 24 messages until they matched, so it
# flatters itself. The deterministic checks further down (recency
# markers, live topics, active-file requests) are the real safety net,
# and they run before the model is asked at all.
ROUTER_SYSTEM_PROMPT = """You decide how to answer a user's message.

Return SAFE if the answer comes from ordinary knowledge, explanation,
writing, or the conversation itself - things that do not change over
time and do not have to be worked out.

Return CALCULATE if the answer has to be worked out rather than
recalled: maths of any kind, physics, chemistry or engineering
problems, unit conversions, statistics, dates and durations, or
algorithms such as converting an expression to postfix or prefix.
Anything with a single correct answer that depends on applying rules
step by step belongs here, however simple it looks. Explaining a
concept is SAFE; producing a specific numeric or symbolic answer is
CALCULATE.

Return NEEDS_LOOKUP if answering needs something outside the model:
live or current information (weather, news, prices, scores, today's
date or time, latest versions), or anything on the user's computer
(files, folders, running code).

A lookup can only run if it has everything it needs. If the message is
missing the one detail that would make a search possible - which city,
which file, which thing "it" refers to - return SAFE, because the right
next move is to ask the user, not to search.

Rules:
- Weather with no city named, or only a country ("weather in the UK",
  "is it raining") -> SAFE, so we can ask which city.
- "there", "it", "that" with nothing earlier to point at -> SAFE.
- Anything already stated earlier in the conversation -> SAFE.
- Dead or historical people -> SAFE.
- "What is a derivative" -> SAFE. "Differentiate x^2 + 3x" -> CALCULATE.
- Money converted between currencies needs today's rate, so it is
  NEEDS_LOOKUP and never CALCULATE, however much it looks like a unit
  conversion. "How much is 50 USD in rupees" -> NEEDS_LOOKUP.
- Converting fixed units - km to miles, kg to pounds, C to F - has no
  rate to look up and stays CALCULATE.
- Any role, title or record that changes hands - president, prime
  minister, CEO, champion, world record, who currently holds or leads
  anything -> NEEDS_LOOKUP. Your training data is old, so you cannot
  know who holds a position now.

Return ONLY valid JSON: {"route": "SAFE"}"""


# A second opinion is useful only after the first classifier has asked
# to run something without an unambiguous signal in the request. It is
# deliberately shorter and framed as an audit: repeating the original
# prompt verbatim tended to reproduce the same mistake verbatim.
ROUTER_REVIEW_SYSTEM_PROMPT = """Review a tentative route for a local assistant.

Return SAFE when the request can be answered from ordinary knowledge,
reasoning, conversation, writing, advice, an explanation, or an example.

Return CALCULATE only when the user asks for one concrete numeric or
symbolic result that must be worked out. Asking for a programming
example, prose, ideas, a folder sketch, or an explanation is SAFE.

Return NEEDS_LOOKUP only when the answer requires current/live facts,
the user's local files, executing code, or creating a real file. Stable
facts such as country capitals are SAFE. Explaining a legal limitation
is SAFE unless the user asks to inspect a document or research current
law.

Judge the actual request, not the tentative route. Return only JSON."""


_RECENCY_MARKERS = re.compile(
    r"\b(current|currently|latest|last|newest|recent(ly)?|today|"
    r"tonight|this (week|month|year)|right now|as of now|"
    r"up[- ]to[- ]date|ongoing|so far)\b",
    re.IGNORECASE
)


# Subjects that are live by nature, whether or not the wording says so.
#
# "whats the weather in rupnagar" carries no recency marker, so it
# reached the model - which, with a fee receipt open, answered
# ACTIVE_FILE and tried to find the forecast in a PDF. Weather is never
# in an open file and never answerable from chat, so it is decided here
# instead of being offered as a judgment call.
_LIVE_TOPICS = re.compile(
    r"\b(weather|forecast|temperature|humidity|rainfall|"
    r"news|headlines|stock price|share price|exchange rate)\b",
    re.IGNORECASE,
)

# Whether a live-topic question says what it is about.
#
# "What's the weather" names no place, and a lookup cannot run without
# one. Forced to a capability anyway, the planner supplied a city
# itself - the answer came back as the current temperature in New
# Delhi, for a user who had not mentioned New Delhi or anywhere else.
#
# The router prompt already says a question missing the one detail that
# would make a search possible should be answered by asking. That rule
# never got a chance to apply, because the override returned before the
# model was consulted. So the override now stands aside when there is
# nothing to look up, and the model applies its own rule.
_NAMES_A_SUBJECT = re.compile(
    r"\b(?:in|at|near|around|for|of)\s+[\w'-]+"    # "in London", "of Apple"
    r"|\b(?:here|there|outside|today|tomorrow|tonight)\b"
    # The company normally comes before the topic: "Tesla stock
    # price". Requiring "price of Tesla" made that ordinary wording
    # look subjectless and Athena asked which company the user meant.
    r"|\b(?!(?:what(?:'s|s)?|is|the|a|any|current|latest)\b)"
    r"[A-Za-z][\w.&'-]*(?:\s+(?!(?:stock|share)\b)[A-Za-z][\w.&'-]*){0,3}"
    r"\s+(?:stock|share)\s+price\b",
    re.IGNORECASE,
)

_PERSONAL_PLANNING = re.compile(
    r"\b(?:help\s+me\s+plan|plan\s+(?:my|the)\s+day|"
    r"schedule\s+(?:my|the)|organize\s+(?:my|the)\s+day|"
    r"prioriti[sz]e\s+(?:my|these))\b",
    re.IGNORECASE,
)


def is_recency_request(message: str) -> bool:
    """Whether recency wording actually refers to changing information.

    The bare regex intentionally notices weak hints, but "last" also
    appears in "last name" and "the last thing I said". Those phrases
    are conversation memory, not a reason to rewrite a web query with
    today's date.
    """

    text = message or ""

    if re.search(
        r"\blast\s+(?:name|message|thing|question|answer|file|one)\b",
        text,
        re.IGNORECASE,
    ):
        return False

    # "Help me plan tonight" is personal scheduling, not a request for
    # changing public information. Treating the word "tonight" as decisive
    # skipped the router's safety review and sent a simple daily plan through
    # code generation in every mode.
    if (
        _PERSONAL_PLANNING.search(text)
        and not _LIVE_TOPICS.search(text)
    ):
        return False

    return bool(_RECENCY_MARKERS.search(text))


# Who someone is, and who holds an office, both go stale.
#
# "who is donald trump" was answered from training data - "45th
# president, 2017 to 2021" - and when told he was also the 47th, the
# model contradicted the user with a confident explanation that it
# hadn't happened yet, then repeated the mistake for the 46th. The chat
# prompt already forbids guessing at "current holders of a position"
# and forbids confident corrections; both were ignored, which is why
# this is decided in code instead.
#
# "who invented the telescope" deliberately does not match: settled
# history doesn't change, and sending it to a search would be slower
# for no gain. The cost of matching too widely is a few seconds; the
# cost of matching too narrowly is a confidently wrong answer.
#
# (The pattern itself is further down, next to the other matchers.)


# Asking for a file to be built is a job, not a conversation.
#
# "can you make a ppt" was answered in chat with "I am text-based; I
# cannot generate files" and an offer to hand over text to paste in
# manually - which is simply untrue, since Athena writes a script and
# runs it. Naming an artifact is the signal; "write me a song" or
# "write me a poem" name no file and stay in chat, where they belong.
# A named file format means a file, whatever the sentence around it is
# doing. Requiring a verb as well meant one slipped key - "mke a ppt on
# photosynthesis" - dropped the whole request back to chat, and these
# rules exist to survive ordinary typing.
_NAMED_FORMAT = re.compile(
    r"\b(?:ppt|pptx|powerpoint|docx|xlsx|excel|word\s+document"
    r"|spreadsheet|slide\s*deck)\b",
    re.IGNORECASE,
)

# Vaguer words need an intent verb, since "tell me about presentations"
# is a question rather than a job. Even then this is only a suggestion:
# "I write scripts for fun" fits the shape and asks for nothing.
_BUILD_INTENT = re.compile(
    r"\b(?:make|create|build|generate|write|prepare|produce)\b[^.?!]*?"
    r"\b(?:presentation|slides|csv|script|program|web\s*page"
    r"|website|html\s+page)\b",
    re.IGNORECASE,
)

_BUILD_ARTIFACT = re.compile(
    _NAMED_FORMAT.pattern + "|" + _BUILD_INTENT.pattern,
    re.IGNORECASE,
)


_VOLATILE_FACTS = re.compile(
    r"\bwho\s+(?:is|are)\b"
    r"|\b(?:president|prime\s+minister|chancellor|monarch|king|queen|pope"
    r"|governor|mayor|senator|chief\s+minister|ceo|chairman|chairperson)\b"
    r"|\b\d+(?:st|nd|rd|th)\s+(?:president|prime\s+minister)\b",
    re.IGNORECASE,
)


# Questions whose answer has to be worked out. Used only by modes with
# force_compute on, where anything numeric is sent through real code
# rather than trusted to the model's mental arithmetic - which is where
# multi-step word problems reliably go wrong.
#
# Deliberately narrow. It has to see either an operator between two
# numbers, or a word that means "work this out", because widening it to
# "mentions a number" would drag "born in 1947" through a Python round
# trip for nothing.
# An operator sitting between two numbers. Unambiguous on its own.
_ARITHMETIC_EXPRESSION = re.compile(r"\d+\s*[-+*/^%]\s*\d+")

# Verbs that ask for the work to be done. Kept to verbs on purpose:
# the nouns were worse than useless, because "what is a derivative"
# is a definition and "differentiate x^2" is a calculation, and only
# the verb tells them apart.
_COMPUTE_VERB = re.compile(
    r"\b(calculate|compute|evaluate|solve|simplify|convert"
    r"|integrate|differentiate|factorial|permutations?|combinations?"
    r"|percent(?:age)?\s+of|square\s+root|sqrt|round\s+to)\b",
    re.IGNORECASE,
)

# "How far", "how fast" and the like ask for a quantity - but only
# count as arithmetic when there are actual numbers to work with,
# which is what separates "how long is a piece of string" from "how
# long does 120 km at 60 km/h take".
_QUANTITY_QUESTION = re.compile(
    r"\bhow\s+(?:much|many|long|far|fast|old|tall|heavy)\b",
    re.IGNORECASE,
)

_HAS_NUMBER = re.compile(r"\d")

# An operator between two operands - "A+B", "2 * 3", "x^2".
#
# Notation questions are usually symbolic and carry no digits at all,
# which is what made requiring a number wrong: "the prefix of A+B" has
# nothing to match, so it was left to the model, which read "prefix" as
# a STRING prefix and answered "A+". The expression is the signal, not
# the digits.
#
# The last two alternatives are for expressions already in postfix or
# prefix, where the operator is not between its operands: converting
# "AB+" back to infix is the same kind of question and would otherwise
# have looked like ordinary prose.
_EXPRESSION = re.compile(
    r"[A-Za-z0-9]\s*[-+*/^]\s*[A-Za-z0-9]"     # A+B, 2 * 3
    r"|[A-Za-z0-9]{2,}\s*[-+*/^]"              # AB+
    r"|[-+*/^]\s*[A-Za-z0-9]{2,}"              # +AB
)


def looks_notation_conversion(message: str) -> bool:
    """Whether this asks for an expression in another notation.

    Both halves are needed. "Postfix" on its own is an ordinary word -
    a country dialling prefix, a filename prefix - and only becomes a
    conversion when there is something to convert.
    """

    text = message or ""

    if not _NOTATION.search(text):
        return False

    return bool(_EXPRESSION.search(text) or _HAS_NUMBER.search(text))

# Notation conversions, which are computation even when no arithmetic
# is visible.
_NOTATION = re.compile(
    r"\b(postfix|prefix|infix|binary|hexadecimal|octal|two'?s\s+complement)\b",
    re.IGNORECASE,
)

# Asking what something IS, not what it works out to. Checked first,
# so a definition is never dragged through a Python round trip.
_DEFINITION = re.compile(
    r"^\s*(?:what|whats|what's)\s+(?:is|are)\s+(?:a|an|the)\b"
    r"|\bwhat\s+does\b.*\bmean\b"
    r"|\b(?:define|meaning\s+of|difference\s+between)\b",
    re.IGNORECASE,
)


# Sums small enough that computing them costs more than it buys.
#
# "What's 2+2" was sent through code generation: a script written, saved
# and run in a subprocess, then an answer composed from its output -
# five model calls and fifteen seconds. Worse than slow, it introduced a
# way to fail. When the generated script had a bug the repair path ran,
# and if that missed too the reply was "I couldn't find that in what I
# looked up", about two plus two.
#
# Computation earns its cost on multi-step problems, where models
# fabricate confidently. One small addition is not that: a 12B model
# answers it correctly and instantly.
#
# Addition and subtraction only, at most two digits each, and only when
# that single operation is the entire question. Multiplication stays
# computed even at two digits - "45 * 12" is exactly where mental
# arithmetic starts slipping.
#
# The lead-in is an explicit list rather than "any non-digits", which
# was the first attempt and was far too loose: "differentiate x^2 + 3x"
# matched it, because "differentiate x^" is non-digits, then 2, then +,
# then 3, then "x". A calculus problem was being answered from memory.
# Anything before the sum now has to be a way of asking what it is.
_TRIVIAL_SUM = re.compile(
    r"^\s*(?:(?:what(?:'?s| is| are)?|how\s+much\s+is|calculate|whats)\s+)?"
    r"\d{1,2}\s*[-+]\s*\d{1,2}"
    r"\s*=?\s*\??\s*$",
    re.IGNORECASE,
)


def looks_arithmetic(message: str) -> bool:
    """Whether the answer has to be worked out rather than recalled."""

    text = message or ""

    if _TRIVIAL_SUM.match(text.strip()):
        return False

    if _ARITHMETIC_EXPRESSION.search(text):
        # Real numbers and an operator. Nothing else needs deciding,
        # not even a definition-shaped opening.
        return True

    if _DEFINITION.search(text):
        return False

    if _COMPUTE_VERB.search(text):
        return True

    if looks_notation_conversion(text):
        return True

    return bool(_QUANTITY_QUESTION.search(text) and _HAS_NUMBER.search(text))


_CURRENT_DATETIME_REQUEST = re.compile(
    r"\b(?:what(?:'?s|s| is)?|tell me|give me|check)\s+"
    r"(?:the\s+)?(?:current\s+)?(?:time|date|day|year)\b"
    r"|\bwhat\s+time\s+is\s+it\b"
    r"|\b(?:current|today'?s?)\s+(?:time|date|day|year)\b",
    re.IGNORECASE,
)


def asks_current_datetime(message: str) -> bool:
    """Whether the user explicitly asks for this machine's clock/date."""

    return bool(_CURRENT_DATETIME_REQUEST.search(message or ""))


# Asking which document something is in, rather than naming one.
#
# "Where did I write about linked lists" is a question about this
# computer, but nothing in it looks like a filename - no extension, no
# "named X", not even a verb about searching. Left to itself the router
# read it as ordinary conversation and answered "you did not write
# about linked lists", about a lecture PDF sitting in Downloads.
#
# A hint rather than an override: "which file" and "my notes" turn up
# in conversation too, and the model can see the difference between
# asking for one and mentioning one.
_DOCUMENT_REFERENCE = re.compile(
    r"\b(?:which|what)\s+(?:one\s+)?(?:of\s+)?(?:my\s+)?"
    r"(?:files?|documents?|pdfs?|notes?|papers?)\b"
    r"|\bwhere\s+(?:did|do|have)\s+i\s+(?:write|save|put|note|store)\b"
    r"|\bmy\s+notes?\s+(?:on|about)\b"
    r"|\b(?:file|document|pdf)\s+(?:that\s+)?(?:mentions?|contains?|about|with)\b"
    r"|\bsearch\s+(?:my|through\s+my)\s+(?:files?|documents?|notes?)\b",
    re.IGNORECASE,
)


_EXPLICIT_FILE_REFERENCE = re.compile(
    r"\b(file|document|pdf|receipt|spreadsheet)\b",
    re.IGNORECASE,
)

_PRONOUN_REFERENCE = re.compile(
    r"\b(it|this|that)\b",
    re.IGNORECASE,
)

_FILE_ACTION = re.compile(
    r"\b(summar(?:ize|ise)|read|open|inspect|analy(?:ze|se)|review|"
    r"explain|extract|show|describe|tell me about)\b",
    re.IGNORECASE,
)

_FILE_CONTENT_TERM = re.compile(
    r"\b(contents?|details?|total|amount|fees?|summary|text|table|pages?)\b",
    re.IGNORECASE,
)

_QUESTION_START = re.compile(
    r"^\s*(what|which|who|when|where|why|how|does|do|is|are|can|could|would)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------
# Statement guard
# ----------------------------------------------------------------
#
# The overrides below run before the model and cannot be overruled,
# which is what makes them reliable and also what makes them dangerous.
# Every one of them matches on words, and words appear in ordinary
# conversation: "I always find reasons not to go" was forced to the
# capability route on the word "find" and answered with a generated
# script, in the middle of a chat about football.
#
# The guard asks one question before an override is allowed to fire -
# is this person asking for something, or just talking? A sentence with
# no question mark, no request, no imperative, and a subject followed
# by a verb is someone talking.
#
# It only ever SUPPRESSES an override. A suppressed message goes to the
# model to be classified normally, so the cost of being wrong here is
# one model call, not a wrong answer. That is also why the default is
# False: anything the guard cannot confidently read as a statement
# keeps the old behaviour.

_REQUEST_MARKER = re.compile(
    r"\b(?:please|can\s+you|could\s+you|would\s+you|will\s+you"
    r"|help\s+me|i\s+need|i\s+want|i'?d\s+like|show\s+me|tell\s+me"
    r"|give\s+me|let\s+me\s+know)\b",
    re.IGNORECASE,
)

# A verb at the very start with no subject in front of it is an
# instruction: "find the receipt", "make a ppt", "convert this".
_IMPERATIVE_START = re.compile(
    r"^\s*(?:please\s+)?(?:find|locate|search|look|make|create|build"
    r"|generate|write|prepare|produce|open|read|show|tell|give|list"
    r"|convert|calculate|compute|solve|check|get|fetch|summar(?:ize|ise)"
    r"|explain|describe|analy(?:ze|se)|run|execute)\b",
    re.IGNORECASE,
)

# A subject followed by something it is doing. Deliberately narrow -
# together with the narrative markers below these are the only patterns
# that can return True, so a loose one would start switching overrides
# off.
#
# "the" is pointedly absent. "the weather in paris" is a perfectly
# normal way to ask for a forecast, and treating it as a statement
# would hold back the very override that exists to catch it.
_DECLARATIVE_START = re.compile(
    r"^\s*(?:i|i'?m|i'?ve|i'?d|we|we'?re|they|they'?re|he|she|it|it'?s"
    r"|my|our|there'?s|that'?s|thats|that\s+was|this\s+is"
    r"|today|yesterday|tomorrow|everyone|nobody|no\s+one)\b",
    re.IGNORECASE,
)

# Words that only appear when someone is recounting something. A
# sentence carrying one of these is describing rather than asking, even
# when it opens with a bare noun: "temperature always drops at night",
# "the forecast was wrong again".
#
# The past-tense catch needs four or more letters before the "ed" so it
# matches "ruined" and "started" without matching "need", "red" or
# "based", none of which are verbs in the sense meant here.
_NARRATIVE = re.compile(
    r"\b(?:was|were|been|being|had|used\s+to"
    r"|always|never|usually|often|lately|recently|already|ago"
    r"|\w{4,}ed)\b",
    re.IGNORECASE,
)

# Question words as they are actually typed. "whats the weather in
# rupnagar" has no apostrophe and no question mark, so the ordinary
# pattern misses it - and that is precisely the message the live-topic
# override was added for, so letting the guard call it a statement
# would undo the fix.
_QUESTION_START_LOOSE = re.compile(
    r"^\s*(?:whats?|whos?|hows?|wheres?|whens?|whys?|which"
    r"|does|do|did|is|are|was|were|can|could|would|will|should"
    r"|any|got|have|has)\b",
    re.IGNORECASE,
)


def is_plain_statement(message: str) -> bool:
    """Whether the message is someone talking rather than asking.

    Used to hold back the deterministic overrides. Says False whenever
    it is unsure, so the overrides keep working as they did.
    """

    text = (message or "").strip()

    if not text:
        return False

    # Any of these make it a request, whatever else it contains.
    if "?" in text:
        return False

    if _QUESTION_START.match(text) or _QUESTION_START_LOOSE.match(text):
        return False

    if _REQUEST_MARKER.search(text):
        return False

    if _IMPERATIVE_START.match(text):
        return False

    # Either an explicit subject at the front, or the sentence is
    # plainly recounting something.
    return bool(_DECLARATIVE_START.match(text) or _NARRATIVE.search(text))


# A message that names a file, or asks for one to be found, is about
# THAT file - not whichever one happens to be selected.
# Naming a file outright, or writing one out with its extension.
# Unambiguous on their own.
_FILE_NAME_SIGNAL = re.compile(
    r"\b(?:named|called)\s+\S"      # "a file named Hostel Fees"
    r"|(?:\S+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|md|py|json|log|png|jpe?g|env)"
    r"|\.env(?:\.[A-Za-z0-9_-]+)?)\b",
    re.IGNORECASE,
)

# Asking for something to be located. "find out" is excluded because it
# means discover, not search - "find out what time it is" is a question
# about the time, not about a file.
_LOCATE_VERB = re.compile(
    r"\b(?:find|locate|search\s+for|look\s+for)\b(?!\s+out\b)",
    re.IGNORECASE,
)

# The same verbs used to say that someone notices something, which is
# not a request to go and look for anything.
#
# This is what "I always find reasons not to go" was: a sentence about
# procrastinating, matched on the bare word "find", forced to the
# capability route before the model was consulted, and answered with a
# generated script. A deterministic override cannot be out-voted, which
# is why it produced the same wrong answer every time and why running
# the router twice would not have helped.
_LOCATE_AS_STATEMENT = re.compile(
    # "I always find", "people find", "we never find"
    r"\b(?:i|you|we|they|he|she|it|people|users?|someone|everyone)\s+"
    r"(?:always|often|usually|sometimes|never|just|really|only|still|"
    r"generally|tend\s+to|)\s*"
    r"(?:find|locate)\b"
    # "hard to find", "struggling to find" - the infinitive, which is
    # about difficulty rather than about a file.
    r"|\b(?:hard|difficult|easy|impossible|tough|unable|struggl\w*"
    r"|trying|try|manage[ds]?)\b[^.?!]{0,30}?\bto\s+(?:find|locate)\b",
    re.IGNORECASE,
)

# Words that make a sentence a request even when it is phrased around a
# subject. "Where can I find the report" is asking for the report;
# "I find reasons" is not asking for anything.
_LOCATE_AS_REQUEST = re.compile(
    r"\b(?:can|could|would|will|please|where|help)\b",
    re.IGNORECASE,
)


def names_a_file(message: str) -> bool:
    """Whether the message is asking for a particular file.

    A function rather than one regex because the locate verbs need
    context to mean anything: "find" is a request when it points at
    something to be found and ordinary English the rest of the time.
    """

    text = message or ""

    if _FILE_NAME_SIGNAL.search(text):
        return True

    if not _LOCATE_VERB.search(text):
        return False

    if _LOCATE_AS_STATEMENT.search(text) and not _LOCATE_AS_REQUEST.search(text):
        return False

    return True


# Greetings and acknowledgements refer to nothing at all.
#
# With a file open, "Hi" was classified ACTIVE_FILE: the router is
# asked what the message refers to, and given a choice between the open
# file and a lookup, a contentless message gets forced into one of
# them. The file was then read and the model, seeing it plus the
# earlier exchange, simply repeated its last answer - so "Hi" returned
# a reference number. Anchored to the whole message, so "hi, what is
# the reference number" is unaffected.
_SMALL_TALK = re.compile(
    r"^\s*(?:"
    r"hi|hey|hello|yo|hiya|sup|greetings"
    r"|thanks|thank\s+you|thx|ty|cheers"
    r"|ok|okay|k|kk|cool|nice|great|awesome|perfect|got\s+it|understood"
    r"|bye|goodbye|see\s+ya|good\s+(?:morning|afternoon|evening|night)"
    r"|how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)


def is_active_file_request(state, message: str) -> bool:
    """Return True when the request refers to the selected file."""

    if not getattr(state, "last_file_path", None):
        return False

    text = re.sub(r"\s+", " ", (message or "").strip())

    if not text:
        return False

    # "summarize a file named Budget" has both an action and the word
    # "file", so it used to be treated as being about the currently
    # selected file - silently re-reading the old one and never
    # searching for the name the user actually typed.
    if names_a_file(text):
        return False

    # These are complete new intents. A stale selected document must
    # never win merely because ordinary English contains "it" or
    # "that" ("what time is it", "isn't that 418", "run that
    # script").
    if (
        _LIVE_TOPICS.search(text)
        or _VOLATILE_FACTS.search(text)
        or _NAMED_FORMAT.search(text)
        or _DOCUMENT_REFERENCE.search(text)
        or looks_arithmetic(text)
        or looks_notation_conversion(text)
        or re.search(
            r"\b(?:what(?:'s|s)?\s+time\s+is\s+it|time\s+right\s+now|"
            r"today'?s\s+date|what\s+(?:date|day|year)\s+is\s+it)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"^\s*(?:please\s+)?(?:run|execute|test)\b.*\b(?:script|code|program)\b",
            text,
            re.IGNORECASE,
        )
    ):
        return False

    has_explicit_reference = bool(_EXPLICIT_FILE_REFERENCE.search(text))
    has_pronoun_reference = bool(_PRONOUN_REFERENCE.search(text))
    has_action = bool(_FILE_ACTION.search(text))

    if has_action and (has_explicit_reference or has_pronoun_reference):
        return True

    action_only = re.fullmatch(
        r"(?:yes[, ]+)?(?:please\s+)?"
        r"(?:can|could|would|will)?\s*"
        r"(?:you\s+)?(?:please\s+)?"
        r"(?:summar(?:ize|ise)|read|open|inspect|"
        r"analy(?:ze|se)|review|explain|extract|show|describe)"
        r"(?:\s+(?:it|this|that|the\s+(?:file|document|pdf)))?"
        r"\s*[?.!]*",
        text,
        re.IGNORECASE,
    )

    if action_only:
        return True

    # Bare field questions such as "what's the reference code?" are
    # about a file only while the previous turn was actually file work.
    # The path may remain selected for hours, but topic focus does not.
    recent_file_context = any(
        str(tool).startswith("filesystem.")
        for tool in (getattr(state, "last_capabilities", None) or [])
    )
    is_question = bool(
        _QUESTION_START.search(text)
        or _QUESTION_START_LOOSE.search(text)
        or re.match(
            r"^\s*(?:isn'?t|isnt|wasn'?t|wasnt|aren'?t|arent|"
            r"weren'?t|werent|don'?t|dont|doesn'?t|doesnt)\b",
            text,
            re.IGNORECASE,
        )
        or "?" in text
    )

    if has_explicit_reference and is_question:
        return True

    if not recent_file_context:
        return False

    # Discourse cues can refer to any field in a document. A fixed list of
    # words such as "reference", "amount" and "date" missed perfectly normal
    # follow-ups including "remind me what the main risk is" and challenges
    # such as "isn't the expiry in 2027?". Keep this tied to an immediately
    # preceding filesystem turn and exclude every independently recognised
    # new subject above, so a stale file cannot capture unrelated questions.
    file_followup_cue = re.search(
        r"^\s*(?:(?:quickly\s+)?remind\s+me\b|and\b|what\s+about\b|"
        r"how\s+about\b|but\b|nah\b|no\b|wait\b|really\b|"
        r"are\s+you\s+sure\b|u\s+sure\b|isn'?t\b|isnt\b|"
        r"wasn'?t\b|wasnt\b|i\s+thought\b)",
        text,
        re.IGNORECASE,
    )

    if file_followup_cue and (is_question or "remind me" in text.casefold()):
        return True

    if has_pronoun_reference and is_question:
        return True

    return bool(
        is_question
        and (
            _FILE_CONTENT_TERM.search(text)
            or re.search(
                r"\b(reference|ref|code|number|date|name|issuer|issued|paid|entry|branch)\b",
                text,
                re.IGNORECASE,
            )
        )
    )


_ORDINAL_WORDS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

_ORDINAL_RE = re.compile(
    r"\b(" + "|".join(_ORDINAL_WORDS) + r")\b",
    re.IGNORECASE,
)

# A bare number only counts as a selection when the message is
# essentially just that number ("2", "the 2nd", "number 3", "#3") -
# so "the first 3 lines" is not mistaken for picking file 3.
_EXPLICIT_NUMBER_RE = re.compile(
    r"\b(?:number|option|file|#)\s*(\d{1,2})\b",
    re.IGNORECASE,
)

_BARE_NUMBER_RE = re.compile(
    r"^\s*(?:the\s+)?(\d{1,2})\s*[.!]?\s*$",
    re.IGNORECASE,
)


# Only whole words are compared, and only against the ordinal names, so
# a stray close match on ordinary vocabulary is unlikely: the cutoff
# needs a single-character slip such as "frist"/"first" or
# "secodn"/"second", not a different word.
_ORDINAL_FUZZ_CUTOFF = 0.8


def _fuzzy_ordinal(text: str):
    """Position named by a mistyped ordinal word, or None."""

    for token in re.findall(r"[a-z]+", text.lower()):

        if len(token) < 4:
            continue

        match = difflib.get_close_matches(
            token, _ORDINAL_WORDS.keys(), n=1, cutoff=_ORDINAL_FUZZ_CUTOFF
        )

        if match:
            return _ORDINAL_WORDS[match[0]]

    return None


def selection_is_pending(state) -> bool:
    """Whether the last turn asked the user to choose between files."""

    return bool(getattr(state, "pending_file_paths", None))


# Capabilities whose answers go stale immediately, and which a short
# follow-up almost always means to repeat with one detail changed.
_LIVE_CAPABILITIES = {
    "weather.current", "finance.quote", "finance.exchange",
    "web.search", "system.datetime",
}

# A direct calculation is also safe to repeat when the user challenges
# its result.  Keep this narrower than every executable capability:
# ``python.run`` may be an arbitrary saved script with side effects,
# while ``code.run`` is Athena's short-lived calculation sandbox.
_RECHECKABLE_CAPABILITIES = _LIVE_CAPABILITIES | {"code.run"}

# The words a sentence uses when it is carrying on from the last one
# rather than starting something.
_CONTINUES = re.compile(
    r"^\s*(?:and|&|what\s+about|how\s+about|also|then|or|in|"
    r"(?:nah|no|wait)[,\s]+(?:i\s+)?mean(?:t)?)\b",
    re.IGNORECASE,
)

_LIVE_CHALLENGE = re.compile(
    r"^\s*(?:but|nah|no|wait|really|are\s+you\s+sure|u\s+sure|"
    r"isn'?t|isnt|wasn'?t|wasnt|could\s+it|i\s+thought)\b",
    re.IGNORECASE,
)

_LIVE_SUBJECT_CORRECTION = re.compile(
    r"^\s*(?:nah|no|wait)[,\s]+(?:i\s+)?mean(?:t)?\s+"
    r"(?:in|at|for)\s+\S+",
    re.IGNORECASE,
)


def challenges_last_lookup(state, message: str) -> bool:
    """Whether the user disputes a fact Athena can safely re-check."""

    previous = getattr(state, "last_capabilities", None) or []
    text = (message or "").strip()

    # "Nah, I mean in Tokyo" changes the lookup's subject. Repeating
    # the exact previous arguments turned this into the local date
    # again; only a real challenge such as "isn't it 45 degrees?"
    # should rerun the same request unchanged.
    if _LIVE_SUBJECT_CORRECTION.match(text):
        return False

    return bool(
        any(capability in _RECHECKABLE_CAPABILITIES for capability in previous)
        and _LIVE_CHALLENGE.match(text)
        and len(text.split()) <= 16
    )


def continues_a_lookup(state, message: str) -> bool:
    """Whether this short message continues the previous lookup.

    "And in Mumbai?" contains no weather word, no place the router
    knows, and nothing else to go on - so the small model classified it
    as ordinary conversation and answered it from the chat path, which
    has no evidence behind it and no grounding check in front of it. It
    invented Mumbai's weather: partly cloudy and 31.2 degrees, against
    a real overcast 27.2, with every other figure wrong as well.

    What makes it recognisable is not the words but what came before.
    A short message with no subject of its own, right after a lookup,
    is that lookup again.
    """

    previous = getattr(state, "last_capabilities", None) or []

    if not any(c in _LIVE_CAPABILITIES for c in previous):
        return False

    text = (message or "").strip()

    # A correction or challenge after live data must be checked against
    # the source again. Sending it to chat lets the model agree with a
    # false number or invent a compromise between the two values.
    if challenges_last_lookup(state, text):
        return True

    if not text or names_a_new_subject(text):
        # Names its own subject, so it is a new question - "who is the
        # president" after a weather lookup is not about the weather.
        return False

    words = text.split()

    # It has to read as a FRAGMENT, not merely be short. Length alone
    # was the first attempt and it grabbed "explain how recursion
    # works" - four words, a complete question, and nothing to do with
    # the lookup before it.
    if _CONTINUES.match(text):
        return len(words) <= 6

    # A bare noun with no verb in front of it: "Mumbai?", "Apple".
    # Anything that opens with a question word or an instruction is
    # asking something new, however short.
    return (
        len(words) <= 3
        and not _QUESTION_START.match(text)
        and not _IMPERATIVE_START.match(text)
    )


def missing_subject_capability(message: str):
    """Capability awaiting a subject, or None when the request is complete."""

    text = message or ""

    if (
        is_plain_statement(text)
        or not _LIVE_TOPICS.search(text)
        or _NAMES_A_SUBJECT.search(text)
    ):
        return None

    lowered = text.lower()

    if re.search(r"\b(weather|forecast|temperature|humidity|rainfall)\b", lowered):
        return "weather.current"

    if re.search(r"\b(stock|share)\s+price\b", lowered):
        return "finance.quote"

    if "exchange rate" in lowered:
        return "finance.exchange"

    return "web.search"


def missing_subject_question(message: str) -> str:
    """The question to ask back when a lookup has nothing to look up.

    "What's the weather" names a live subject and no place, so there
    is nothing a lookup could run on. The router deliberately hands
    these to the model so it can ask which city - and the model
    answered them instead, from the conversation: after looking up
    Delhi and Mumbai it reported "overcast, 28.6 degrees, humidity
    88%", numbers close enough to Delhi's real reading to pass for a
    fresh one and belonging to nowhere at all.

    Chat has no evidence behind it and no grounding check in front of
    it, so the only safe answer to an unanswerable question is the
    question back. Returns "" when nothing is missing.
    """

    text = message or ""

    capability = missing_subject_capability(text)

    if not capability:
        return ""

    if capability == "weather.current":
        return "Which city's weather would you like?"

    if capability == "finance.quote":
        return "Which company's share price would you like?"

    if capability == "finance.exchange":
        return "Between which two currencies?"

    return "Which one would you like me to look up?"


def names_a_new_subject(message: str) -> bool:
    """Whether a message clearly starts a different topic.

    Used to decide when a short reply is NOT an answer to "which file
    did you mean". "whats the weather in paris" is five words, which
    was short enough to be treated as a failed file choice and answered
    by asking again - even though it names a subject that has nothing
    to do with files. Anything a deterministic rule already recognises
    is a new subject, not a fumbled selection.

    Sums and notation conversions count too. "Convert 255 to binary" is
    four words and matched none of the topic patterns, so after an
    unanswered "which of these files did you mean?" it was taken as a
    fumbled choice - and the reply to a question about binary was a
    list of PDFs in the Downloads folder. Nobody answers "which file?"
    with a base conversion.
    """

    text = message or ""

    return bool(
        _LIVE_TOPICS.search(text)
        or _VOLATILE_FACTS.search(text)
        or _BUILD_ARTIFACT.search(text)
        or is_recency_request(text)
        or looks_arithmetic(text)
        # Trivial sums are excluded from looks_arithmetic on purpose -
        # they are answered directly rather than computed - but "whats
        # 2+2" is still obviously not the name of a file.
        or _TRIVIAL_SUM.match(text.strip())
    )


def resolve_pending_file_selection(state, message: str):
    """Resolve an ordinal reply against the file matches we actually
    showed the user on the previous turn.

    Which file "the first one" means is state we already hold - it is
    not something that needs to be inferred. Resolving it here keeps
    the choice deterministic, instead of depending on the planner
    reconstructing a full Windows path out of the conversation text.

    Returns the selected path, or None when the message isn't a
    selection (or there is nothing pending to select from).
    """

    pending = getattr(state, "pending_file_paths", None) or []

    if not pending:
        return None

    text = re.sub(r"\s+", " ", (message or "").strip())

    if not text:
        return None

    def _at(position):
        if 1 <= position <= len(pending):
            return pending[position - 1]
        return None

    if re.search(r"\b(last|final)\s+(one|file)?\b", text, re.IGNORECASE):
        return pending[-1]

    ordinal = _ORDINAL_RE.search(text)

    if ordinal:
        return _at(_ORDINAL_WORDS[ordinal.group(1).lower()])

    # Mistyped ordinals still count. "frist one" reached none of the
    # patterns above, so the selection silently failed and the reply was
    # handled as ordinary chat - where the model, with no file and no
    # grounding, invented a whole fee schedule for a payment receipt.
    # A near-miss on a word like "first", right after we asked which
    # file was meant, is a selection and not a new subject.
    close = _fuzzy_ordinal(text)

    if close:
        return _at(close)

    explicit = _EXPLICIT_NUMBER_RE.search(text)

    if explicit:
        return _at(int(explicit.group(1)))

    bare = _BARE_NUMBER_RE.match(text)

    if bare:
        return _at(int(bare.group(1)))

    return None


# ----------------------------------------------------------------
# Hints
# ----------------------------------------------------------------
#
# Not every signal deserves the same authority.
#
# Some are unambiguous by construction. "invoice.pdf" contains a file
# extension; "a file named Budget" says so outright; "make a ppt" names
# a format. Nothing else in English looks like those, so they still
# decide the route outright.
#
# The rest are ordinary words that happen to be useful signals. "find",
# "make", "weather", "who is", "last" all appear constantly in normal
# conversation, and a pattern that forces a route on one of them cannot
# be argued with - which is how "I always find reasons not to go"
# became a generated Python script.
#
# Those are passed to the model as hints instead. The model sees what
# was noticed and what it usually means, reads the whole sentence, and
# decides. A hint that is wrong costs nothing; an override that is
# wrong costs the answer.
#
# What the hints do NOT do is restore the choice that caused the worst
# failures: when a hint fires, the open file is dropped from the
# schema. "Whats the weather in rupnagar" with a receipt open was
# answered by searching the PDF for a forecast, and no amount of
# guidance is worth leaving that reachable.

# The line is drawn at verbs.
#
# Nouns like "weather", "forecast" and "share price" name subjects that
# are live whatever sentence they appear in, and once the statement
# guard has removed "I love this weather" from consideration, what is
# left really is a request about them. The same goes for asking who
# holds a position. Both stay deterministic, because being wrong there
# means answering from stale training data or searching a PDF for a
# forecast - fabrications, not slow answers.
#
# Verbs are different. "Find", "make", "build" and "write" describe
# what someone is doing as often as what they want done, and they are
# what actually broke: "I always find reasons not to go" became a
# generated script. Those become hints.
#
# "Last", "current" and "recent" join them - "last" matches "last
# name" as readily as "last week".
_HINTS = (
    (
        "_RECENCY",
        lambda m: is_recency_request(m),
        "mentions something current or recent, which usually cannot be "
        "answered from memory",
    ),
    (
        "_BUILD_INTENT",
        lambda m: _BUILD_INTENT.search(m),
        "may be asking for a file to be built rather than for an "
        "explanation",
    ),
    (
        # names_a_file rather than the bare verb: it already knows that
        # "people find this confusing" and "find out what time it is"
        # are not requests to look for anything, and that logic should
        # not be written twice. The extension case is excluded because
        # it is decided outright above, not hinted.
        "_LOCATE",
        lambda m: names_a_file(m) and not _FILE_NAME_SIGNAL.search(m),
        "may be asking for something to be located on this computer",
    ),
)


def gather_hints(message: str) -> list:
    """Weak signals found in the message, as sentences for the model.

    Weak meaning the words are also ordinary English. Each one is
    something worth noticing and nothing worth forcing.
    """

    return [
        description
        for _name, matches, description in _HINTS
        if matches(message or "")
    ]


class Router:

    def __init__(self, model):
        self.model = model

    def route(self, state, message) -> str:

        # Small talk needs no file, no lookup and no model call. Asking
        # costs a round trip to answer "hello", and - as the greeting
        # that returned a reference number showed - asking can get the
        # wrong answer as well as a slow one.
        if _SMALL_TALK.match(message):
            print("[ROUTER OVERRIDE] small talk -> chat")
            return "chat"

        if _PERSONAL_PLANNING.search(message or "") and not _LIVE_TOPICS.search(message or ""):
            print("[ROUTER OVERRIDE] personal planning -> chat")
            return "chat"

        # Held back for anything that reads as the user talking rather
        # than asking. Every override below matches on words that also
        # occur in ordinary conversation, and a forced route cannot be
        # argued with - so the guard decides once, here, instead of
        # each pattern having to defend itself against English.
        #
        # A guarded message is not routed to chat: it goes to the model
        # and is classified normally. The cost of guarding wrongly is
        # one model call.
        talking = is_plain_statement(message)

        if talking:
            print("[ROUTER GUARD] reads as a statement -> overrides held back")

        if is_active_file_request(state, message):
            print("[ROUTER OVERRIDE] active file request -> capability")
            return "capability"

        # A numeric transformation that explicitly continues the previous
        # file answer needs both that context and a real calculation. Letting
        # ACTIVE_FILE win merely rereads and repeats the original value;
        # letting ordinary chat win performs unverified mental arithmetic.
        if (
            any(
                str(tool).startswith("filesystem.")
                for tool in (getattr(state, "last_capabilities", None) or [])
            )
            # An explicit filename/path is a new read request, even when it
            # also says "how much" or "how many". Treating it as a follow-up
            # calculation bypasses the file and makes the planner invent its
            # contents. Only subjectless follow-ups may reuse the last file's
            # values for arithmetic.
            and not names_a_file(message)
            and looks_arithmetic(message)
            and re.search(
                r"\b(?:instead|scale|scaled|double|triple|half|per\b|each\b|"
                r"for\s+\d+|how\s+much|how\s+many|new\s+total)\b",
                message or "",
                re.IGNORECASE,
            )
        ):
            print("[ROUTER OVERRIDE] file-derived calculation -> calculate")
            return "calculate"

        # ---- Strong signals: still decided here ----
        #
        # A file extension, an explicit "named X", a named document
        # format. Nothing else in English looks like these, so there is
        # no judgment to exercise and nothing for the model to add.
        #
        # Withholding ACTIVE_FILE from the schema was tried for these
        # and was not enough: it stopped the wrong open file being
        # reused but left SAFE on the table, and with a few prior turns
        # in the prompt the model chose it - answering "is there a file
        # named hostel fees" with "I don't have access to your files"
        # about three files sitting on the Desktop.
        if not talking and _FILE_NAME_SIGNAL.search(message):
            print("[ROUTER OVERRIDE] names a file outright -> capability")
            return "capability"

        if not talking and _NAMED_FORMAT.search(message):
            print("[ROUTER OVERRIDE] names a document format -> capability")
            return "capability"

        # Live subjects and who-holds-what. These are nouns, not verbs:
        # once the statement guard has taken out "I love this weather",
        # a message naming the weather is asking about the weather.
        # Kept deterministic because the failures here are fabrications
        # rather than slow answers - a forecast read out of an open PDF,
        # or a president four years out of date defended confidently.
        # Only when the question says what it is about. "What's the
        # weather" has nothing to look up, and forcing it to a lookup
        # made the planner choose a city on the user's behalf.
        if not talking and _LIVE_TOPICS.search(message):

            if _NAMES_A_SUBJECT.search(message):
                print("[ROUTER OVERRIDE] live topic -> capability")
                return "capability"

            print("[ROUTER] live topic with nothing named -> asking the model")

        if not talking and _VOLATILE_FACTS.search(message):
            print("[ROUTER OVERRIDE] who/office question -> capability")
            return "capability"

        # A short follow-up to a lookup, which the words alone cannot
        # identify. Decided here because sending it to chat is how
        # Mumbai's weather got invented - the chat path has no evidence
        # and nothing checking what it says.
        if not talking and continues_a_lookup(state, message):
            print("[ROUTER OVERRIDE] continues the previous lookup -> capability")
            return "capability"

        # An expression to be rewritten in another notation. Nothing
        # else "the postfix of A+B" could mean, and it has to be worked
        # out rather than recalled.
        #
        # Left to the model this went to chat, where "prefix" was read
        # as a STRING prefix: asked for the prefix of A+B it answered
        # "A+", then defended it. The correct answer is "+AB", and no
        # amount of conversation was going to get there.
        #
        # Deterministic rather than a hint because the failure is a
        # confident wrong answer, not a slow one - and because it only
        # fires when there is an actual expression present, which no
        # ordinary use of the word "prefix" has.
        if not talking and looks_notation_conversion(message):
            print("[ROUTER OVERRIDE] notation conversion -> calculate")
            return "calculate"

        # Asking which document something is in. Deterministic for the
        # same reason as the two above: these are phrases, not bare
        # verbs, and "where did I write about linked lists" cannot mean
        # anything except "search my documents".
        #
        # This was a hint first, and gemma3:12b declined it - answering
        # "I don't have access to your personal files or writing", which
        # is untrue, about a lecture PDF in Downloads. qwen3:8b took the
        # hint and the 12b did not, which is the whole problem with a
        # guarantee that depends on which model is loaded.
        if not talking and _DOCUMENT_REFERENCE.search(message):
            print("[ROUTER OVERRIDE] asks which document -> capability")
            return "capability"

        # ---- Weak signals: passed to the model as hints ----
        #
        # Everything else these patterns match is ordinary English.
        # Forcing a route on "find", "make", "weather" or "last" is
        # what produced the wrong answers; the model gets to read the
        # sentence and decide.
        hints = [] if talking else gather_hints(message)

        if hints:
            print(f"[ROUTER HINT] {len(hints)} signal(s) noticed")

        active_file = getattr(state, "last_file_path", None)

        # A selected path persists so explicit "summarize the file"
        # can work later, but topic focus does not persist forever.
        # Offering ACTIVE_FILE after unrelated turns let the classifier
        # answer maths, weather and image follow-ups from an old PDF.
        recent_file_context = any(
            str(tool).startswith("filesystem.")
            for tool in (getattr(state, "last_capabilities", None) or [])
        )
        if not recent_file_context:
            active_file = None

        # A hint means the answer is somewhere other than the open
        # file, so the open file stops being an option. This is the
        # part of the old overrides worth keeping outright: a weather
        # question answered out of a PDF is not a near miss.
        if hints:
            active_file = None

        history = state.messages[-6:]

        conversation = ""

        for msg in history:
            content = str(msg.get("content") or "")
            if len(content) > 1200:
                content = content[:1200].rstrip() + " [...]"
            conversation += (
                f'{msg["role"].capitalize()}: '
                f'{content}\n'
            )

        # Placed after the request, so they read as notes on what was
        # just asked rather than as instructions that outrank it.
        hint_block = ""

        if hints:
            listed = "\n".join(f"- This message {h}." for h in hints)
            hint_block = f"""
Signals noticed in this message:

{listed}

These are observations, not instructions. They come from matching
words, and words are often used in other ways - a message can mention
the weather without asking for a forecast. Judge the sentence as a
whole; if the signal does not fit what is actually being asked,
ignore it.
"""

        prompt = f"""
Conversation History:

{conversation}

Current User Request:

{message}
{hint_block}
Decide the route.

Return ONLY valid JSON.
"""

        # Classified once, not twice. The second call used to re-ask
        # the identical question to confirm a SAFE, but measured over
        # 16 messages the two calls never once disagreed - which is
        # what you would expect, since the same prompt and the same
        # model with a constrained two-value output lands in the same
        # place. It doubled the cost of every chat turn and caught
        # nothing. The same reasoning is already noted above for the
        # recency override: this router's mistakes are systematic, so
        # asking again reproduces the mistake rather than catching it.
        decision = self._classify(prompt, active_file)

        # SAFE is cheap and harmless, while a mistaken positive route
        # can start web requests or generate and execute code. Review
        # only those model-selected positives which were not already
        # backed by a concrete arithmetic/clock signal. This catches
        # occasional classifications such as a Python *example*, a
        # folder sketch, or a legal disclaimer being labelled
        # CALCULATE/NEEDS_LOOKUP, without doubling every chat turn.
        if (
            decision in {"CALCULATE", "NEEDS_LOOKUP"}
            and not looks_arithmetic(message)
            and not looks_notation_conversion(message)
            and not asks_current_datetime(message)
            # "latest", "last winner" and other explicit recency
            # requests genuinely require public information. The audit
            # model intermittently downgraded them to SAFE, producing a
            # stale Python version and a 2022 World Cup answer from
            # model memory. is_recency_request already excludes ordinary
            # phrases such as "last name" and "last message".
            and not is_recency_request(message)
        ):
            reviewed = self._review(
                f"""Recent conversation:

{conversation}

Current request:

{message}

Tentative route: {decision}

Return the correct route as JSON."""
            )

            if reviewed in {"SAFE", "CALCULATE", "NEEDS_LOOKUP"}:
                if reviewed != decision:
                    print(f"[ROUTER REVIEW] {decision} -> {reviewed}")
                decision = reviewed

        if decision == "ACTIVE_FILE":
            route = "file"
        elif decision == "SAFE":
            route = "chat"
        elif decision == "CALCULATE":

            # The model is right that this is arithmetic and wrong that
            # it needs computing. Checked here as well as in
            # looks_arithmetic because the two reach CALCULATE by
            # different routes - one from this decision, one from a
            # mode that forces computation - and fixing only the second
            # left "what's 2+2" still writing itself a Python script.
            if _TRIVIAL_SUM.match((message or "").strip()):
                print("[ROUTER] arithmetic small enough to just answer -> chat")
                route = "chat"
            else:
                # Kept distinct from a lookup so the planner can be told
                # what was decided here, rather than working it out
                # again.
                route = "calculate"
        else:
            # CALCULATE and NEEDS_LOOKUP both go to the planner. They
            # are separate answers because the distinction is one the
            # model can make reliably and a word list cannot: "what is
            # a derivative" and "differentiate x^2 + 3x" share their
            # vocabulary entirely, and only one of them has to be
            # worked out.
            route = "capability"

        print("[ROUTER DECISION]:", route)

        return route

    def _classify(self, prompt, active_file=None) -> str:

        system = ROUTER_SYSTEM_PROMPT

        if active_file:
            system += _ACTIVE_FILE_ROUTER_BLOCK.format(path=active_file)

        response = self.model.complete(
            system,
            prompt,
            schema=_schema_for(active_file),
            num_predict=ROUTER_MAX_TOKENS,
            think=False
        )

        print("\n[ROUTER RAW]:", repr(response))

        try:
            parsed = json.loads(response)
            return parsed.get("route", "").strip().upper()
        except Exception:
            return ""

    def _review(self, prompt) -> str:
        """Audit a costly model-selected route with a smaller contract."""

        response = self.model.complete(
            ROUTER_REVIEW_SYSTEM_PROMPT,
            prompt,
            schema=_schema_for(None),
            num_predict=ROUTER_MAX_TOKENS,
            think=False,
        )

        print("\n[ROUTER REVIEW RAW]:", repr(response))

        try:
            parsed = json.loads(response)
            return parsed.get("route", "").strip().upper()
        except Exception:
            return ""
