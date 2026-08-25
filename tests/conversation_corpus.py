"""
Ordinary conversation.

Every sentence here is something a person might reasonably say to
Athena in the middle of a chat, and none of them is asking for a web
search, a file, or a script. The router's deterministic overrides must
not fire on any of them.

The overrides run before the model and cannot be overruled, which is
what makes them reliable and also what makes them dangerous: they all
match on words, and words turn up in ordinary English. "I always find
reasons not to go", said during a conversation about football, matched
the bare word "find", was forced to the capability route, and came back
as a generated Python script.

That bug was found by a user having a conversation. This file exists so
the next one is found by the test suite instead. Adding a pattern that
catches innocent English now fails the build immediately, rather than
surfacing weeks later as a strange answer nobody can reproduce.

When a false positive does get reported, add the sentence here first,
watch it fail, then fix the pattern.
"""

# Grouped by the override each set is most likely to trip, so a failure
# points at the pattern that needs narrowing rather than at a wall of
# strings.

FIND_AND_SEARCH = [
    "I always find reasons not to go",
    "I find that interesting",
    "we find it hard to say no",
    "people find this confusing",
    "you find the strangest things funny",
    "they never find time for it",
    "it is hard to find motivation",
    "I'm still trying to find my rhythm",
    "my friends find it funny",
    "that's a hard one to find words for",
]

BUILD_AND_MAKE = [
    "I made a presentation yesterday",
    "we built a treehouse when I was young",
    "my sister writes poetry",
    "I'm creating a lot of excuses lately",
    "that made me laugh",
    "I write in a journal most nights",
    "it makes sense now",
    "I generate a lot of ideas but never act on them",
]

RECENCY = [
    "I recently started playing football again",
    "I've been busy lately",
    "my last relationship ended badly",
    "that was my last day there",
    "I'm currently reading a good book",
    "today has been a long one",
    "it's been a rough month so far",
    "I saw him last week",
    "my current mood is not great",
]

LIVE_TOPICS = [
    "I love this weather",
    "the weather ruined our match",
    "I don't follow the news much",
    "temperature always drops at night here",
    "my dad checks share prices every morning",
    "the forecast was wrong again",
]

PEOPLE = [
    "I am Alex",
    "he is my best friend",
    "my coach is very strict",
    "she is the one who got me into it",
    "they are all older than me",
]

FILE_WORDS = [
    "I keep a document of my thoughts",
    "that image stuck with me",
    "my notes are a mess",
    "I read a lot of pdfs for college",
    "this text is hard to follow",
]

SMALL_TALK_AND_FEELINGS = [
    "nothing much, just tired",
    "I'm alright I think",
    "that's kind of you to say",
    "I don't really know how to answer that",
    "it's complicated",
    "I guess so",
    "we'll see how it goes",
    "I'd rather not talk about it",
]

CONVERSATION = (
    FIND_AND_SEARCH
    + BUILD_AND_MAKE
    + RECENCY
    + LIVE_TOPICS
    + PEOPLE
    + FILE_WORDS
    + SMALL_TALK_AND_FEELINGS
)


# The other half of the check. Narrowing a pattern until it stops
# catching conversation is easy if nothing insists it still catches the
# real thing.
#
# These specifically depend on an override firing. Each one is here
# because it once went wrong without it: the weather query answered
# from an open PDF, the file question answered "I don't have access to
# your files" about files on the Desktop, the PowerPoint request
# answered "I am text-based, I cannot generate files".
OVERRIDE_BACKED_REQUESTS = [
    "whats the weather in paris",
    "whats the weather in rupnagar",
    "is there a file named hostel fees",
    "find the hostel fees pdf",
    "can you find my resume",
    "who is the prime minister",
    "make a ppt on photosynthesis",
    "what is the current price of gold",
    "search for the budget spreadsheet",
    "look for invoice.pdf",
    "the weather in paris",          # terse, no question mark
]

# Real requests that reach no override by design - the model or a
# separate check decides these. They are still listed because the
# guard must not mistake any of them for conversation.
MODEL_DECIDED_REQUESTS = [
    "convert A+B*C to postfix",      # decided by the arithmetic check
    "summarize this document",       # decided by the active-file check
    "what time is it",               # decided by the router model
    "explain how recursion works",
]

REAL_REQUESTS = OVERRIDE_BACKED_REQUESTS + MODEL_DECIDED_REQUESTS
