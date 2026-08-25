"""
Quality checks that need a real model.

The regression suite next door is deterministic: fake models, fake
executor, no network. That makes it fast and trustworthy, and blind to
the thing Athena is actually judged on. It cannot tell you that an
answer invented a number, or repeated a fee schedule that was never in
the file, or confidently corrected the user with four-year-old
information. Every one of those has happened here.

So this file asks the real thing real questions and checks properties
of the answers rather than their exact words. A local model does not
produce the same sentence twice, and asserting on wording would fail
constantly for no reason - but "did not invent a figure" and "asked
which city instead of guessing" are stable regardless of phrasing.

Run it by hand before a release, not in a loop:

    .venv/Scripts/python tests/eval_quality.py
    .venv/Scripts/python tests/eval_quality.py --mode fast
    .venv/Scripts/python tests/eval_quality.py --only weather

It is slow (a model call or several per case), it needs Ollama running,
and some cases need the internet. Failures are worth reading rather
than fixing blindly: a local model will occasionally miss one, so the
signal is a case that fails repeatedly, or a run that is suddenly far
worse than the last.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent import Agent, DEFAULT_MODE


# ----------------------------------------------------------------
# Properties an answer can be checked for
# ----------------------------------------------------------------
#
# Each returns (passed, detail). They describe shape, not wording.


def says(*needles):
    """The answer mentions at least one of these."""

    def check(answer, _agent):
        found = [n for n in needles if n.lower() in answer.lower()]
        return bool(found), f"looked for any of {list(needles)}"

    return check


def avoids(*needles):
    """The answer mentions none of these."""

    def check(answer, _agent):
        found = [n for n in needles if n.lower() in answer.lower()]
        return not found, f"should not contain {found}" if found else ""

    return check


def asks_a_question():
    """The answer asks for something rather than guessing at it.

    The failure this guards against is answering a question that was
    never fully asked - picking a city for "what's the weather", or a
    file when three matched.
    """

    def check(answer, _agent):
        return "?" in answer, "no question mark; it answered instead of asking"

    return check


def no_invented_numbers():
    """No figures beyond those in the evidence.

    The worst failure this project has had: a mistyped ordinal fell
    through to ungrounded chat and the model produced an entire fee
    schedule - room categories, deposits, late fees - for what was a
    one-line payment receipt.
    """

    def check(answer, agent):
        numbers = set(re.findall(r"\d[\d,]*\.?\d*", answer))

        if not numbers:
            return True, ""

        evidence = " ".join(
            str(m.get("content") or "") for m in agent.state.messages
        )

        invented = [n for n in numbers if n not in evidence]

        # Small numbers are usually ordinary prose ("two options"), not
        # claims. Only larger figures are worth failing over.
        invented = [n for n in invented if len(n.replace(",", "")) >= 3]

        return not invented, f"figures not in the evidence: {invented}"

    return check


def used(*tools):
    """The named capability actually ran."""

    def check(_answer, _agent):
        from core.agent import PROGRESS
        ran = list(PROGRESS.tools)
        hit = [t for t in tools if t in ran]
        return bool(hit), f"wanted one of {list(tools)}, ran {ran}"

    return check


def stayed_in_chat():
    """No capability ran - it was answered directly."""

    def check(_answer, _agent):
        from core.agent import PROGRESS
        ran = list(PROGRESS.tools)
        return not ran, f"expected no tools, ran {ran}"

    return check


def admits_ignorance():
    """It says it could not find something, rather than inventing it."""

    phrases = ("couldn't find", "could not find", "don't have", "do not have",
               "no information", "not sure", "unable to", "couldn't",
               "i don't know", "not able", "no file found", "no such file",
               "doesn't exist", "does not exist", "nothing matching",
               "no matches", "no results")

    def check(answer, _agent):
        lowered = answer.lower()
        return any(p in lowered for p in phrases), "did not admit it lacked the answer"

    return check


def no_evidence_markers():
    """The reply contains no [S1]-style citation tags.

    These are scaffolding for the grounding check and never meant to
    be read. They only started leaking once script output became
    citable evidence: a file's text gets paraphrased, so its tags
    rarely survive, but a computed result is copied out exactly as
    printed - "The prefix of A+B is [S1] + A B."
    """

    import re as _re
    marker = _re.compile(r"(?<!\w)\[S\d+\]")

    def check(answer, _agent):
        found = marker.search(answer)
        return not found, f"leaked marker: {found.group(0)!r}" if found else ""

    return check


def longer_than(words):
    def check(answer, _agent):
        n = len(answer.split())
        return n >= words, f"only {n} words"

    return check


# ----------------------------------------------------------------
# The cases
# ----------------------------------------------------------------
#
# "needs" marks what a case depends on, so a run without the internet
# can skip the ones that would fail for that reason rather than for a
# real one.

CASES = [
    # ---- Not guessing ----
    {
        "name": "weather-no-city",
        "ask": "what's the weather",
        "why": "No city was named, so there is nothing to look up. It "
               "should ask rather than pick somewhere.",
        "checks": [asks_a_question()],
    },
    {
        "name": "weather-country-only",
        "ask": "whats the weather in japan",
        "why": "A country has no single weather. Answering for one "
               "arbitrary point inside it and calling it Japan is wrong.",
        "checks": [asks_a_question(), says("city", "which")],
        "needs": "internet",
    },

    # ---- Live information ----
    {
        "name": "weather-city",
        "ask": "what's the weather in London right now",
        "why": "A real lookup, not a recollection.",
        "checks": [used("weather.current", "web.search")],
        "needs": "internet",
    },
    {
        "name": "current-officeholder",
        "ask": "who is the prime minister of the united kingdom",
        "why": "Training data goes stale, and this model has previously "
               "defended an out-of-date answer when corrected.",
        "checks": [used("web.search", "weather.current")],
        "needs": "internet",
    },

    # ---- Computation ----
    {
        "name": "arithmetic-word-problem",
        "ask": "a train travels 240 km in 3 hours, then 150 km in 2 hours. "
               "what is its average speed for the whole journey?",
        "why": "78 km/h. Mental arithmetic on multi-step problems is "
               "where this model reliably goes wrong.",
        "checks": [says("78")],
    },
    {
        "name": "small-sum-answered-directly",
        "ask": "WHats 2+2",
        "why": "This was routed to code generation: a script written, "
               "saved and run in a subprocess - five model calls and "
               "fifteen seconds. When the generated script had a bug, "
               "the reply was 'I couldn't find that in what I looked "
               "up', about two plus two.",
        # "four" as well as "4": the 12b answered "That would be four."
        # - correct, and a digit check called it a failure.
        "checks": [stayed_in_chat(), says("4", "four")],
    },
    {
        "name": "postfix-conversion",
        "ask": "convert A+B*C to postfix",
        "why": "ABC*+. Also guards the markdown stripper, which once ate "
               "the asterisks and corrupted the answer.",
        "checks": [says("ABC*+", "A B C * +"), no_evidence_markers()],
    },
    {
        "name": "prefix-conversion-routes-correctly",
        "ask": "whats the prefix of A+B",
        "why": "+AB. Left to the model this went to chat, where 'prefix' "
               "was read as a STRING prefix and answered 'A+' - then "
               "defended across several more turns. The notation check "
               "used to require a digit, and this expression has none.",
        "checks": [used("code.generate", "python.run"),
                   says("+AB", "+ A B", "+A B"),
                   no_evidence_markers()],
    },
    {
        "name": "computed-answer-has-no-leaked-markers",
        "ask": "convert A+B*C-D to postfix",
        "why": "A computed result is copied into the reply exactly as "
               "printed, unlike a file's text which gets paraphrased - "
               "so an evidence tag like [S1] survives verbatim if it is "
               "not stripped before the answer is shown.",
        "checks": [no_evidence_markers()],
    },

    # ---- Plain conversation ----
    {
        "name": "explanation-stays-in-chat",
        "ask": "explain how recursion works",
        "why": "Nothing to look up. Reaching for a capability here is "
               "just slower.",
        "checks": [stayed_in_chat(), longer_than(30)],
    },
    {
        "name": "conversational-find",
        "ask": "I always find reasons not to go",
        "why": "The bug that started the override audit: matched on the "
               "word 'find' and came back as a generated Python script.",
        "checks": [stayed_in_chat(), avoids("import ", "def ", "```")],
    },
    {
        "name": "conversational-weather",
        "ask": "I love this weather",
        "why": "Names a live topic and asks for nothing.",
        "checks": [stayed_in_chat()],
    },

    # ---- Not fabricating ----
    {
        "name": "unknowable-file",
        "ask": "what does my file zzqqxx_not_real.pdf say",
        "why": "There is no such file. Saying so is the only correct "
               "answer; describing its contents is the worst failure "
               "mode this project has.",
        "checks": [admits_ignorance(), no_invented_numbers()],
    },
    {
        "name": "obscure-fact",
        "ask": "what was the exact attendance at the 1923 Wembley FA Cup final",
        "why": "Genuinely disputed. Either look it up or say it is not "
               "certain - do not state a precise figure from memory.",
        "checks": [no_invented_numbers()],
    },

    # ---- Capabilities ----
    {
        "name": "date",
        "ask": "what is today's date",
        "why": "The model cannot know this. It has to ask the system.",
        "checks": [used("system.datetime", "web.search")],
    },
    {
        "name": "currency",
        "ask": "how much is 50 US dollars in indian rupees",
        "why": "Needs today's rate. Treated as a unit conversion it "
               "invented a plausible-looking rate.",
        "checks": [used("finance.exchange", "web.search")],
        "needs": "internet",
    },

    # ---- Finding documents by their contents ----
    #
    # These depend on what is actually on the machine, so they are
    # written against the folders rather than specific files: the check
    # is that a document was found and named, not which one.
    {
        "name": "semantic-names-the-file",
        "ask": "which of my files has my hostel payment in it",
        "why": "Filename search cannot answer this - the receipt is "
               "called DUP7369024.pdf. The answer has to name the file "
               "it found, since naming it is the whole question.",
        "checks": [used("filesystem.semantic_search"),
                   says(".pdf", ".docx", ".txt")],
        "needs": "documents",
    },
    {
        "name": "semantic-by-subject",
        "ask": "where did I write about linked lists",
        "why": "Describes contents, names no file. Also guards the "
               "grounding path: the filename has to be citable "
               "evidence, or naming it counts as unsupported and the "
               "answer is replaced by a raw passage from nowhere.",
        "checks": [used("filesystem.semantic_search"),
                   says(".pdf", ".docx", ".txt")],
        "needs": "documents",
    },

    # ---- Conversation memory ----
    {
        "name": "remembers-name",
        "ask": ["my name is Alex", "what is my name"],
        "why": "The second turn is answerable only from the first.",
        "checks": [says("Alex")],
    },
    {
        "name": "follow-up-keeps-subject",
        "ask": ["what's the weather in London", "and in Mumbai?"],
        "why": "The follow-up carries no subject of its own. It once "
               "reached for the clock instead.",
        "checks": [used("weather.current", "web.search"),
                   says("mumbai")],
        "needs": "internet",
    },
]


def run(agent, case):
    """Ask one case and check the answer. Returns (ok, failures, seconds)."""

    from core.agent import PROGRESS

    asks = case["ask"] if isinstance(case["ask"], list) else [case["ask"]]

    started = time.perf_counter()
    answer = ""

    # A multi-turn case is judged on its LAST answer; the earlier turns
    # are there to set it up.
    for question in asks:
        PROGRESS.start_turn()
        answer = agent.respond(question)

    seconds = time.perf_counter() - started

    failures = []

    for check in case["checks"]:
        ok, detail = check(answer, agent)

        if not ok:
            failures.append(detail)

    return not failures, failures, seconds, answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        help="which Athena mode to evaluate")
    parser.add_argument("--only", default=None,
                        help="run only cases whose name contains this")
    parser.add_argument("--offline", action="store_true",
                        help="skip cases that need the internet")
    parser.add_argument("--no-documents", action="store_true",
                        help="skip cases that need documents on this machine")
    args = parser.parse_args()

    cases = CASES

    if args.only:
        cases = [c for c in cases if args.only in c["name"]]

    if args.offline:
        cases = [c for c in cases if c.get("needs") != "internet"]

    if args.no_documents:
        cases = [c for c in cases if c.get("needs") != "documents"]

    if not cases:
        print("No cases matched.")
        return 1

    print(f"Athena quality evaluation - mode: {args.mode}, {len(cases)} case(s)")
    print("Each case starts a fresh conversation.\n")

    agent = Agent(mode=args.mode)

    passed, failed, elapsed = [], [], 0.0

    for case in cases:

        # Fresh conversation per case, or an earlier answer would be
        # sitting in the history influencing the next one.
        agent.new_conversation()

        try:
            ok, failures, seconds, answer = run(agent, case)

        except Exception as error:
            print(f"ERROR {case['name']}: {error}\n")
            failed.append(case["name"])
            continue

        elapsed += seconds

        print(f"{'PASS' if ok else 'FAIL'}  {case['name']}  ({seconds:.1f}s)")

        if ok:
            passed.append(case["name"])
        else:
            failed.append(case["name"])
            print(f"        why it matters: {case['why']}")

            for detail in failures:
                if detail:
                    print(f"        {detail}")

            trimmed = " ".join(answer.split())[:240]
            print(f"        answered: {trimmed}")

        print()

    print("=" * 55)
    print(f"{len(passed)} passed, {len(failed)} failed in {elapsed:.0f}s")

    if failed:
        print("\nFailed: " + ", ".join(failed))
        print("\nA local model misses one occasionally. Worry about a case "
              "that fails every run,\nor a run much worse than the last.")

    agent.shutdown()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
