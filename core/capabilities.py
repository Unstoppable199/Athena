"""
Capability registry.

Defines every capability Athena has.
"""


CAPABILITIES = [
    {
        "type": "chat",
        "purpose": "General reasoning and conversation.",
        "when": [
            "general knowledge",
            "programming",
            "reasoning",
            "math",
            "writing",
            "brainstorming",
            "concepts",
        ],
    },

    {
        "type": "system.datetime",
        "purpose": (
            "Get the current date, time, and day of the week. "
            "Can also get the current date or time in another location."
        ),
        "when": [
            "current date",
            "current day",
            "current time",
            "what day is it",
            "what time is it",
            "day of the week",
            "time in another country or city",
            "timezone conversion",
        ],
        "args": [
            (
                "timezone (optional) - IANA timezone name, such as "
                "'Asia/Tokyo', 'America/New_York', or 'Asia/Kolkata'. "
                "Omit it to use the system timezone."
            ),
        ],
    },

    {
        "type": "weather.current",
        "purpose": (
            "Get current weather conditions for a place: temperature, "
            "what it feels like, humidity, precipitation and wind."
        ),
        "when": [
            "weather",
            "temperature",
            "how hot or cold it is",
            "humidity",
            "is it raining",
            "wind speed",
        ],
        "args": [
            (
                "location - the city, town or place name, such as "
                "'London', 'Rupnagar' or 'Tokyo'"
            ),
        ],
    },

    {
        "type": "finance.quote",
        "purpose": (
            "Get the latest share price for a ticker symbol, with the "
            "change since the previous close."
        ),
        "when": [
            "share price",
            "stock price",
            "how a stock is doing",
            "market value of a company",
        ],
        "args": [
            (
                "symbol - the ticker symbol, such as 'AAPL', 'MSFT' or "
                "'INFY.NS'. Work it out from the company name if needed."
            ),
        ],
    },

    {
        "type": "finance.exchange",
        "purpose": "Convert an amount between two currencies at today's rate.",
        "when": [
            "exchange rate",
            "currency conversion",
            "how much is one currency worth in another",
        ],
        "args": [
            "base - the currency being converted from, as a code such as 'USD'",
            "target - the currency being converted to, as a code such as 'INR'",
            "amount (optional) - how much to convert; defaults to 1",
        ],
    },

    {
        "type": "web.search",
        "purpose": (
            "Search the internet for current or externally verifiable "
            "information. Use this only when no more specific capability "
            "above covers the request - the specific ones return exact "
            "figures, where this returns pages that have to be read."
        ),
        "when": [
            "current information",
            "information that changes over time",
            "news",
            "latest versions",
            "live information",
            "sports results",
            "anything current with no dedicated capability above",
        ],
        "args": [
            "query - the main search query",
            (
                "queries (optional) - up to three focused search-query "
                "variants when one query may not provide enough evidence"
            ),
            (
                "category - one of 'sports', 'finance', 'weather', or "
                "'general'; use 'general' when none clearly apply"
            ),
        ],
    },

    {
        "type": "filesystem.list",
        "purpose": "List files and folders inside a directory.",
        "when": [
            "list files",
            "show folder contents",
            "what files are in a folder",
        ],
        "args": [
            "path - the directory to list",
        ],
    },

    {
        "type": "filesystem.exists",
        "purpose": "Check whether a file or directory exists at an exact path.",
        "when": [
            "check whether a file exists",
            "check whether a folder exists",
            "verify an exact path",
        ],
        "args": [
            "path - the exact file or directory path",
        ],
    },

    {
        "type": "filesystem.info",
        "purpose": (
            "Get information about a file or directory, such as its name, "
            "path, type, size, and modification time."
        ),
        "when": [
            "get file information",
            "get folder information",
            "check file size",
            "check modification time",
            "inspect file metadata",
        ],
        "args": [
            "path - the exact file or directory path",
        ],
    },

    {
        "type": "filesystem.read",
        "purpose": (
            "Read a file from the system, including plain text, source code, "
            "PDF, Word, and spreadsheet documents."
        ),
        "when": [
            "read a file",
            "inspect code",
            "summarize a document",
            "summarize the selected file",
            "read the selected file",
            "analyze the selected file",
            "open a PDF",
            "read a Word document",
            "read a spreadsheet",
            "answer a question about a file",
        ],
        "args": [
            "path - the exact path of the file to read",
        ],
    },

    {
        "type": "filesystem.search",
        "purpose": (
            "Find files anywhere on the system by full or partial filename "
            "when the exact path is unknown."
        ),
        "when": [
            "find a file",
            "locate a file",
            "look for a file",
            "check whether a named file exists",
            "the file is somewhere on my computer",
            "the exact path is unknown",
        ],
        "args": [
            "name - the full or partial filename, with or without an extension",
        ],
    },

    {
        "type": "filesystem.semantic_search",
        "purpose": (
            "Find a document by what is written inside it, rather than by "
            "its filename. Use this when the user describes the contents "
            "of a file but does not know what it is called - which is "
            "usual, since most documents are named things like "
            "Scan_20241203.pdf."
        ),
        "when": [
            "which file mentions something",
            "the document about a subject",
            "I don't remember what it's called",
            "the file where I wrote about something",
            "find my notes on a topic",
            "search my documents for a subject",
        ],
        "args": [
            "query - what the document is about, in the user's own words",
        ],
    },

    {
        "type": "python.run",
        "purpose": "Execute an existing Python script.",
        "when": [
            "run a Python file",
            "execute a Python script",
            "test a Python script",
        ],
        "args": [
            "path - the exact path of the Python script",
        ],
    },

    {
        "type": "code.generate",
        "purpose": (
            "Generate code from a description and save it to a file. "
            "To produce a document, spreadsheet or presentation, generate "
            "a SCRIPT that builds it. Athena automatically runs that script "
            "for artifact requests; the file this step writes is always the "
            "code, never the finished document."
        ),
        "when": [
            "write code",
            "generate a script",
            "create a program",
            "write a function",
            "build a script",
            "make a PowerPoint, Word document or spreadsheet",
        ],
        "args": [
            (
                "path - where the generated CODE should be saved; this "
                "always ends in a code extension such as '.py', never "
                "'.pptx', '.docx' or '.xlsx'"
            ),
            "spec - a clear description of what the code should do",
            (
                "overwrite (optional) - true only when the user explicitly "
                "allows an existing file to be overwritten"
            ),
        ],
    },

    {
        "type": "code.run",
        "purpose": (
            "Execute a code snippet directly without saving it first. "
            "This is also how any question that must be worked out is "
            "answered - write a short program that prints the answer "
            "and run it, rather than reasoning it out and reporting "
            "the result."
        ),
        "when": [
            "run this code",
            "execute this snippet",
            "try this code",
            "test this code",
            "maths, physics, chemistry or engineering problems",
            "unit conversions, statistics, dates and durations",
            "algorithms, including infix to postfix or prefix",
            "anything needing exact calculation rather than recall",
        ],
        "args": [
            "code - the exact code to execute",
        ],
    },
]
