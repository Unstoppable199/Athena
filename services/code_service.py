"""
Code capability.

Generates code from a specification and executes arbitrary
code snippets. Generation uses an LLM; execution runs in a separate
subprocess with a timeout, but it is not a security sandbox.
"""

import ast
import copy
import re
import tempfile
import os
from pathlib import Path

from config import CODE_MAX_TOKENS, WORKSPACE_DIR
from services.filesystem import FilesystemService
from services.python_runner import PythonRunnerService


# ----------------------------------------------------------------
# Repairing impossible character ranges
# ----------------------------------------------------------------
#
# Asked to tokenise an expression, the model reliably writes something
# like r"[A-Za-z0-9+-*/()]". That is not a typo it can be talked out of:
# the operators are being listed in the order a person says them, and
# the hyphen between "+" and "*" reads to Python as a RANGE from "+"
# (43) to "*" (42), which runs backwards. It raises
#
#   re.PatternError: bad character range +-* at position 7
#
# at runtime, so nothing catches it until the script is already running.
# Asked for the postfix of A+B, Athena answered with that traceback.
#
# The system prompt has told the model to put the hyphen last since the
# first time this happened, and the model still writes it this way. A
# rule the generator can forget is not a fix, so the code is repaired
# after generation instead: a hyphen that cannot possibly be a range is
# escaped, which is exactly what was meant.
_STRING_LITERAL = re.compile(
    r"""(?P<prefix>[rbuRBU]{0,2})(?P<quote>'''|\"\"\"|'|")(?P<body>.*?)(?<!\\)(?P=quote)""",
    re.DOTALL,
)

_CHARACTER_CLASS = re.compile(r"\[\^?(?:\\.|[^\]\\])*\]")


def _escape_impossible_ranges(char_class: str) -> str:
    """Escape hyphens in a character class that cannot be ranges."""

    out = []
    i = 0

    while i < len(char_class):

        ch = char_class[i]

        # Anything escaped is copied across untouched, so "\-" and
        # "\]" keep their meaning.
        if ch == "\\" and i + 1 < len(char_class):
            out.append(char_class[i:i + 2])
            i += 2
            continue

        if (ch == "-"
                and 0 < i < len(char_class) - 1
                and out and len(out[-1]) == 1
                and out[-1] not in "[^"):

            left = out[-1]
            right = char_class[i + 1]

            # A range only makes sense left-to-right. "a-z" is fine;
            # "+-*" is backwards and is what Python rejects.
            if right not in "]" and ord(left) > ord(right):
                out.append("\\-")
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


# ----------------------------------------------------------------
# Scripts that wait for input nobody is going to give them
# ----------------------------------------------------------------
#
# The script runs unattended: nothing on standard input, no arguments,
# no data files. Every value it needs is in the specification.
#
# The prompt has said so since a script asked for a train's distance
# and produced no answer. Told not to prompt, the model started reading
# the value out of a file instead - "could not find the input file
# 'infix_expression.txt'", a file that has never existed - which fails
# in exactly the same way for exactly the same reason.
#
# So it is checked rather than only asked for. Writing files stays
# allowed, and so does reading one the request actually named, which is
# why the spec is consulted before a read is called wrong.
_WAITS_FOR_INPUT = (
    (re.compile(r"\binput\s*\("), "calls input()"),
    (re.compile(r"\bsys\.argv\b"), "reads sys.argv"),
    (re.compile(r"\bsys\.stdin\b"), "reads sys.stdin"),
    (re.compile(r"\bargparse\b"), "expects command-line arguments"),
)

# open("x.txt") and open("x.txt", "r") - a read. Write and append
# modes are deliberately not matched: producing a file is a normal
# thing for these scripts to do.
_READS_A_FILE = re.compile(
    r"""\bopen\s*\(\s*['"](?P<name>[^'"]+)['"]\s*"""
    r"""(?:,\s*['"]r[b+]*['"]\s*)?\)""",
)


def waits_for_input(code: str, spec: str = "") -> str:
    """Why this script would sit waiting, or "" if it would not.

    The spec is passed so a program legitimately told to read a named
    file is not accused of inventing one.
    """

    for pattern, reason in _WAITS_FOR_INPUT:
        if pattern.search(code):
            return reason

    lowered = (spec or "").lower()

    for match in _READS_A_FILE.finditer(code):

        filename = match.group("name")

        # A file the request actually named is fair to read. One the
        # model invented to hold a value it was already given is not.
        if filename.lower() in lowered:
            continue

        return f"reads {filename!r}, which nothing provides"

    return ""


def repair_character_ranges(source: str) -> str:
    """Fix character classes the model wrote backwards.

    Applied only inside string literals, since that is where regexes
    live. A bare "[a-b]" elsewhere in Python is a list, and subtraction
    inside one is perfectly legal.
    """

    def fix_literal(match):
        body = _CHARACTER_CLASS.sub(
            lambda m: _escape_impossible_ranges(m.group(0)),
            match.group("body"),
        )

        return (match.group("prefix") + match.group("quote")
                + body + match.group("quote"))

    return _STRING_LITERAL.sub(fix_literal, source)


_EXTRA_FSTRING_PRINT_BRACE = re.compile(
    r"^(?P<prefix>\s*print\(\s*f(?P<quote>['\"])[^\r\n]*?(?P=quote))"
    r"\}(?P<suffix>\s*\)\s*(?:#.*)?)$",
    re.MULTILINE,
)

_BROKEN_SLIDE_ADD = re.compile(
    r"^(?P<prefix>\s*[A-Za-z_]\w*\s*=\s*)"
    r"prs\.slides\.add\s*,\s*"
    r"(?P<layout>(?:prs\.)?slide_layouts\[[^\]\r\n]+\]|[A-Za-z_]\w*)"
    r"\s*\)(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def _is_add_paragraph_call(node) -> bool:
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_paragraph"
        and len(node.args) == 1
        and not node.keywords
    )


class _PptxApiRepair(ast.NodeTransformer):
    """Correct two unambiguous python-pptx API mistakes."""

    def __init__(self):
        self.changed = False
        self.paragraph_number = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "slides"
        ):
            node.func.attr = "add_slide"
            self.changed = True
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not _is_add_paragraph_call(node.value) or len(node.targets) != 1:
            return node
        text_value = node.value.args[0]
        node.value.args = []
        text_target = ast.Attribute(
            value=copy.deepcopy(node.targets[0]),
            attr="text",
            ctx=ast.Store(),
        )
        text_assignment = ast.Assign(targets=[text_target], value=text_value)
        self.changed = True
        return [node, ast.copy_location(text_assignment, node)]

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        if not _is_add_paragraph_call(node.value):
            return node
        self.paragraph_number += 1
        variable = f"_athena_paragraph_{self.paragraph_number}"
        text_value = node.value.args[0]
        node.value.args = []
        create = ast.Assign(
            targets=[ast.Name(id=variable, ctx=ast.Store())],
            value=node.value,
        )
        set_text = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=variable, ctx=ast.Load()),
                    attr="text",
                    ctx=ast.Store(),
                )
            ],
            value=text_value,
        )
        self.changed = True
        return [ast.copy_location(create, node), ast.copy_location(set_text, node)]


def repair_pptx_api_usage(source: str) -> str:
    """Repair model-written python-pptx calls with one clear meaning."""

    candidate = _BROKEN_SLIDE_ADD.sub(
        lambda match: (
            match.group("prefix")
            + "prs.slides.add_slide("
            + match.group("layout")
            + ")"
            + match.group("suffix")
        ),
        source,
    )
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return candidate

    repair = _PptxApiRepair()
    tree = repair.visit(tree)
    if not repair.changed:
        return candidate
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


_EXACT_ARTIFACT_PATH_RE = re.compile(
    r"The finished .*?exact absolute path:\s*(?P<path>[^\r\n]+)",
    re.IGNORECASE,
)


class _ArtifactPathRepair(ast.NodeTransformer):
    """Keep generated Office builders on the user-supplied destination."""

    def __init__(self, expected: str):
        self.expected = expected
        self.basename = Path(expected).name.casefold()
        self.suffix = Path(expected).suffix.casefold()
        self.changed = False

    def visit_Constant(self, node):
        if not isinstance(node.value, str) or not self.suffix:
            return node
        value = node.value.strip()
        if (
            value.casefold().endswith(self.suffix)
            and (
                value.casefold() == self.basename
                or "\\" in value
                or "/" in value
            )
            and value != self.expected
        ):
            self.changed = True
            return ast.copy_location(ast.Constant(value=self.expected), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        # python-pptx, python-docx, and openpyxl all write with .save(path).
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
            and node.args
            and not (
                isinstance(node.args[0], ast.Constant)
                and node.args[0].value == self.expected
            )
        ):
            node.args[0] = ast.Constant(value=self.expected)
            self.changed = True
        return node


def enforce_exact_artifact_path(source: str, spec: str) -> str:
    """Make a generated Office builder use the exact requested path.

    The model once repaired a valid builder but duplicated one directory in
    the absolute destination. The process exited successfully while the file
    the user requested did not exist. The destination is deterministic input,
    so it is enforced after generation instead of left to language modelling.
    """

    match = _EXACT_ARTIFACT_PATH_RE.search(spec or "")
    if not match:
        return source
    expected = match.group("path").strip().strip("`\"'")
    if not Path(expected).suffix:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    repair = _ArtifactPathRepair(expected)
    tree = repair.visit(tree)
    if not repair.changed:
        return source
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def repair_common_generated_syntax(source: str) -> str:
    """Repair tightly-defined slips in model-written code.

    Gemma occasionally emits ``print(f'Value: {value}'})``: one extra
    brace after an otherwise complete f-string. We only touch source
    that already fails to parse, remove only that exact brace shape,
    and keep the candidate only if the whole program then parses. This
    never runs on literal code supplied by the user.
    """

    candidate = _EXTRA_FSTRING_PRINT_BRACE.sub(
        lambda match: match.group("prefix") + match.group("suffix"),
        source,
    )
    candidate = repair_pptx_api_usage(candidate)

    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        return source


CODE_GENERATION_SYSTEM_PROMPT = """
You write clean, correct, working code.

Rules:
- Output ONLY the code. No explanation, no markdown code fences,
  no commentary before or after.
- The code must be complete and runnable as-is.
- Prefer simple, readable code over clever tricks.
- Save output files next to the script, using an absolute path built
  from the script's own location, not a bare relative filename.
- If the specification supplies an exact absolute output path, that
  path overrides the rule above. Use it verbatim, create its parent
  directory if needed, and do not also save a copy beside the script.
- When the task is to compute an answer, print the intermediate values
  as well as the result, each labelled on its own line.
- The script is given NO input of any kind. There is nothing on
  standard input, no command-line arguments, and no data file waiting
  to be read. Every value the program needs is already written in the
  specification above, and belongs in the code as a named variable.

  That means all of these are wrong, not just the first:
    input("Enter the expression: ")
    sys.argv[1]
    sys.stdin.read()
    open("infix_expression.txt").read()

  Asked for a train's average speed, a script that prompted for the
  distance produced no answer at all. Told not to prompt, a later one
  read the expression from "infix_expression.txt" - a file that has
  never existed - and failed the same way for the same reason. The
  rule is not about which function you call. It is that the data is
  in the specification and nowhere else.

  Writing files is different and is fine: a program asked to produce
  a document should write one.
- Never assume input is conveniently spaced or formatted. To read
  symbols out of text, match them with a regular expression rather
  than splitting on whitespace: "A + B + C*(D+E)" split on spaces
  leaves "C*(D+E)" as a single item, and everything after that is
  wrong.

  Match names and operators as SEPARATE alternatives, and never put a
  word boundary around a class that contains operators:

    re.findall(r"[A-Za-z_]\\w*|\\d+|[-+*/^()]", expression)   correct
    re.findall(r"\\b[A-Za-z0-9+\\-*/^]+\\b", expression)        wrong

  The second looks reasonable and returns ["A+B"] for "A+B" - one
  token that is neither a name nor an operator, so a loop testing for
  each finds neither, appends nothing, and prints an empty answer with
  no error anywhere.
- Inside a regular expression character class, a hyphen between two
  characters means a range. To match a literal minus sign, put it
  last: [+*/^-] is correct, [+-*/^] raises "bad character range" and
  the program does not run at all.
- When the task converts a number between bases or units, check the
  result by converting it BACK and comparing to what you started with.
  The inverse is a single call and is reliable: int("11111111", 2)
  returns 255, and if it does not return 255 the conversion is wrong.
  Print the digits only, not Python's own prefix - bin(255) returns
  "0b11111111" and the "0b" is not part of the answer.

  Do NOT attempt this for expression notation - postfix, prefix,
  infix. There is no one-call inverse: it needs an evaluator written
  from scratch, and the attempts produced eval("A B C * +"), which is
  a SyntaxError. The failing CHECK then printed itself next to a
  perfectly correct answer, so the reply carried the result and
  "VERIFICATION FAILED ... SyntaxError" together. A check that cannot
  be written correctly is worse than no check, because it turns right
  answers into confusing ones. Convert the expression and print it.

  This catches the mistake that otherwise looks like success: asked
  for the prefix of A+B, a script that quietly skipped the
  input-reversal step ran without error and printed the POSTFIX
  answer instead. Nothing about that run looked like a failure - the
  return code was 0 and nothing was written to stderr - which is
  exactly why the check has to be part of the program, not left to
  whoever reads its output.

  There are now exactly two acceptable outputs, and nothing between
  them:

  1. The check agrees. Print the result and NOTHING about the
     verification - no mention that a check ran or passed.
  2. The check disagrees, OR THE CHECK ITSELF FAILS TO RUN for any
     reason. Print exactly the line VERIFICATION FAILED, then the two
     values or the error, and stop. Do not also print the result you
     were about to verify.

  Wrap the whole verification step in one try/except that treats ANY
  exception - not only a numeric disagreement - as case 2. eval() on a
  string containing bare letters raises NameError unless a namespace
  supplies them: eval(expression, {}, {"A": 2, "B": 3}) evaluates
  "A+B" correctly; eval(expression) alone does not.

  A verification step that raises NameError and is not caught prints
  a message like "there was an error during verification" ALONGSIDE
  the computed result - so the reply reports success and failure in
  the same breath, which is worse than either alone. The rule is
  binary specifically to prevent that: report the answer, or report
  that verification failed, never a mixture of the two.

python-pptx, common mistakes that stop a deck being written:
- Add every slide with prs.slides.add_slide(layout). There is no
  prs.slides.add method, and never separate add_slide from prs.slides
  with a comma.
- add_paragraph() takes NO arguments. Write
  "p = tf.add_paragraph()" then "p.text = '...'" on the next line.
  "tf.add_paragraph('text')" raises TypeError and nothing is saved.
- Layout 1 ("Title and Content") has placeholders[1] for the body;
  layout 0 is the title slide and its placeholders[1] is the
  subtitle. Use layout 1 for any slide with bullet points.

python-docx: paragraph text goes through add_paragraph("text"), and
tables are built with add_table(rows, cols) before cells are filled.

openpyxl: write with ws.cell(row=r, column=c, value=v), where rows
and columns both start at 1, and save the workbook at the end.
"""


class CodeService:

    def __init__(self, model):

        self.model = model
        self.filesystem = FilesystemService()
        self.runner = PythonRunnerService()

    def generate(self, spec: str, path: str, overwrite: bool = False):

        try:
            if not path or not path.strip():

                return {
                    "success": False,
                    "error": "What would you like to name the file? Include an extension, e.g. 'answer.py'."
                }

            file = Path(path)

            if not file.suffix:
                file = file.with_suffix(".py")

            # A bare filename lands in the workspace rather than
            # wherever Athena happens to be running. Scripts written to
            # work something out are scratch, and they were piling up
            # in the project root - multiply_12_8.py, calculate.py,
            # speed_calculation.py - mixed in with the source. An
            # absolute path the user gave is left exactly as asked.
            if not file.is_absolute():
                file = WORKSPACE_DIR / file
                file.parent.mkdir(parents=True, exist_ok=True)

            path = str(file)


            # Asking before replacing matters for a file the user named
            # and expects to keep. Inside the workspace it is only ever
            # Athena's own scratch, so the question is meaningless - and
            # it stopped a request outright: asked for a percentage a
            # second time, the reply was "a file already exists at
            # calculate_percentage.py, shall I overwrite it?" instead of
            # the number.
            in_workspace = WORKSPACE_DIR in file.parents

            if file.exists() and not overwrite and not in_workspace:

                return {
                    "success": False,
                    "error": (
                        f"A file already exists at '{path}'. "
                        f"Would you like me to overwrite it, or save "
                        f"it under a different filename?"
                    ),
                    "conflict": True
                }

            code = self._write_program(spec)

            return self.filesystem.write(path, code, overwrite=True)

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }

    def _write_program(self, spec: str) -> str:
        """Generate a program, and reject one that waits for input.

        A script that asks for a value it was already given does not
        fail loudly - it sits there, or dies looking for a file nobody
        wrote, and the request comes back as an error about the file
        rather than an answer. The prompt has forbidden this since the
        first time it happened and the model still does it, so it is
        checked here as well.

        One retry. If the second attempt has the same problem then
        reading the complaint is not what solves it, and returning the
        code lets the error at least be a real one.
        """

        code = enforce_exact_artifact_path(
            repair_common_generated_syntax(repair_character_ranges(
            self._strip_fences(
                self.model.complete(
                    CODE_GENERATION_SYSTEM_PROMPT,
                    spec,
                    num_predict=CODE_MAX_TOKENS,
                    think=False,
                )
            )
            )),
            spec,
        )

        problem = waits_for_input(code, spec)

        if not problem:
            return code

        print(f"[CODE] generated script {problem} -> regenerating")

        retry = self.model.complete(
            CODE_GENERATION_SYSTEM_PROMPT,
            f"""{spec}

The previous attempt {problem}. Nothing will provide that: there is no
standard input, no arguments, and no such file. Every value is in the
description above. Put it directly in the code as a variable.""",
            num_predict=CODE_MAX_TOKENS,
            think=False,
        )

        return enforce_exact_artifact_path(
            repair_common_generated_syntax(
                repair_character_ranges(self._strip_fences(retry))
            ),
            spec,
        )

    @staticmethod
    def repair_generated_syntax(code: str) -> str:
        """Public hook used only for planner-generated calculations."""

        return repair_common_generated_syntax(repair_character_ranges(code))

    def repair_snippet(self, request: str, code: str, error: str) -> str:
        """Return one corrected, self-contained calculation snippet.

        This is deliberately an in-memory repair. A calculation should
        not leave a helper file behind just because the first generated
        snippet contained a typo. The original request, complete code,
        and observed failure are all included so the model can repair
        the cause instead of starting from an ambiguous error message.
        """

        spec = f"""Solve this request with a short, self-contained Python program:

{request}

The previous program was:

{code}

It failed this check:

{error}

Return the complete corrected program. Keep every value from the
request in the code, do not use input(), do not read or write files,
and print the final result clearly. This is the only repair attempt."""

        return self._write_program(spec)

    def run_snippet(self, code: str):

        tmp = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as f:

                f.write(code)
                tmp = f.name

            return self.runner.run(tmp)

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }

        finally:

            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def _strip_fences(text: str) -> str:

        text = text.strip()

        if text.startswith("```"):

            lines = text.split("\n")

            if lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines)

        return text.strip()
