/*
 * Run the page's pure functions for real.
 *
 * The only JavaScript check before this was `node --check`, which
 * parses and stops. It cannot see an undefined variable, because that
 * is a ReferenceError at run time and not a syntax error - which is
 * how `sessionTokens += computed` reached the browser and turned the
 * first message of a conversation into
 *
 *     Error: ReferenceError: computed is not defined
 *
 * The variable had been renamed to `read` and `written` a few edits
 * earlier; one line still referred to the old name, and nothing
 * looked at it until a person did.
 *
 * So the functions worth checking are pulled out of Athena's page script and
 * actually called, with the browser stubbed out. The script may be inline or
 * linked from /static; the checker follows either form. This only suits
 * functions that compute rather than touch the DOM - which is the
 * point, since those are the ones with logic to get wrong.
 *
 * Run by tests/test_regressions.py; also runs alone:
 *
 *     node tests/check_page_functions.js core/templates/index.html
 */

const fs = require("fs");
const path = require("path");

const pagePath = process.argv[2]
  || path.join(__dirname, "..", "core", "templates", "index.html");

const page = fs.readFileSync(pagePath, "utf8");

const inlineScripts = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)]
  .map((m) => m[1]);

const linkedScripts = [...page.matchAll(/<script\s+src=["']([^"']+)["'][^>]*><\/script>/g)]
  .map((m) => m[1].split("?")[0])
  .filter((src) => src.startsWith("/static/"))
  .map((src) => path.join(
    path.dirname(pagePath), "..", "static", path.basename(src)
  ))
  .map((file) => fs.readFileSync(file, "utf8"));

const script = [...inlineScripts, ...linkedScripts].join("\n");

const failures = [];

function check(name, run) {
  try {
    run();
  } catch (error) {
    failures.push(`${name}: ${error.constructor.name}: ${error.message}`);
  }
}

/* Lift one named function out of the page by matching its braces. */
function extract(name) {
  const start = script.indexOf(`function ${name}(`);

  if (start === -1) {
    throw new Error(`function ${name} not found in the page`);
  }

  let depth = 0;

  for (let i = script.indexOf("{", start); i < script.length; i++) {
    if (script[i] === "{") depth++;
    else if (script[i] === "}" && --depth === 0) {
      return script.slice(start, i + 1);
    }
  }

  throw new Error(`function ${name} has unbalanced braces`);
}

/* Everything renderReply reaches for that lives outside it. Kept
   deliberately small: a stub for something the function should not be
   using is a way to hide a real dependency. */
const STUBS = `
  let sessionTokens = 0;
  const sessionTokensEl = { textContent: "" };
  const rows = [];
  function addRow(text, who, opts) { rows.push({ text, who, opts }); }
  function beep() {}
  function fmtDuration(s) { return s.toFixed(1) + "s"; }
`;

check("renderReply", () => {
  const run = new Function(`
    ${STUBS}
    ${extract("renderReply")}

    /* A turn where the prompt was really read. */
    renderReply({
      response: "Hi there!", seconds: 1.1, tokens: 1395,
      prompt_tokens: 1384, output_tokens: 11, model_calls: 1,
      read_tokens: 1384, cached_tokens: 0
    }, 1.2);

    /* A turn served entirely from the prompt cache - the shape that
       made the old display contradict itself. */
    renderReply({
      response: "Cached", seconds: 0.9, tokens: 1994,
      prompt_tokens: 1975, output_tokens: 19, model_calls: 2,
      read_tokens: 0, cached_tokens: 1975
    }, 1.0);

    /* A stopped reply: no token counts at all. */
    renderReply({ response: "Stopped.", stopped: true, seconds: 0.2 }, 0.3);

    return { rows, sessionTokens, chip: sessionTokensEl.textContent };
  `);

  const out = run();

  if (out.rows.length !== 3) {
    throw new Error(`expected 3 replies rendered, got ${out.rows.length}`);
  }

  /* Read plus written, with the cached tokens left out: 1384 + 11
     from the first turn, 0 + 19 from the second. */
  if (out.sessionTokens !== 1414) {
    throw new Error(
      `session total should count real work only, got ${out.sessionTokens}`
    );
  }

  const cached = out.rows[1].opts.meta.join(" ");

  if (!cached.includes("reused")) {
    throw new Error(`a cached turn should say so: ${cached}`);
  }

  if (!out.rows[2].opts.stopped) {
    throw new Error("a stopped reply should be marked stopped");
  }
});

if (failures.length) {
  failures.forEach((f) => console.log("FAIL  " + f));
  process.exit(1);
}

console.log("OK");
