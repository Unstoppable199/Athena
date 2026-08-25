# Security policy and boundaries

Athena is a local, single-user learning project. It has not undergone a formal
security audit and should not be treated as a hardened confidential-document
platform.

## Important boundaries

- The web server is unauthenticated. Keep it bound to `127.0.0.1`.
- Conversations are stored locally as unencrypted JSON.
- Public-research capabilities send queries and fetch pages over the internet.
- Retrieved pages and documents are untrusted input and may contain prompt
  injection or misleading instructions.
- Generated Python is executed with the current user's permissions. The timeout
  is not a sandbox and does not prevent filesystem or network access.
- Common secret files such as `.env`, private keys and credential files are
  refused, but this list cannot identify every sensitive file.
- Local terminal logs may contain paths, model prompts and extracted text.
- Static prompt source is intentionally public. This does not authorize
  exposing dynamically assembled prompts containing conversation memory,
  document evidence, credentials or other private runtime context.

Use disposable data and a restricted operating-system account when evaluating
code execution. Do not expose Athena directly to a network or use it for
unattended high-stakes decisions.

## Reporting a vulnerability

Please avoid opening a public issue containing private documents, credentials,
absolute personal paths or full logs. Use a private GitHub security advisory if
the repository has that feature enabled; otherwise provide a minimal sanitized
reproduction to the maintainer.

## Supported version

Only the latest revision on the default branch is maintained during active
development.
