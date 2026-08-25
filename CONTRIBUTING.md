# Contributing

Athena is developed from real conversation failures. Small, evidence-backed
changes are preferred over broad prompt rewrites or capability-specific hacks.

## Before changing code

1. Reproduce the failure with the exact conversation and selected mode.
2. Add a deterministic regression when the failure is structural.
3. Preserve grounding, privacy and the contracts between router, planner,
   executor, prompt builder and agent.
4. Do not include personal documents, conversation files or credentials.

## Checks

```powershell
python -m compileall -q core models services main.py
node --check core/static/athena.js
node tests/check_page_functions.js
python tests/test_regressions.py
```

Real-model checks are intentionally manual:

```powershell
python tests/eval_quality.py
python tests/deep_test.py
```

Describe model-dependent results as observations rather than deterministic
guarantees.
