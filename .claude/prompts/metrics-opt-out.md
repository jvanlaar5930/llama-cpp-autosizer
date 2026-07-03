# Metrics Opt-Out Prompt

Use this prompt when the user wants to skip repo-memory session metrics.

## Instruction

For this session, do not print the repo-memory session metrics summary unless the user explicitly asks for it again.

Do not read `.claude/prompts/session-metrics.md` automatically.

Do not report token usage, file counts, search counts, validation counts, irrelevant file counts, or memory-maintenance counts unless directly requested.

Continue following:

- `CLAUDE.md`
- `.claude/project-map.md`
- `.claude/architecture.md` when relevant
- `.claude/conventions.md` when relevant
- `.claude/workflows.md` when relevant
- `.claude/update-project-map.md` when relevant

This opt-out affects only metrics reporting.

It does not disable repo memory.

It does not disable project-map usage.

It does not disable memory-file updates when durable project knowledge changes.

Do not write this preference to disk unless the user explicitly asks.
