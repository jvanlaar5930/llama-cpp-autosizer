# Metrics Opt-In Prompt

Use this prompt when the user wants to enable repo-memory session metrics again.

## Instruction

For this session, print the repo-memory session metrics summary at the end of the task.

At the end of the session, read:

```text
.claude/prompts/session-metrics.md
```

Then print the metrics summary to the terminal/chat output.

Do not write metrics to disk.

Do not create metrics files.

Do not create telemetry, analytics, logs, databases, local reports, or network calls.

If exact token usage is unavailable, write `unknown`.

Continue following:

* `CLAUDE.md`
* `.claude/project-map.md`
* `.claude/architecture.md` when relevant
* `.claude/conventions.md` when relevant
* `.claude/workflows.md` when relevant
* `.claude/update-project-map.md` when relevant

This opt-in affects only terminal/chat metrics reporting.
