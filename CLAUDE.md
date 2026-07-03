<!-- BEGIN create-claude-repo-memory -->
# Claude Instructions for llama-cpp-autosizer

Before answering project-specific questions:

1. Read `.claude/project-map.md`.
2. Use the project map to identify the relevant domain, files, and conventions.
3. Read only the specific files needed for the task.
4. Do not scan generated, build, cache, vendor, dependency, or IDE folders unless explicitly asked.
5. Prefer existing project patterns over introducing new architecture.
6. When changing code, identify tests or validation steps that should be added or updated.
7. When discovering durable project knowledge, update the most specific relevant file under `.claude/`.

## Project Memory Workflow

Use this order for project-specific work:

1. Read `.claude/project-map.md`.
2. If the task is domain-specific, read the relevant file under `.claude/domains/` if one exists.
3. Read `.claude/conventions.md` before making code-style or pattern decisions.
4. Read `.claude/workflows.md` before changing build, test, deployment, release, data, or integration workflows.
5. Read source files only after narrowing the task to the relevant area.

## Project Memory Update Rule

After making changes, decide whether the change affects durable project knowledge.

Update `.claude/` memory files only when the change affects:

- repository structure
- architecture
- major domains
- entry points
- dependency flow
- data access
- external integrations
- build, test, deployment, or release workflow
- coding conventions
- operational concerns
- known gotchas

Do not update project memory for small local implementation changes.

When updating memory:

1. Update the most specific file possible.
2. Keep `.claude/project-map.md` concise.
3. Prefer domain files under `.claude/domains/` for detailed notes.
4. Do not duplicate source code.
5. Do not include secrets.
6. If two memory updates conflict, preserve the facts that match current code and remove stale statements.

## Do Not Scan By Default

Do not scan these unless directly relevant:

- `.git/`
- dependency folders
- build output folders
- generated files
- package/cache folders
- IDE/editor folders
- coverage/report folders
- local environment folders

Common examples include:

- `node_modules/`
- `vendor/`
- `bin/`
- `obj/`
- `dist/`
- `build/`
- `target/`
- `out/`
- `.next/`
- `.nuxt/`
- `.turbo/`
- `.cache/`
- `.pytest_cache/`
- `__pycache__/`
- `.venv/`
- `venv/`
- `.gradle/`
- `.idea/`
- `.vscode/`
- `.vs/`
- `coverage/`

## Project Memory Files

Claude-specific project context is stored in:

- `.claude/project-map.md`
- `.claude/architecture.md`
- `.claude/conventions.md`
- `.claude/workflows.md`
- `.claude/update-project-map.md`
- `.claude/domains/`

## Metrics Opt In / Opt Out

Users can opt out of terminal-only metrics for a session with:

```text
Read .claude/prompts/metrics-opt-out.md and follow it for this session.
```

Users can opt back in with:

```text
Read .claude/prompts/metrics-opt-in.md and follow it for this session.
```

Opting out of metrics does not disable repo memory. It only disables the terminal/chat metrics summary.
<!-- END create-claude-repo-memory -->
