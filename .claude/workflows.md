# Workflows: llama-cpp-autosizer

## Add a New Feature

1. Decide which layer it belongs to: `Models/` (data), `Services/` (logic/I/O), or `UI/` (presentation) — see [`.claude/conventions.md`](conventions.md).
2. If it's a new tunable llama-server parameter, add it to `Models/LlamaSettings.cs` (property + `ToServerArgs()` + `Fingerprint()` + `Summary()`), then to `Services/RecommendationService.cs`'s `ExplorationOrder` array and its apply logic — see [`.claude/domains/optimizer.md`](domains/optimizer.md).
3. Register any new service as a DI singleton in `src/LlamaCppAutosizer/Program.cs`.
4. Validate with `dotnet build` and a manual run.

## Add or Update a Test

Needs verification — no test project exists in this repo yet. If one is added, update this section and `.claude/project-map.md`'s Testing section.

## Trace a Bug

- Optimization loop issues: start at `Services/OptimizerService.cs` (`OptimizeAsync`), then check `Services/RecommendationService.cs` (parameter choice) and `Services/BenchmarkService.cs` (scoring inputs).
- Server won't start / crashes / wrong settings applied: `Services/LlamaServerService.cs` — check `LastEffectiveSettings`/`LastStartAdjustmentNote` since the service self-heals unsupported settings.
- Hardware misdetection: `Services/HardwareDetectionService.cs` (Windows vs Linux code paths differ).
- Wrong/missing saved data: check the single owning service for that file type — `SessionPersistenceService` (`sessions/`), `ProfileLibraryService` (`profiles/`), `AppSettingsService` (`autosizer-config.json`).

## Modify Data or State-Related Code

All state is flat JSON files (`sessions/`, `profiles/`, `autosizer-config.json`), each with one owning service (see [`.claude/conventions.md`](conventions.md) Data Access Conventions). Changing a model's JSON shape (e.g. `OptimizationSession`, `SavedProfile`) is a breaking change for existing files on disk — there is no migration mechanism, so consider backward compatibility of `System.Text.Json` deserialization (new optional fields are safe; renames/removals are not).

## Work With Configuration and Secrets

`autosizer-config.json` (gitignored) holds local machine paths only — no secrets. Do not commit it or embed real local paths into tracked files. There are no API keys or credentials anywhere in this repo (llama-server is an unauthenticated local subprocess).

## Update External Integrations

- **llama-server HTTP API**: DTOs and calls are centralized in `Services/LlamaServerService.cs`. Changing the API surface used means updating the request/response DTO classes at the top of that file.
- **TurboQuant**: subprocess invocation logic is centralized in `Services/TurboQuantService.cs`.

## Build and Run Locally

```bash
dotnet run --project src/LlamaCppAutosizer
```

Requires: .NET 10 SDK, a `llama-server` executable (path configured via the TUI or `autosizer-config.json`), and at least one local `.gguf` model file.

## Validate Before Commit

No automated test suite or CI exists. Minimum validation:
```bash
dotnet build src/LlamaCppAutosizer
```
For behavior changes, manually run the app (`dotnet run --project src/LlamaCppAutosizer`) and exercise the affected menu/flow against a real `llama-server`.

## Deployment or Release

Publish self-contained single-file executables per platform:
```bash
dotnet publish src/LlamaCppAutosizer -r win-x64 -c Release -o ./publish
dotnet publish src/LlamaCppAutosizer -r linux-x64 -c Release -o ./publish
dotnet publish src/LlamaCppAutosizer -r linux-arm64 -c Release -o ./publish
```
No CI/CD pipeline automates this — it's a manual local publish.

## Rollback or Recovery

Needs verification — no deployment/release automation exists to define a rollback procedure. Session/profile JSON files in `sessions/`/`profiles/` are the only persistent artifacts; recovery would mean restoring from a prior file backup since there is no versioning of these directories (they're gitignored).