# Architecture: llama-cpp-autosizer

## High-Level Architecture

Single-project .NET console/TUI app with classic constructor-injected service layer:

```
Program.cs (composition root)
  -> UI/MainMenu (Spectre.Console TUI, top-level nav)
       -> UI/ProfileMenu, UI/AppSettingsMenu, UI/SettingsEditor, UI/BenchmarkDisplay
       -> Services/OptimizerService (orchestrates the tuning loop)
            -> Services/HardwareDetectionService
            -> Services/LlamaServerService   (spawns/controls llama-server subprocess + HTTP client)
            -> Services/BenchmarkService     (drives timed prompts against the running server)
            -> Services/RecommendationService (LLM-guided + heuristic next-parameter choice)
            -> Services/SessionPersistenceService (writes sessions/*.json)
       -> Services/ProfileLibraryService (profiles/*.json CRUD)
       -> Services/TurboQuantService (optional external quantization tool)
```

All services are registered as DI singletons in `Program.cs`; there is no separate application/domain layering beyond `Models` / `Services` / `UI`.

## Major Layers or Components

- **Models** — plain data classes/records: `LlamaSettings`, `HardwareInfo`, `BenchmarkResult`, `OptimizationProfile`, `OptimizationSession`, `SavedProfile`, `TurboQuantOptions`. No behavior beyond simple methods (e.g. `OptimizationProfile.ScoreResult()`).
- **Services** — all business logic: hardware probing, server process lifecycle, HTTP calls to llama-server, benchmarking, recommendation (LLM + heuristic), persistence, TurboQuant subprocess calls.
- **UI** — Spectre.Console-based menus and live displays; no business logic beyond user interaction flow.

## Dependency Flow

`UI -> Services -> Models`. Services depend on each other via constructor injection (primary constructors, e.g. `OptimizerService(LlamaServerService, BenchmarkService, RecommendationService, HardwareDetectionService, SessionPersistenceService, ILogger<...>)`). No circular dependencies observed.

## Data Flow

1. User selects model + profile via TUI.
2. `HardwareDetectionService` scans VRAM/RAM/CPU.
3. `OptimizerService` starts baseline: `LlamaServerService` spawns `llama-server`, polls `/health`.
4. `BenchmarkService` sends timed prompts to the server's HTTP API and records `BenchmarkResult` (PP speed, TG speed, TTFT).
5. `OptimizationProfile.ScoreResult()` produces a composite score per profile's weighting.
6. `RecommendationService` asks the *running model itself* (via `/v1/chat/completions`) for the next parameter to change, parsing a JSON response; falls back to heuristic `ExplorationOrder` walk if the LLM doesn't cooperate or repeats an already-tested config.
7. `LlamaServerService` restarts the server with the new setting (self-healing retry if unsupported).
8. Loop until score improvement < threshold for N consecutive rounds or iteration cap hit.
9. `SessionPersistenceService` writes the full run to `sessions/*.json`; user may save the winning config as a named profile via `ProfileLibraryService` (`profiles/*.json`).

## External Integrations

- **llama-server** (llama.cpp) — local subprocess, controlled via `System.Diagnostics.Process` and its HTTP API (`/completion`, `/v1/chat/completions`, `/health`). Not a remote service.
- **TurboQuant** (`llama-cpp-turboquant` fork) — optional external `llama-server`/quantization binary, invoked as a subprocess for extra KV cache types (`turbo4`/`turbo3`/etc.) and model quantization.
- **nvidia-smi` / `rocm-smi`** — shelled out to for GPU VRAM detection.
- **wmic` / `System.Management`** (Windows) and **`/proc/meminfo`, `/proc/cpuinfo`** (Linux) — hardware detection.

## Background, Scheduled, or Async Work

No background/scheduled jobs. The optimization loop is a long-running `IAsyncEnumerable<OptimizationIteration>` driven interactively from the TUI, cancellable via `CancellationToken` (wired to Ctrl+C in `Program.cs`).

## Data Storage and State Management

Flat JSON files only, via `System.Text.Json`:
- `sessions/*.json` — full run history (gitignored)
- `profiles/*.json` — saved named configs, GUID filenames (gitignored)
- `autosizer-config.json` — last-used local paths/profile (gitignored)

No database.

## Error Handling and Logging

- `Microsoft.Extensions.Logging` with console provider, minimum level `Warning` by default (to avoid cluttering the TUI); errors surface to stderr.
- `LlamaServerService` implements self-healing: if the server fails to start with a given setting (e.g. quantized KV cache without flash attention, mmap failure), it automatically retries once with that setting reverted rather than failing the whole run. Callers must read `LastEffectiveSettings`/`LastStartAdjustmentNote` to know what actually applied.
- `RecommendationService` wraps LLM recommendation calls in try/catch and falls back to heuristics on any exception.

## Security and Secrets Handling

No secrets/API keys in the repo — llama-server is a local, unauthenticated subprocess. `autosizer-config.json` and the `sessions/`/`profiles/` directories are gitignored because they contain local absolute file paths, not credentials.

## Performance or Scalability Notes

N/A — single-user local desktop tool, not a scaled service. Optimization loop cost is bounded by `OptimizationOptions.MaxIterations` (default 20) and convergence patience (default 3).

## Architectural Risks or Unclear Areas

- No automated test project exists — `Needs verification` whether one is planned.
- No CI/CD configuration found.
