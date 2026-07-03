# Conventions: llama-cpp-autosizer

## Naming Conventions

- Standard C# PascalCase for types/members, camelCase for locals/parameters.
- Services are suffixed `*Service` (e.g. `BenchmarkService`, `OptimizerService`) and registered as DI singletons in `Program.cs`.
- UI classes are suffixed `*Menu` (top-level or sub-navigation) or descriptive (`SettingsEditor`, `BenchmarkDisplay`, `MenuHelper`).
- Primary constructors are used throughout services for DI (e.g. `public class OptimizerService(LlamaServerService server, BenchmarkService benchmarks, ...)`), not field + constructor-body assignment.

## Project Organization

Flat three-folder split, no deeper layering:
- `Models/` — data only (classes/records), plus small pure helper methods on the model itself (e.g. `LlamaSettings.ToServerArgs()`, `.Fingerprint()`, `.Clone()`; `OptimizationProfile.ScoreResult()`).
- `Services/` — all business logic and I/O (process control, HTTP, hardware probing, file persistence).
- `UI/` — Spectre.Console presentation only.

New functionality should fit into this structure rather than introducing new top-level folders unless a genuinely new domain is added.

## Dependency Management

- NuGet package versions are pinned with wildcard minor (`0.49.*`, `9.*`) in the `.csproj`.
- `System.Management` is conditionally referenced only on Windows (`Condition="$([MSBuild]::IsOSPlatform('Windows'))"`).
- No dependency injection container beyond `Microsoft.Extensions.DependencyInjection`; everything is a singleton — there is no scoped/transient usage.

## Application Structure

- `Program.cs` is the single composition root: registers logging, HttpClient factory, all services and UI classes, then resolves and runs `MainMenu`.
- Long-running work (the optimization loop) is exposed as `IAsyncEnumerable<T>` so the UI can stream progress rather than the service pushing to the UI directly.

## Testing Conventions

Needs verification — no test project exists in the repo as of this scan. There is no `dotnet test` target.

## Logging Conventions

- `Microsoft.Extensions.Logging` injected via constructor (`ILogger<T>`), console provider, default minimum level `Warning` (set in `Program.cs`) to keep the TUI clean; services log more verbosely at `Debug`/`Information` for cases the TUI doesn't need to show directly (e.g. `RecommendationService` logs LLM fallback reasons at `Debug`).

## Error Handling Conventions

- Prefer graceful degradation over throwing: e.g. `RecommendationService` catches LLM call failures and falls back to heuristic recommendations; `LlamaServerService` retries once with an adjusted setting rather than failing the whole optimization run on an unsupported config.
- `OperationCanceledException` from `CancellationToken` (wired to Ctrl+C) is caught at the top level in `Program.cs` for a clean exit, not treated as an error.

## Data Access Conventions

- All persistence is `System.Text.Json` serialization to flat files — no ORM, no database.
- `SessionPersistenceService` owns `sessions/*.json`; `ProfileLibraryService` owns `profiles/*.json`; `AppSettingsService` owns `autosizer-config.json`. Each persistence concern has exactly one owning service — don't read/write these files directly from UI or other services.

## API or Interface Conventions

- Communication with `llama-server` is exclusively via its HTTP API (`/completion`, `/v1/chat/completions`, `/health`, `/metrics`) through `LlamaServerService` — never invoke the llama.cpp CLI directly for inference.
- DTOs for the llama-server API live at the top of `LlamaServerService.cs` (`CompletionRequest`, `ChatCompletionRequest`, etc.) rather than in `Models/`, since they're wire-format specific to that one integration.

## Frontend or UI Conventions

- All TUI rendering goes through Spectre.Console (`AnsiConsole`, markup strings like `[yellow]...[/]`). Menus support Escape to go back and Ctrl+C to exit cleanly.

## Infrastructure Conventions

- No infrastructure-as-code; this is a locally-run desktop tool. Publishing targets self-contained single-file executables for `win-x64`, `linux-x64`, `linux-arm64`.

## Documentation Conventions

- `README.md` is the primary user-facing reference (features, usage, optimizer algorithm, output format) — keep it in sync with behavior changes visible to end users.
- `.claude/*.md` files are Claude-specific durable project memory, not user documentation — keep them concise and update per `.claude/update-project-map.md` rules.