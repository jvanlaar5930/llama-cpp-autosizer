# Domain: Optimizer (iterative benchmark/tune loop)

## Purpose

Core feature of the app: automatically finds the best `llama-server` settings for a given GGUF model by repeatedly benchmarking and adjusting one parameter at a time until scores converge.

## Key Files

- `src/LlamaCppAutosizer/Services/OptimizerService.cs` — the loop itself. `OptimizeAsync(...)` returns `IAsyncEnumerable<OptimizationIteration>` so the TUI can render progress live. Caller supplies the `OptimizationSession` object (avoids duplicating history state).
- `src/LlamaCppAutosizer/Services/RecommendationService.cs` — decides the next parameter to change. Tries the LLM first (via `LlamaServerService`, `/v1/chat/completions`, JSON response), falls back to a fixed heuristic order (`ExplorationOrder`) on failure or if the LLM suggests a config that's already been tested.
- `src/LlamaCppAutosizer/Services/BenchmarkService.cs` — runs the actual timed prompts (warmup + benchmark + optional tool/agent-loop/accuracy/stress-test cases) against the running server and produces a `BenchmarkResult`.
- `src/LlamaCppAutosizer/Models/OptimizationProfile.cs` — defines `Chat` and `Agentic` built-in profiles: benchmark prompts, tool definitions, accuracy prompts, agent-loop cases, and `ScoringWeights`. `ScoreResult(BenchmarkResult)` computes the composite score the optimizer maximizes.
- `src/LlamaCppAutosizer/Models/LlamaSettings.cs` — the tunable parameter set. `ToServerArgs()` converts to CLI args, `Fingerprint()` gives a canonical identity string used to detect and skip duplicate configurations, `Clone()` for per-iteration mutation.
- `src/LlamaCppAutosizer/Models/OptimizationSession.cs` — `ParameterChange`, `OptimizationIteration`, `OptimizationSession` — the history/result records persisted at the end of a run.

## Baseline planning (`OptimizerService.BuildInitialPlan`)

The starting config is **computed per model from its GGUF header**, not from file-size tiers.
`BuildInitialPlan` returns a `BaselinePlan(Settings, Explanation)`; `BuildInitialSettings` is a
thin wrapper kept for callers that only want the settings.

Order of priorities, which is also the order the code searches:

1. **Context** — walk a descending ladder from `profile.TargetContextSize`
   (`OptimizationProfile.DefaultTargetContext` = 65,536), capped by the model's own
   `context_length`.
2. **GPU residency** — at each context, spend whatever VRAM the KV cache doesn't need on weights.
3. **KV precision** — f16 is kept unless q8_0 moves at least `MinResidencyGainForKvQuant` (10%)
   more of the model into VRAM. Resident weights buy more speed than KV precision buys quality,
   but only when the gain is real.

Key points:

- `GgufMetadata.KvCacheBytes()` sizes the KV cache layer by layer, honouring per-layer KV-head
  counts and sliding-window layers. This is why a Gemma-style SWA model costs ~1.6 GB at 64K
  while a same-size non-SWA MoE costs ~6 GB — the naive `2 × layers × heads × dim × ctx` formula
  is wrong by 4× for SWA models and would needlessly shrink the context.
- For expert-dominated MoE (`ExpertShareOfLayer() >= 0.5`) the planner sets `GpuLayers = -1` and
  uses `NCpuMoe` (`--n-cpu-moe`) to park expert FFN weights on the CPU, keeping all attention on
  the GPU. Lowering `GpuLayers` instead would evict whole layers, attention included, which is
  exactly the wrong thing at long context. Unlike `MoeExpertUsed` this costs no quality.
- A manually chosen KV cache type must be passed in as `preferredKvType`, not applied to the
  returned settings — the layer budget is computed against the KV footprint, so overriding it
  afterwards leaves `GpuLayers`/`NCpuMoe` sized for a cache that isn't the one being used.
- If the header can't be parsed, `LegacyGpuBaseline` reproduces the old file-size tier behaviour
  unchanged.
- The CPU-only path still uses the RAM tier table and does **not** chase the 65,536 target.

## Main Flows

1. **Baseline** — `HardwareDetectionService.DetectAsync()` runs, `LlamaServerService` starts the server with initial settings (derived from hardware + model file size), `BenchmarkService.RunAsync()` produces the baseline `BenchmarkResult`, scored via `profile.ScoreResult()`.
   - Note: the server may self-heal an unsupported initial setting (e.g. quantized KV cache without flash attention). The optimizer must use `server.LastEffectiveSettings` (not the requested `initialSettings`) as the true baseline going forward.
2. **Recommend** — `RecommendationService.GetNextRecommendationAsync(session, profile, hardware, ct)`:
   - If the server is running and at least 1 iteration has completed, asks the model itself for the next `ParameterChange` as JSON.
   - Rejects LLM suggestions that reproduce an already-tested config (`RecreatesTestedConfig`, uses `LlamaSettings.Fingerprint()`), falling back to heuristic in that case.
   - Heuristic fallback walks `ExplorationOrder`: `GpuLayers, FlashAttention, CacheTypeK, CacheTypeV, BatchSize, UBatchSize, ContextSize, Threads, Mlock`.
3. **Apply and re-benchmark** — settings cloned + mutated with the one parameter change, server restarted (self-healing retry if the new setting fails to start), benchmarked again, scored.
4. **Convergence check** — stops when score improvement drops below `OptimizationOptions.ConvergenceThreshold` (default 0.01) for `ConvergencePatience` consecutive rounds (default 3), or `MaxIterations` is hit (default 20). `VerifyBestAtEnd` (default true) re-runs the best config once more before finalizing.
5. **Persist** — `SessionPersistenceService` writes the completed `OptimizationSession` to `sessions/<model>_<profile>_<timestamp>.json`.

## Data and State

- In-memory: `OptimizationSession` accumulates `OptimizationIteration` entries as the loop runs; the same object is used by the UI for live display and by persistence at the end (no duplication).
- On disk: `sessions/*.json` (full history), optionally `profiles/*.json` if the user saves the winning config as a named profile afterward (`ProfileLibraryService`, separate from the optimizer itself).

## External Systems

- `llama-server` subprocess (via `LlamaServerService`) — started/stopped once per iteration.
- The model being tuned is also used as the recommender (self-referential: the model under test suggests its own next tuning step via chat completion).

## Tests and Validation

Needs verification — no automated tests exist for the optimizer loop. Validate by running a full optimization (`dotnet run --project src/LlamaCppAutosizer`, menu option 5) against a real model and checking the resulting `sessions/*.json` and printed summary/command.

## Known Gotchas

- Always read `LlamaServerService.LastEffectiveSettings` after a start attempt rather than assuming the requested `LlamaSettings` took effect — self-healing retries silently revert unsupported values.
- Adding a new tunable parameter requires three coordinated edits: `LlamaSettings` (property + `ToServerArgs`/`Fingerprint`/`Summary`), `RecommendationService.ExplorationOrder` (heuristic order), and wherever the LLM-recommendation JSON schema/prompt enumerates valid parameter names — missing one causes the optimizer to silently never explore the new parameter via one of the two recommendation paths.
- `OptimizationProfile.ScoreResult()` weights differ significantly between Chat and Agentic — a change to scoring normalization (`NormalizeRate`/`NormalizeTtft` bounds) affects both profiles; check both when tuning scoring behavior.
- Scoring bounds are **calibrated per run**, not fixed. `OptimizerService` calls `profile.CalibratedTo(baselineResult)` immediately after the baseline benchmark and reassigns its local `profile`, so baseline and every later iteration share one scale (no rescoring of history needed). Without this, fast hardware saturates the TG/PP/TTFT terms at 1.0 at baseline and the composite becomes completely insensitive to further speed gains — the optimizer then reports "no improvement" for changes that genuinely raised throughput. Bounds only ever widen from the defaults, so slow hardware scores as before.
  - `CalibratedTo` returns a **copy** (`record` `with`). This matters: `MainMenu._profile` is a long-lived field reused across runs, so mutating in place would leak one run's bounds into the next.
  - Anything that scores a `BenchmarkResult` outside the optimizer loop (e.g. a standalone benchmark) uses the uncalibrated defaults — scores from such runs are not comparable with scores from an optimization session.
- The MoE expert lever (`MoeExpertUsed`) was a silent no-op until the `--override-kv` key was corrected to the architecture-scoped `{arch}.expert_used_count`. If `GgufMetadataService` cannot read the header, `ToServerArgs` now emits nothing for it — but `Fingerprint()` still includes `MoeExpertUsed`, so in that (rare) case the optimizer can treat two identical server configs as distinct.
