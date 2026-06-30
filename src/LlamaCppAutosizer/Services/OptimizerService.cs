using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public record OptimizationOptions(
    int MaxIterations = 20,
    double ConvergenceThreshold = 0.01,
    int ConvergencePatience = 3,
    int Port = 8080
);

public class OptimizerService(
    LlamaServerService server,
    BenchmarkService benchmarks,
    RecommendationService recommender,
    HardwareDetectionService hardware,
    SessionPersistenceService persistence,
    ILogger<OptimizerService> logger)
{
    /// <summary>
    /// Main optimization loop. Yields each completed iteration so the UI can
    /// display live progress. Call StopAsync() on the token to abort early.
    /// </summary>
    /// <summary>
    /// Main optimization loop. The caller provides the <paramref name="session"/> so that
    /// CompletionReason, IsComplete, and iteration history all land on the same object used
    /// for the final display — no session duplication.
    /// </summary>
    public async IAsyncEnumerable<OptimizationIteration> OptimizeAsync(
        string serverExecutable,
        string modelPath,
        LlamaSettings initialSettings,
        OptimizationProfile profile,
        OptimizationSession session,
        OptimizationOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        options ??= new OptimizationOptions();
        var hw = await hardware.DetectAsync();
        session.Hardware = hw;

        try
        {
            // ── Iteration 0: baseline ─────────────────────────────────────────
            logger.LogInformation("Starting baseline benchmark");
            await StartServerAsync(serverExecutable, modelPath, initialSettings, options.Port, ct);
            // Server may have auto-reverted an unsupported setting (e.g. mmap, quantized KV
            // cache) to get started — use what's actually running, not what was requested.
            var effectiveInitialSettings = server.LastEffectiveSettings ?? initialSettings;
            string? initialAdjustment = server.LastStartAdjustmentNote;

            var baselineResult = await benchmarks.RunAsync(effectiveInitialSettings, modelPath, profile, ct);
            baselineResult.CompositeScore = profile.ScoreResult(baselineResult);
            baselineResult.Notes = "Baseline";

            var baseline = new OptimizationIteration
            {
                Number = 0,
                Settings = effectiveInitialSettings.Clone(),
                Result = baselineResult,
                AppliedChange = null,
                IsBestSoFar = true,
                StatusMessage = initialAdjustment is null
                    ? "Baseline — establishing reference score"
                    : $"Baseline — establishing reference score (auto-adjusted: {initialAdjustment})",
            };
            session.AddIteration(baseline);
            await persistence.SaveAsync(session);

            yield return baseline;

            // ── Iterative tuning ──────────────────────────────────────────────
            // consecutiveNonImprovements: resets on any iteration that beats the best score.
            // When it hits 3 we ask the LLM for a completely fresh angle ("final push").
            // If the LLM says DONE → stop early. Otherwise reset the counter and continue.
            // We never stop just because the normal recommendation pool is exhausted —
            // instead we trigger the final-push path there too.
            int consecutiveNonImprovements = 0;
            int consecutiveStartFailures = 0;
            const int MaxConsecutiveNonImprovements = 3;
            const int MaxConsecutiveStartFailures = 5;

            for (int iter = 1; iter <= options.MaxIterations && !ct.IsCancellationRequested; iter++)
            {
                // ── Pick the next change to try ───────────────────────────────
                ParameterChange? change;

                if (consecutiveNonImprovements >= MaxConsecutiveNonImprovements)
                {
                    // 3 consecutive non-improvements: present best to LLM and ask for a new angle
                    consecutiveNonImprovements = 0;
                    logger.LogInformation("3 consecutive non-improvements — asking LLM for final push");

                    change = await recommender.GetFinalPushAsync(session, profile, hw, ct);
                    if (change is null)
                    {
                        session.CompletionReason = "LLM confirmed no further improvements possible";
                        break;
                    }
                }
                else
                {
                    change = await recommender.GetNextRecommendationAsync(session, profile, hw, ct);
                    if (change is null)
                    {
                        // Normal pool exhausted — try the final push before giving up
                        logger.LogInformation("Recommendation pool empty — trying final-push LLM prompt");
                        change = await recommender.GetFinalPushAsync(session, profile, hw, ct);
                        if (change is null)
                        {
                            session.CompletionReason = "All parameters explored — LLM confirmed nothing further to try";
                            break;
                        }
                        consecutiveNonImprovements = 0;
                    }
                }

                var nextSettings = RecommendationService.Apply(session.BestSettings!, change);
                nextSettings.Label = $"iter{iter}";

                logger.LogInformation("Iteration {N}: {Param} → {Val} [{Source}]",
                    iter, change.Parameter, change.NewValue, change.Source);

                // ── Start server with new settings ────────────────────────────
                await StopServerAsync();
                string? startFailReason = null;
                string? startAdjustment = null;
                bool fatalStartFailure = false;
                try
                {
                    await StartServerAsync(serverExecutable, modelPath, nextSettings, options.Port, ct);
                    consecutiveStartFailures = 0;
                    // Server may have auto-reverted an unsupported setting (e.g. quantized KV
                    // cache rejected without flash-attn) to get started — record what actually ran.
                    nextSettings = server.LastEffectiveSettings ?? nextSettings;
                    startAdjustment = server.LastStartAdjustmentNote;
                }
                catch (Exception ex)
                {
                    startFailReason = ex.Message;
                    consecutiveStartFailures++;
                    logger.LogInformation(
                        "Server failed to start with {Param}={Val} ({Reason})",
                        change.Parameter, change.NewValue, ex.Message);

                    // Restore best settings so the server stays available for LLM prompts
                    try { await StartServerAsync(serverExecutable, modelPath, session.BestSettings!, options.Port, ct); }
                    catch (Exception revEx)
                    {
                        logger.LogInformation("Could not restore best settings ({Reason}); stopping", revEx.Message);
                        session.CompletionReason = "Server failed to restart even on best settings";
                        fatalStartFailure = true;
                    }

                    if (!fatalStartFailure && consecutiveStartFailures >= MaxConsecutiveStartFailures)
                    {
                        session.CompletionReason = $"Server failed to start {consecutiveStartFailures} times in a row — stopping";
                        fatalStartFailure = true;
                    }
                }

                // Yield a visible skipped-iteration for start failures (outside the catch)
                if (startFailReason is not null)
                {
                    string sourceLabel = SourceTag(change.Source);
                    var skipped = new OptimizationIteration
                    {
                        Number = iter,
                        Settings = nextSettings,
                        Result = new BenchmarkResult(),
                        AppliedChange = change,
                        StatusMessage = $"[{sourceLabel}] {change.Parameter} → {change.NewValue}  [SKIP — server failed to start]",
                    };
                    session.AddIteration(skipped);
                    yield return skipped;
                    consecutiveNonImprovements++;
                    if (fatalStartFailure) break;
                    continue;
                }

                // ── Benchmark ─────────────────────────────────────────────────
                BenchmarkResult iterResult;
                try
                {
                    iterResult = await benchmarks.RunAsync(nextSettings, modelPath, profile, ct);
                    iterResult.CompositeScore = profile.ScoreResult(iterResult);
                }
                catch (Exception ex)
                {
                    logger.LogInformation("Benchmark failed ({Reason}); counting as non-improvement", ex.Message);
                    consecutiveNonImprovements++;
                    continue;
                }

                string tag = SourceTag(change.Source);
                string adjustmentSuffix = startAdjustment is null ? "" : $"  (auto-adjusted: {startAdjustment})";
                var iteration = new OptimizationIteration
                {
                    Number = iter,
                    Settings = nextSettings,
                    Result = iterResult,
                    AppliedChange = change,
                    StatusMessage = $"[{tag}] {change.Parameter} → {change.NewValue}  \"{change.Reasoning}\"{adjustmentSuffix}",
                };
                session.AddIteration(iteration);
                await persistence.SaveAsync(session);

                yield return iteration;

                // ── Track non-improvement streak ──────────────────────────────
                if (iteration.IsBestSoFar)
                    consecutiveNonImprovements = 0;
                else
                    consecutiveNonImprovements++;
            }
        }
        finally
        {
            // Always runs — even if an unexpected exception exits the loop
            session.IsComplete = true;
            session.CompletedAt = DateTime.UtcNow;
            session.CompletionReason ??= ct.IsCancellationRequested
                ? "Cancelled by user"
                : "Max iterations reached";

            await persistence.SaveAsync(session);
            await StopServerAsync();

            logger.LogInformation("Optimization complete. Best score: {Score:F3}", session.BestResult?.CompositeScore ?? 0);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static string SourceTag(string source) => source switch
    {
        "llm"      => "LLM",
        "llm-push" => "LLM★",
        _          => "Heuristic",
    };

    private async Task StartServerAsync(
        string exe, string model, LlamaSettings settings, int port, CancellationToken ct)
    {
        await server.StartAsync(exe, model, settings, port, ct);
    }

    private async Task StopServerAsync()
    {
        try { await server.StopAsync(); }
        catch (Exception ex) { logger.LogDebug("Error during server stop: {Msg}", ex.Message); }
    }

    /// <summary>
    /// Build a good initial set of settings using hardware info and model size.
    /// Uses a VRAM-tier preset table so the baseline is conservative enough to
    /// actually start, rather than computing GPU layers with rough math that can
    /// over-commit VRAM.
    /// </summary>
    public static LlamaSettings BuildInitialSettings(
        string modelPath, OptimizationProfile profile, HardwareInfo hw)
    {
        long modelSizeMb = new FileInfo(modelPath).Length / (1024 * 1024);

        // Rough layer count — used only to estimate MB-per-layer for ngl calculation.
        int estimatedLayers = modelSizeMb switch
        {
            < 2000  => 24,   // ~1B
            < 5000  => 32,   // ~3–7B
            < 10000 => 40,   // ~8–13B
            < 20000 => 60,   // ~14–30B
            _       => 80,   // ~34–70B+
        };

        double mbPerLayer = (double)modelSizeMb / estimatedLayers;

        int threads = hw.CpuCores > 0 ? hw.CpuCores : -1;

        bool isThinking = LlamaSettings.IsThinkingModel(modelPath);

        if (!hw.HasGpu)
        {
            // CPU-only: use RAM-based presets; no GPU offload
            var cpuPreset = CpuPreset(hw.RamFreeMb);
            return new LlamaSettings
            {
                GpuLayers      = 0,
                ContextSize    = Math.Min(cpuPreset.CtxSize, profile.TargetContextSize),
                BatchSize      = cpuPreset.Batch,
                UBatchSize     = cpuPreset.UBatch,
                Threads        = threads,
                ThreadsBatch   = threads,
                FlashAttention = false,
                Mmap           = true,
                ThinkingEnabled = isThinking ? false : null,
                Label          = "baseline",
            };
        }

        // GPU path — select preset tier from free VRAM
        var preset = GpuPreset(hw.FreeVramMb);

        // Calculate GPU layers: leave 'preset.ReserveMb' free for KV cache + overhead.
        long vramForWeights = Math.Max(0, hw.FreeVramMb - preset.ReserveMb);
        int maxLayersInVram = mbPerLayer > 0 ? (int)(vramForWeights / mbPerLayer) : estimatedLayers;

        int gpuLayers = maxLayersInVram >= estimatedLayers
            ? -1                  // entire model fits — offload all
            : maxLayersInVram;

        // Context: start from the tier default, respect profile cap.
        // If the model barely squeezes into VRAM (< 2 GB headroom), start conservatively.
        int contextSize = Math.Min(preset.CtxSize, profile.TargetContextSize);
        if (gpuLayers != -1)
        {
            long modelVramUsed = (long)(gpuLayers * mbPerLayer);
            long headroom = hw.FreeVramMb - modelVramUsed;
            if      (headroom < 1536) contextSize = Math.Min(contextSize, 2048);
            else if (headroom < 3072) contextSize = Math.Min(contextSize, 4096);
            else if (headroom < 6144) contextSize = Math.Min(contextSize, 8192);
        }

        return new LlamaSettings
        {
            GpuLayers      = gpuLayers,
            ContextSize    = contextSize,
            BatchSize      = preset.Batch,
            UBatchSize     = preset.UBatch,
            Threads        = threads,
            ThreadsBatch   = threads,
            FlashAttention = preset.FlashAttn,
            Mmap           = true,
            ThinkingEnabled = isThinking ? false : null,
            Label          = "baseline",
        };
    }

    // ── VRAM preset table ────────────────────────────────────────────────────
    // Keyed by minimum free VRAM (MB).  When free VRAM is between two tiers
    // the lower tier is used — this is intentionally conservative so the
    // baseline always starts successfully.
    //
    // ReserveMb: VRAM held back from model weights for KV cache + driver overhead.
    // Higher tiers get more reserve because they run larger contexts.

    private readonly record struct GpuTier(
        int MinVramMb, int ReserveMb, int CtxSize, bool FlashAttn, int Batch, int UBatch);

    private static readonly GpuTier[] GpuTiers =
    [
        new(     0,   512,  2048, false,  256,  256),  //  < 4 GB  (iGPU / very old dGPU)
        new(  4096,   768,  2048,  true,  256,  256),  //    4 GB  (GTX 1050 Ti, GTX 1650)
        new(  6144,  1024,  4096,  true,  256,  256),  //    6 GB  (RTX 2060, GTX 1060)
        new(  8192,  1536,  4096,  true,  512,  512),  //    8 GB  (RTX 3070, RX 6700 XT)
        new( 10240,  1536,  8192,  true,  512,  512),  //   10 GB  (RTX 3080 10GB)
        new( 12288,  2048,  8192,  true,  512,  512),  //   12 GB  (RTX 3060, RTX 4070)
        new( 16384,  2560, 16384,  true,  512,  512),  //   16 GB  (RX 6800, RTX 4080)
        new( 20480,  3072, 16384,  true,  512,  512),  //   20 GB  (RTX 3080 20GB)
        new( 24576,  3584, 32768,  true,  512,  512),  //   24 GB  (RTX 3090 / 4090, A5000)
        new( 32768,  4096, 32768,  true, 1024,  512),  //   32 GB  (RTX 6000 Ada, A6000 48-32)
        new( 49152,  6144, 65536,  true, 1024, 1024),  //   48 GB  (RTX A6000, L40)
        new( 65536,  8192, 131072, true, 2048, 1024),  //   64 GB+ (A100, H100)
    ];

    private static (int ReserveMb, int CtxSize, bool FlashAttn, int Batch, int UBatch)
        GpuPreset(long freeVramMb)
    {
        // Walk backwards to find the highest tier that doesn't exceed free VRAM.
        for (int i = GpuTiers.Length - 1; i >= 0; i--)
        {
            if (freeVramMb >= GpuTiers[i].MinVramMb)
            {
                var t = GpuTiers[i];
                return (t.ReserveMb, t.CtxSize, t.FlashAttn, t.Batch, t.UBatch);
            }
        }
        var fallback = GpuTiers[0];
        return (fallback.ReserveMb, fallback.CtxSize, fallback.FlashAttn, fallback.Batch, fallback.UBatch);
    }

    // CPU-only preset: context is the scarce resource (RAM-limited), no flash attention.
    private readonly record struct CpuTier(int MinRamMb, int CtxSize, int Batch, int UBatch);

    private static readonly CpuTier[] CpuTiers =
    [
        new(     0,  2048,  64,  64),   //  < 8 GB RAM
        new(  8192,  4096, 128, 128),   //    8 GB RAM
        new( 16384,  8192, 256, 256),   //   16 GB RAM
        new( 32768, 16384, 512, 256),   //   32 GB RAM
        new( 65536, 32768, 512, 512),   //   64 GB RAM
    ];

    private static (int CtxSize, int Batch, int UBatch) CpuPreset(long freeRamMb)
    {
        for (int i = CpuTiers.Length - 1; i >= 0; i--)
        {
            if (freeRamMb >= CpuTiers[i].MinRamMb)
            {
                var t = CpuTiers[i];
                return (t.CtxSize, t.Batch, t.UBatch);
            }
        }
        var fallback = CpuTiers[0];
        return (fallback.CtxSize, fallback.Batch, fallback.UBatch);
    }
}
