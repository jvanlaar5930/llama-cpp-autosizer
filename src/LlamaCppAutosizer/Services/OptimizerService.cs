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
    public async IAsyncEnumerable<OptimizationIteration> OptimizeAsync(
        string serverExecutable,
        string modelPath,
        LlamaSettings initialSettings,
        OptimizationProfile profile,
        OptimizationOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        options ??= new OptimizationOptions();
        var hw = await hardware.DetectAsync();

        var session = new OptimizationSession
        {
            ModelPath = modelPath,
            Profile = profile.Type,
            Hardware = hw,
        };

        // ── Iteration 0: baseline ───────────────────────────────────────────
        logger.LogInformation("Starting baseline benchmark");
        await StartServerAsync(serverExecutable, modelPath, initialSettings, options.Port, ct);

        var baselineResult = await benchmarks.RunAsync(initialSettings, modelPath, profile, ct);
        baselineResult.CompositeScore = profile.ScoreResult(baselineResult);
        baselineResult.Notes = "Baseline";

        var baseline = new OptimizationIteration
        {
            Number = 0,
            Settings = initialSettings.Clone(),
            Result = baselineResult,
            AppliedChange = null,
            IsBestSoFar = true,
        };
        session.AddIteration(baseline);
        await persistence.SaveAsync(session);

        yield return baseline;

        // ── Iterative tuning ────────────────────────────────────────────────
        int noImprovementCount = 0;

        for (int iter = 1; iter <= options.MaxIterations && !ct.IsCancellationRequested; iter++)
        {
            var change = await recommender.GetNextRecommendationAsync(session, profile, hw, ct);
            if (change is null)
            {
                logger.LogInformation("No more recommendations — stopping");
                session.CompletionReason = "All candidate parameters explored";
                break;
            }

            var nextSettings = RecommendationService.Apply(session.BestSettings!, change);
            nextSettings.Label = $"iter{iter}";

            logger.LogInformation(
                "Iteration {N}: trying {Param} = {Val} ({Source})",
                iter, change.Parameter, change.NewValue, change.Source);

            // Restart server with new settings
            await StopServerAsync();
            try
            {
                await StartServerAsync(serverExecutable, modelPath, nextSettings, options.Port, ct);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Server failed to start with new settings: {Msg}", ex.Message);
                // Revert to best settings and continue
                await StartServerAsync(serverExecutable, modelPath, session.BestSettings!, options.Port, ct);
                continue;
            }

            BenchmarkResult iterResult;
            try
            {
                iterResult = await benchmarks.RunAsync(nextSettings, modelPath, profile, ct);
                iterResult.CompositeScore = profile.ScoreResult(iterResult);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Benchmark failed: {Msg}", ex.Message);
                continue;
            }

            double prevBest = session.BestResult!.CompositeScore;
            var iteration = new OptimizationIteration
            {
                Number = iter,
                Settings = nextSettings,
                Result = iterResult,
                AppliedChange = change,
            };
            session.AddIteration(iteration);
            await persistence.SaveAsync(session);

            double improvement = iterResult.CompositeScore - prevBest;
            if (improvement < options.ConvergenceThreshold)
            {
                noImprovementCount++;
                logger.LogDebug("No significant improvement ({Delta:F3}), patience {N}/{Max}",
                    improvement, noImprovementCount, options.ConvergencePatience);
            }
            else
            {
                noImprovementCount = 0;
            }

            yield return iteration;

            if (noImprovementCount >= options.ConvergencePatience)
            {
                session.CompletionReason = $"Converged after {noImprovementCount} non-improving iterations";
                logger.LogInformation("Converged — stopping optimization");
                break;
            }
        }

        // ── Finalize ────────────────────────────────────────────────────────
        session.IsComplete = true;
        session.CompletedAt = DateTime.UtcNow;
        session.CompletionReason ??= ct.IsCancellationRequested
            ? "Cancelled by user"
            : "Max iterations reached";

        await persistence.SaveAsync(session);
        await StopServerAsync();

        logger.LogInformation("Optimization complete. Best score: {Score:F3}", session.BestResult?.CompositeScore ?? 0);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

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
    /// </summary>
    public static LlamaSettings BuildInitialSettings(
        string modelPath, OptimizationProfile profile, HardwareInfo hw)
    {
        var modelSizeMb = new FileInfo(modelPath).Length / (1024 * 1024);

        // Rough layer estimate: most models 7B–70B have 32–80 transformer layers
        int estimatedLayers = modelSizeMb switch
        {
            < 2000 => 24,    // ~1B
            < 5000 => 32,    // ~3–7B
            < 10000 => 40,   // ~8–13B
            < 20000 => 60,   // ~14–30B
            _ => 80,         // ~34–70B
        };

        int gpuLayers = hw.HasGpu
            ? hw.EstimateMaxGpuLayers(modelSizeMb, estimatedLayers)
            : 0;

        int contextSize = hw.SuggestContextSize(modelSizeMb, gpuLayers > 0);
        // Respect the profile's target (don't exceed it initially)
        contextSize = Math.Min(contextSize, profile.TargetContextSize);

        int threads = hw.CpuCores > 0 ? hw.CpuCores : -1;

        return new LlamaSettings
        {
            GpuLayers = gpuLayers,
            ContextSize = contextSize,
            BatchSize = 512,
            UBatchSize = 512,
            Threads = threads,
            ThreadsBatch = threads,
            FlashAttention = false,
            Mmap = true,
            Label = "baseline",
        };
    }
}
