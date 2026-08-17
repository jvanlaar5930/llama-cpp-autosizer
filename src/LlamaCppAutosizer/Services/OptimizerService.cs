using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public record OptimizationOptions(
    int MaxIterations = 20,
    double ConvergenceThreshold = 0.01,
    int ConvergencePatience = 3,
    int Port = 8080,
    bool IncludeRepetitionStressTest = false,
    bool VerifyBestAtEnd = true
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
        Action<string>? onPhase = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        options ??= new OptimizationOptions();
        onPhase?.Invoke("Detecting hardware…");
        var hw = await hardware.DetectAsync();
        session.Hardware = hw;

        try
        {
            // ── Iteration 0: baseline ─────────────────────────────────────────
            logger.LogInformation("Starting baseline benchmark");
            onPhase?.Invoke("Starting llama-server (baseline)…");
            await StartServerAsync(serverExecutable, modelPath, initialSettings, options.Port, ct);
            // Server may have auto-reverted an unsupported setting (e.g. mmap, quantized KV
            // cache) to get started — use what's actually running, not what was requested.
            var effectiveInitialSettings = server.LastEffectiveSettings ?? initialSettings;
            string? initialAdjustment = server.LastStartAdjustmentNote;

            onPhase?.Invoke("Running baseline benchmark…");
            var baselineResult = await benchmarks.RunAsync(effectiveInitialSettings, modelPath, profile, ct,
                options.IncludeRepetitionStressTest, onPhase);

            // Rescale the scoring bounds around what this machine actually produced, before
            // anything is scored — otherwise fast hardware saturates the speed terms at 1.0 and
            // the optimizer goes blind to exactly the throughput gains it is meant to find.
            // Calibrating here (rather than after) means the baseline and every later iteration
            // are scored on one identical scale, so no rescoring of history is needed.
            // CalibratedTo returns a copy: the caller's profile instance is long-lived and
            // reused across runs, so it must not pick up this run's bounds.
            profile = profile.CalibratedTo(baselineResult);
            logger.LogInformation("Scoring bounds calibrated to baseline: {Bounds}",
                profile.DescribeScoringBounds());

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
            // consecutiveNonImprovements: resets on any iteration that makes meaningful
            // progress (capability gain, or a composite gain of at least ConvergenceThreshold —
            // smaller gains are indistinguishable from benchmark noise). When it reaches
            // ConvergencePatience we ask the LLM for a completely fresh angle ("final push").
            // If the LLM says DONE → stop early. Otherwise reset the counter and continue.
            // We never stop just because the normal recommendation pool is exhausted —
            // instead we trigger the final-push path there too.
            int consecutiveNonImprovements = 0;
            int consecutiveStartFailures = 0;
            int maxConsecutiveNonImprovements = Math.Max(1, options.ConvergencePatience);
            const int MaxConsecutiveStartFailures = 5;

            for (int iter = 1; iter <= options.MaxIterations && !ct.IsCancellationRequested; iter++)
            {
                // Refresh hardware readings while the previous iteration's server is still
                // running — recommendations then see real remaining VRAM/RAM headroom with
                // the model loaded, not the stale pre-load numbers from session start.
                onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — refreshing hardware info…");
                hw = await hardware.DetectAsync();

                // ── Pick the next change to try ───────────────────────────────
                ParameterChange? change;

                if (consecutiveNonImprovements >= maxConsecutiveNonImprovements)
                {
                    // Patience exhausted: present best to LLM and ask for a new angle
                    consecutiveNonImprovements = 0;
                    logger.LogInformation("{N} consecutive non-improvements — asking LLM for final push",
                        maxConsecutiveNonImprovements);

                    onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — asking LLM for a fresh angle…");
                    change = await recommender.GetFinalPushAsync(session, profile, hw, ct);
                    if (change is null)
                    {
                        // LLM said DONE — but a single DONE against a no-improvement history
                        // shouldn't end the run while the heuristic pool still has untried
                        // moves (e.g. the quality-phase steps). Only stop when both agree.
                        change = recommender.GetHeuristicRecommendation(session, profile, hw);
                        if (change is null)
                        {
                            session.CompletionReason = "LLM confirmed no further improvements possible";
                            break;
                        }
                        logger.LogInformation(
                            "LLM said DONE but heuristic still has an untried move — continuing with {Change}",
                            change.Describe());
                    }
                }
                else
                {
                    onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — asking for next recommendation…");
                    change = await recommender.GetNextRecommendationAsync(session, profile, hw, ct);
                    if (change is null)
                    {
                        // Normal pool exhausted — try the final push before giving up
                        logger.LogInformation("Recommendation pool empty — trying final-push LLM prompt");
                        onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — asking LLM for a fresh angle…");
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

                logger.LogInformation("Iteration {N}: {Change} [{Source}]",
                    iter, change.Describe(), change.Source);

                // ── Start server with new settings ────────────────────────────
                onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — applying {change.Describe()}…");
                await StopServerAsync();
                string? startFailReason = null;
                string? startAdjustment = null;
                bool fatalStartFailure = false;
                try
                {
                    onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — starting llama-server…");
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
                        "Server failed to start with {Change} ({Reason})",
                        change.Describe(), ex.Message);

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
                        StatusMessage = $"[{sourceLabel}] {change.Describe()}  [SKIP — server failed to start]",
                    };
                    session.AddIteration(skipped);
                    yield return skipped;
                    consecutiveNonImprovements++;
                    if (fatalStartFailure) break;
                    continue;
                }

                // ── Duplicate guard ───────────────────────────────────────────
                // The recommender refuses changes that recreate a tested config, but the
                // server's startup auto-adjustments (e.g. reverting an unsupported KV quant)
                // can still land us back on one. Re-benchmarking it would just duplicate a
                // known data point — record a visible skip and move on.
                if (session.HasTestedConfiguration(nextSettings))
                {
                    string dupNote = startAdjustment is null ? "" : $" after auto-adjustment: {startAdjustment}";
                    var duplicate = new OptimizationIteration
                    {
                        Number = iter,
                        Settings = nextSettings,
                        Result = new BenchmarkResult(),
                        AppliedChange = change,
                        StatusMessage = $"[{SourceTag(change.Source)}] {change.Describe()}  [SKIP — configuration already benchmarked{dupNote}]",
                    };
                    session.AddIteration(duplicate);
                    await persistence.SaveAsync(session);
                    yield return duplicate;
                    // Deliberately NOT counted toward the non-improvement streak: no new
                    // measurement happened, so it says nothing about convergence — counting
                    // it burned patience and ended runs early. MaxIterations still bounds it.
                    continue;
                }

                // ── Benchmark ─────────────────────────────────────────────────
                onPhase?.Invoke($"Iteration {iter}/{options.MaxIterations} — running benchmark…");
                BenchmarkResult? iterResult = null;
                string? benchFailReason = null;
                // Once the speed target is met, Agentic runs get the repetition stress test
                // automatically — it's the direct detector of the quant-induced loops the
                // quality phase is tuning against, and short prompts rarely trigger them.
                bool autoStressTest = profile.Type == ProfileType.Agentic
                    && (session.BestResult?.GenerationRate ?? 0) >= session.TargetTgSpeed;
                try
                {
                    iterResult = await benchmarks.RunAsync(nextSettings, modelPath, profile, ct,
                        options.IncludeRepetitionStressTest || autoStressTest, onPhase);
                    iterResult.CompositeScore = profile.ScoreResult(iterResult);
                }
                catch (OperationCanceledException)
                {
                    break;  // user cancelled — the finally block records the reason
                }
                catch (Exception ex)
                {
                    benchFailReason = ex.Message;
                    logger.LogInformation("Benchmark failed ({Reason}); recording as skipped iteration", ex.Message);
                }

                // Record benchmark failures as visible skipped iterations. This feeds the
                // config into the duplicate guard and the LLM's history (so it isn't
                // suggested again), instead of silently vanishing from the record.
                if (iterResult is null)
                {
                    var benchFailed = new OptimizationIteration
                    {
                        Number = iter,
                        Settings = nextSettings,
                        Result = new BenchmarkResult { Notes = $"Benchmark failed: {benchFailReason}" },
                        AppliedChange = change,
                        StatusMessage = $"[{SourceTag(change.Source)}] {change.Describe()}  [SKIP — benchmark failed: {benchFailReason}]",
                    };
                    session.AddIteration(benchFailed);
                    await persistence.SaveAsync(session);
                    yield return benchFailed;
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
                    StatusMessage = $"[{tag}] {change.Describe()}  \"{change.Reasoning}\"{adjustmentSuffix}",
                };
                var prevAnchor = session.Best;
                session.AddIteration(iteration);
                await persistence.SaveAsync(session);

                yield return iteration;

                // ── Track non-improvement streak ──────────────────────────────
                // Becoming the anchor only counts as progress when the gain is meaningful:
                // a capability gain, a quality gain, or a composite gain of at least
                // ConvergenceThreshold. Sub-threshold composite gains are indistinguishable
                // from benchmark noise, so they keep the convergence counter running even
                // though the anchor advanced.
                bool meaningfulImprovement = iteration.IsBestSoFar &&
                    (prevAnchor is null
                     || HasCapabilityGain(iteration, prevAnchor)
                     || HasQualityGain(iteration, prevAnchor)
                     || iteration.Result.CompositeScore - prevAnchor.Result.CompositeScore
                            >= options.ConvergenceThreshold);
                if (meaningfulImprovement)
                    consecutiveNonImprovements = 0;
                else
                    consecutiveNonImprovements++;
            }

            // ── Champion verification ─────────────────────────────────────────
            // Re-benchmark the winning configuration once and fold the second sample into
            // its reported metrics. A single measurement can win on noise or an early
            // (cooler) thermal state; the second run happens under the same late-run
            // conditions as its rivals, making the reported result trustworthy.
            var champion = session.Best;
            bool shouldVerify = options.VerifyBestAtEnd
                && !ct.IsCancellationRequested
                && server.IsRunning   // false after a fatal start failure — no point retrying
                && champion is not null
                && champion.Result.GenerationRate > 0
                && session.Iterations.Count(i => i.Result.GenerationRate > 0) > 1;

            if (shouldVerify)
            {
                OptimizationIteration? verification = null;
                var champ = champion!;
                try
                {
                    var verifySettings = champ.Settings.Clone();
                    verifySettings.Label = "verify";
                    onPhase?.Invoke("Verifying best configuration — restarting llama-server…");
                    await StopServerAsync();
                    await StartServerAsync(serverExecutable, modelPath, verifySettings, options.Port, ct);

                    onPhase?.Invoke("Verifying best configuration — re-running benchmark…");
                    // Agentic verification always includes the stress test: the champion is
                    // about to be recommended for real agent workloads, so a latent loop
                    // tendency must surface here rather than in production use.
                    var verifyResult = await benchmarks.RunAsync(verifySettings, modelPath, profile, ct,
                        options.IncludeRepetitionStressTest || profile.Type == ProfileType.Agentic, onPhase);
                    verifyResult.CompositeScore = profile.ScoreResult(verifyResult);

                    MergeVerification(champ, verifyResult, profile);

                    verification = new OptimizationIteration
                    {
                        Number = session.Iterations.Max(i => i.Number) + 1,
                        Settings = verifySettings,
                        Result = verifyResult,
                        IsVerification = true,
                        StatusMessage = $"Verification — re-benchmarked best config (iter {champ.Number}); " +
                                        "its reported metrics now combine both runs",
                    };
                    session.AddIteration(verification);
                    await persistence.SaveAsync(session);
                }
                catch (Exception ex)
                {
                    logger.LogInformation("Champion verification failed ({Reason}); keeping single-run metrics", ex.Message);
                }
                if (verification is not null)
                    yield return verification;
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
        "llm"         => "LLM",
        "llm-push"    => "LLM★",
        "claude"      => "Claude",
        "claude-push" => "Claude★",
        _             => "Heuristic",
    };

    // A capability gain is progress even when the speed-weighted composite drops: larger
    // context, more MoE experts (null = model default = all of them), or recovering
    // healthy output from a degenerate config.
    private static bool HasCapabilityGain(OptimizationIteration current, OptimizationIteration previous)
    {
        if (current.Settings.ContextSize > previous.Settings.ContextSize) return true;
        if ((current.Settings.MoeExpertUsed ?? int.MaxValue) > (previous.Settings.MoeExpertUsed ?? int.MaxValue)) return true;
        return current.Result.QualityScore >= OptimizationSession.MinHealthyQuality
            && previous.Result.QualityScore < OptimizationSession.MinHealthyQuality;
    }

    // Quality-phase moves (KV-quant reversal, sampler escalation, thinking mode) often cost
    // speed, so their composite delta is flat or negative — but a real gain in any quality
    // metric is exactly the progress that phase exists to make, and must reset the patience
    // counter or the run ends before quality tuning gets anywhere.
    private const double QualityGainEpsilon = 0.05;
    private static bool HasQualityGain(OptimizationIteration current, OptimizationIteration previous)
        => current.Result.QualityScore    - previous.Result.QualityScore    >= QualityGainEpsilon
        || current.Result.ToolSuccessRate - previous.Result.ToolSuccessRate >= QualityGainEpsilon
        || current.Result.AgentLoopScore  - previous.Result.AgentLoopScore  >= QualityGainEpsilon;

    // Folds a verification re-run into the champion's reported metrics: rates and latencies
    // are averaged (two samples beat one), quality-style scores take the worse of the two
    // (a repetition loop or a wrong answer in either run means the config can produce them).
    private static void MergeVerification(
        OptimizationIteration champion, BenchmarkResult verify, OptimizationProfile profile)
    {
        var r = champion.Result;
        r.PromptProcessingRate = (r.PromptProcessingRate + verify.PromptProcessingRate) / 2;
        r.GenerationRate = (r.GenerationRate + verify.GenerationRate) / 2;
        r.TimeToFirstTokenMs = (r.TimeToFirstTokenMs + verify.TimeToFirstTokenMs) / 2;
        r.AverageTotalMs = (r.AverageTotalMs + verify.AverageTotalMs) / 2;
        r.QualityScore = Math.Min(r.QualityScore, verify.QualityScore);
        r.AccuracyScore = Math.Min(r.AccuracyScore, verify.AccuracyScore);
        r.ToolSuccessRate = Math.Min(r.ToolSuccessRate, verify.ToolSuccessRate);
        r.CompositeScore = profile.ScoreResult(r);
        r.Notes = (r.Notes is null ? "" : r.Notes + " ") +
            "Metrics combined with end-of-run verification re-benchmark.";
    }

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
    /// Explains how a baseline was arrived at, so the UI and session log can show the reasoning
    /// rather than an unexplained set of numbers.
    /// </summary>
    public record BaselinePlan(LlamaSettings Settings, string Explanation);

    /// <summary>
    /// Build the starting configuration for a run, computed from the selected model's own GGUF
    /// header rather than from file-size guesswork.
    ///
    /// The goal is the largest context that still leaves the model running mostly on the GPU.
    /// Concretely, for the chosen model we read its real layer count, per-layer KV-head counts,
    /// attention head dimensions and sliding-window pattern, then compute the exact KV cache cost
    /// at each candidate context and spend whatever VRAM is left on weights. Nothing here is
    /// architecture-specific: a dense model, a GQA model and a sliding-window MoE all fall out of
    /// the same arithmetic, they just land on different answers.
    ///
    /// If the header can't be read the old file-size tier presets are used unchanged, so an
    /// unreadable or non-GGUF model still produces the baseline it always did.
    /// </summary>
    /// <param name="preferredKvType">
    /// When the user has already chosen a KV cache type by hand, pass it here so the layer budget
    /// is computed against that type's actual cost. Overriding the cache type after planning
    /// would leave the ngl/n-cpu-moe numbers sized for a different KV footprint.
    /// </param>
    public static BaselinePlan BuildInitialPlan(
        string modelPath, OptimizationProfile profile, HardwareInfo hw,
        string? preferredKvType = null)
    {
        long modelSizeMb = new FileInfo(modelPath).Length / (1024 * 1024);
        var meta = GgufMetadataService.Read(modelPath);
        int threads = hw.CpuCores > 0 ? hw.CpuCores : -1;
        bool isThinking = LlamaSettings.IsThinkingModel(modelPath);

        LlamaSettings Base() => new()
        {
            Threads         = threads,
            ThreadsBatch    = threads,
            Mmap            = true,
            ThinkingEnabled = isThinking ? false : null,
            RepeatPenalty   = DefaultRepeatPenalty,
            RepeatLastN     = DefaultRepeatLastN,
            DryMultiplier   = DefaultDryMultiplier,
            Label           = "baseline",
        };

        if (!hw.HasGpu)
        {
            var cpuPreset = CpuPreset(hw.RamFreeMb);
            var cpu = Base();
            cpu.GpuLayers      = 0;
            cpu.ContextSize    = Math.Min(cpuPreset.CtxSize, profile.TargetContextSize);
            cpu.BatchSize      = cpuPreset.Batch;
            cpu.UBatchSize     = cpuPreset.UBatch;
            cpu.FlashAttention = false;
            return new BaselinePlan(cpu,
                $"CPU-only: {cpu.ContextSize:N0} ctx from the {hw.RamFreeMb:N0} MB free-RAM tier.");
        }

        var preset = GpuPreset(hw.FreeVramMb);

        // ── No header: keep the historical tier-preset behaviour ─────────────
        if (meta?.BlockCount is not > 0)
            return LegacyGpuBaseline(Base(), preset, modelSizeMb, profile, hw);

        int layers = meta.BlockCount.Value;
        double mbPerLayer = (double)modelSizeMb / layers;

        // Usable VRAM: free, minus a slice for the CUDA/HIP context, kernels, compute buffers and
        // allocator fragmentation. Overshooting here is what makes a baseline fail to start, so
        // the margin is deliberately generous — the optimizer can reclaim it later.
        long usableVram = (long)(hw.FreeVramMb * (1 - VramSafetyFraction)) - GpuOverheadMb;

        // Expert FFN weights dominate an MoE model, and they are the part that can be pushed to
        // the CPU without evicting attention. Split the per-layer cost so the two can be budgeted
        // separately; for a dense model expertShare is 0 and this collapses to "all weights".
        double expertShare = meta.ExpertShareOfLayer();
        bool useCpuMoeOffload = expertShare >= MinExpertShareForOffload;
        double expertMbPerLayer = mbPerLayer * expertShare;
        double nonExpertMbPerLayer = mbPerLayer - expertMbPerLayer;
        long nonExpertTotalMb = (long)(nonExpertMbPerLayer * layers);

        // Context ladder, highest first. The model's own trained context is a hard ceiling —
        // asking for more than it was trained for wastes VRAM on KV that RoPE can't use well.
        int ceiling = Math.Min(profile.TargetContextSize, meta.ContextLength ?? int.MaxValue);
        var ladder = ContextLadder(ceiling);

        // Evaluates one (context, KV type) pair. Returns null when it doesn't fit well enough
        // to be worth starting from.
        Candidate? Evaluate(int ctx, string? kvType)
        {
            long kvMb = meta.KvCacheBytes(ctx, kvType, kvType, preset.UBatch) / (1024 * 1024);
            long weightBudget = usableVram - kvMb;
            if (weightBudget <= 0) return null;

            if (useCpuMoeOffload)
            {
                // Attention and dense weights must all fit, otherwise CPU-MoE offload isn't the
                // right tool and a smaller context should be tried instead.
                if (weightBudget < nonExpertTotalMb) return null;

                int expertLayersOnGpu = expertMbPerLayer > 0
                    ? (int)((weightBudget - nonExpertTotalMb) / expertMbPerLayer)
                    : layers;
                expertLayersOnGpu = Math.Clamp(expertLayersOnGpu, 0, layers);

                int? nCpuMoe = layers - expertLayersOnGpu;
                if (nCpuMoe == 0) nCpuMoe = null;

                return new Candidate(ctx, kvType, kvMb, -1, nCpuMoe,
                    (double)expertLayersOnGpu / layers,
                    nCpuMoe is null
                        ? "whole model fits in VRAM"
                        : $"{expertLayersOnGpu}/{layers} layers keep their experts in VRAM, " +
                          $"--n-cpu-moe {nCpuMoe} sends the rest to CPU (attention stays on GPU)");
            }

            int fits = mbPerLayer > 0 ? (int)(weightBudget / mbPerLayer) : layers;
            if (fits < Math.Max(1, layers / 4)) return null;   // too little on GPU to be worth it

            int ngl = fits >= layers ? -1 : fits;
            return new Candidate(ctx, kvType, kvMb, ngl, null,
                Math.Min(1.0, (double)fits / layers),
                ngl == -1 ? "whole model fits in VRAM" : $"{ngl}/{layers} layers offloaded");
        }

        foreach (int ctx in ladder)
        {
            // Context first, then GPU residency, then KV precision. Once the target context is
            // met, VRAM spent on f16 KV is VRAM not spent on weights — and resident weights buy
            // far more speed than KV precision buys quality. So q8_0 is taken only when it
            // actually moves a meaningful share of the model onto the GPU; when the difference is
            // marginal (a layer or two) f16 is kept for its better quality.
            Candidate? pick;
            if (preferredKvType is not null)
            {
                // The user picked a cache type explicitly — fit around it rather than second-guess it.
                pick = Evaluate(ctx, preferredKvType);
            }
            else
            {
                var f16 = Evaluate(ctx, null);
                var q8 = Evaluate(ctx, "q8_0");
                pick = (f16, q8) switch
                {
                    (null, null) => null,
                    (null, var q) => q,
                    (var f, null) => f,
                    var (f, q) => q!.ResidentFraction - f!.ResidentFraction >= MinResidencyGainForKvQuant ? q : f,
                };
            }
            if (pick is null) continue;

            var s = Base();
            s.ContextSize    = pick.Context;
            s.GpuLayers      = pick.GpuLayers;
            s.NCpuMoe        = pick.NCpuMoe;
            s.BatchSize      = preset.Batch;
            s.UBatchSize     = preset.UBatch;
            // Flash attention cuts attention memory and is a prerequisite for quantized KV.
            s.FlashAttention = true;
            s.CacheTypeK     = pick.KvType;
            s.CacheTypeV     = pick.KvType;

            return new BaselinePlan(s,
                $"{pick.Context:N0} ctx (model max {meta.ContextLength:N0}) · KV {pick.KvType ?? "f16"} " +
                $"≈ {pick.KvMb:N0} MB · {pick.How} · {hw.FreeVramMb:N0} MB VRAM free");
        }

        // Nothing on the ladder fit — fall back to the smallest context and let the server's
        // self-healing/start-failure path report what actually went wrong.
        var minimal = Base();
        minimal.ContextSize    = ladder[^1];
        minimal.GpuLayers      = 0;
        minimal.BatchSize      = preset.Batch;
        minimal.UBatchSize     = preset.UBatch;
        minimal.FlashAttention = true;
        return new BaselinePlan(minimal,
            $"Model does not fit in {hw.FreeVramMb:N0} MB VRAM at any context — starting on CPU " +
            $"at {minimal.ContextSize:N0} ctx.");
    }

    /// <summary>Backwards-compatible entry point — returns just the settings.</summary>
    public static LlamaSettings BuildInitialSettings(
        string modelPath, OptimizationProfile profile, HardwareInfo hw)
        => BuildInitialPlan(modelPath, profile, hw).Settings;

    // Descending context candidates, starting at the ceiling. Powers of two down to 2048 so a
    // step-down halves KV cost rather than shaving it.
    private static int[] ContextLadder(int ceiling)
    {
        int[] all = [131072, 65536, 32768, 16384, 8192, 4096, 2048];
        var usable = all.Where(c => c <= ceiling).ToArray();
        return usable.Length > 0 ? usable : [Math.Max(2048, ceiling)];
    }

    // Pre-GGUF-header behaviour, kept verbatim for models whose header can't be parsed.
    private static BaselinePlan LegacyGpuBaseline(
        LlamaSettings s,
        (int ReserveMb, int CtxSize, bool FlashAttn, int Batch, int UBatch) preset,
        long modelSizeMb, OptimizationProfile profile, HardwareInfo hw)
    {
        int estimatedLayers = modelSizeMb switch
        {
            < 2000  => 24, < 5000 => 32, < 10000 => 40, < 20000 => 60, _ => 80,
        };
        double mbPerLayer = (double)modelSizeMb / estimatedLayers;

        long vramForWeights = Math.Max(0, hw.FreeVramMb - preset.ReserveMb);
        int maxLayersInVram = mbPerLayer > 0 ? (int)(vramForWeights / mbPerLayer) : estimatedLayers;
        int gpuLayers = maxLayersInVram >= estimatedLayers ? -1 : maxLayersInVram;

        int contextSize = Math.Min(preset.CtxSize, profile.TargetContextSize);
        if (gpuLayers != -1)
        {
            long headroom = hw.FreeVramMb - (long)(gpuLayers * mbPerLayer);
            if      (headroom < 1536) contextSize = Math.Min(contextSize, 2048);
            else if (headroom < 3072) contextSize = Math.Min(contextSize, 4096);
            else if (headroom < 6144) contextSize = Math.Min(contextSize, 8192);
        }

        s.GpuLayers      = gpuLayers;
        s.ContextSize    = contextSize;
        s.BatchSize      = preset.Batch;
        s.UBatchSize     = preset.UBatch;
        s.FlashAttention = preset.FlashAttn;
        return new BaselinePlan(s,
            $"GGUF header unreadable — using file-size tier estimate ({contextSize:N0} ctx, ngl={gpuLayers}).");
    }

    // VRAM held back from the plan. GpuOverheadMb covers the driver/runtime context and compute
    // buffers; the fraction absorbs allocator fragmentation and the fact that "free VRAM" drifts
    // between the reading and the actual load.
    private const long GpuOverheadMb = 900;
    private const double VramSafetyFraction = 0.05;
    // Below this expert share, per-layer offload isn't worth it and plain layer offload is used.
    private const double MinExpertShareForOffload = 0.5;
    // How much extra of the model must land in VRAM before quantized KV is preferred over f16.
    private const double MinResidencyGainForKvQuant = 0.10;

    // One evaluated (context, KV type) option. ResidentFraction is the share of the model's
    // layers that would live in VRAM, which is the proxy the planner maximizes for speed.
    private sealed record Candidate(
        int Context, string? KvType, long KvMb, int GpuLayers, int? NCpuMoe,
        double ResidentFraction, string How);

    // Conservative anti-repetition defaults applied to every baseline config. These guard
    // against degenerate decoding loops ("is is is is...") from the very first iteration
    // rather than only being discovered reactively after a benchmark's quality score craters.
    // The optimizer can still loosen or tighten them further — see RecommendationService.
    private const float DefaultRepeatPenalty = 1.1f;
    private const int DefaultRepeatLastN = 256;
    private const float DefaultDryMultiplier = 0.8f;

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
