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
    ProfileLibraryService profileLibrary,
    ILogger<OptimizerService> logger)
{
    /// <summary>Name of the auto-maintained "best known config" profile for a model+profile pair.</summary>
    public static string AutoBestProfileName(string modelName, ProfileType profile)
        => $"Auto-best: {modelName} ({profile})";
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

        // ── Live "auto-best" profile ─────────────────────────────────────────
        // Whenever this run's best config changes, an auto-maintained named profile for the
        // model is written to disk immediately — so a crash or freeze mid-run never loses
        // the best result found so far, and the profile is selectable straight away as a
        // baseline for the next run or for launching the model.
        Guid? autoBestId = null;
        string? autoBestSavedFp = null;
        double? existingScoreGuard = null;   // a previous run's score we must beat before taking over
        string? existingFp = null;
        bool autoBestLookedUp = false;

        async Task SaveAutoBestAsync(bool forceRefresh = false)
        {
            var bestIter = session.Best;
            if (bestIter is null || bestIter.Result.GenerationRate <= 0) return;

            string fp = bestIter.Settings.Fingerprint();
            if (!forceRefresh && fp == autoBestSavedFp) return;

            try
            {
                string name = AutoBestProfileName(session.ModelName, session.Profile);
                if (!autoBestLookedUp)
                {
                    autoBestLookedUp = true;
                    var existing = await profileLibrary.FindByNameAsync(name);
                    if (existing is not null)
                    {
                        autoBestId = existing.Id;
                        existingFp = existing.Settings.Fingerprint();
                        existingScoreGuard = existing.BenchmarkScore;
                    }
                }

                // Don't let a fresh run's weaker interim best clobber a better result from a
                // previous run — hold off until this run matches the stored config or beats
                // its score. (Once we've taken over, we follow this run's best freely.)
                if (existingScoreGuard.HasValue
                    && fp != existingFp
                    && bestIter.Result.CompositeScore < existingScoreGuard.Value)
                    return;
                existingScoreGuard = null;

                var autoProfile = new SavedProfile
                {
                    Id = autoBestId ?? Guid.NewGuid(),
                    Name = name,
                    ModelPath = session.ModelPath,
                    Settings = bestIter.Settings.Clone(),
                    OptimizationProfile = session.Profile,
                    BenchmarkScore = bestIter.Result.CompositeScore,
                    BenchmarkPpRate = bestIter.Result.PromptProcessingRate,
                    BenchmarkTgRate = bestIter.Result.GenerationRate,
                    BenchmarkTtftMs = bestIter.Result.TimeToFirstTokenMs,
                    Notes = $"Auto-saved best configuration, updated live during optimization runs " +
                            $"(last: {DateTime.Now:yyyy-MM-dd HH:mm}, iteration {bestIter.Number}). " +
                            "Use it to launch the model or as the baseline for a new run.",
                };
                autoBestId = autoProfile.Id;
                await profileLibrary.SaveAsync(autoProfile);
                autoBestSavedFp = fp;
                logger.LogDebug("Auto-best profile updated (iteration {N}, score {Score:F3})",
                    bestIter.Number, bestIter.Result.CompositeScore);
            }
            catch (Exception ex)
            {
                // A profile-save hiccup must never abort the optimization run itself.
                logger.LogWarning("Failed to update auto-best profile: {Msg}", ex.Message);
            }
        }

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
            await SaveAutoBestAsync();

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
                await SaveAutoBestAsync();

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
                    // MergeVerification changed the champion's metrics without changing its
                    // fingerprint — refresh the auto-best profile's stored numbers.
                    await SaveAutoBestAsync(forceRefresh: true);
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
            await SaveAutoBestAsync();
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
                RepeatPenalty  = DefaultRepeatPenalty,
                RepeatLastN    = DefaultRepeatLastN,
                DryMultiplier  = DefaultDryMultiplier,
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
            RepeatPenalty  = DefaultRepeatPenalty,
            RepeatLastN    = DefaultRepeatLastN,
            DryMultiplier  = DefaultDryMultiplier,
            Label          = "baseline",
        };
    }

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
