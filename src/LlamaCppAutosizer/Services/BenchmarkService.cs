using System.Diagnostics;
using System.Text.Json;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class BenchmarkService(
    LlamaServerService server,
    ILogger<BenchmarkService> logger)
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    // Sanity ceilings for reported rates, generous enough to never clip a genuine result
    // (the fastest consumer/prosumer hardware tops out far below these) but tight enough to
    // catch the failure mode actually seen: llama-server occasionally reports a near-zero
    // predicted_ms/prompt_ms for a run (e.g. a degenerate near-instant completion), and
    // tokens/ms then blows up to something like 1,000,000 t/s. That single corrupted sample
    // would otherwise dominate the session (falsely "winning" as Best, and — after the
    // rescue-mode addition — masking genuinely insufficient hardware). Samples exceeding
    // these are excluded from the rate aggregate; GeneratedText from the same run still
    // feeds quality/repetition scoring, since that measurement isn't rate-based.
    private const double MaxPlausibleTgRate = 1000.0;
    private const double MaxPlausiblePpRate = 50_000.0;

    public async Task<BenchmarkResult> RunAsync(
        LlamaSettings settings,
        string modelPath,
        OptimizationProfile profile,
        CancellationToken ct = default,
        bool includeRepetitionStressTest = false,
        Action<string>? onPhase = null)
    {
        var result = new BenchmarkResult
        {
            Settings = settings.Clone(),
            ModelPath = modelPath,
            Timestamp = DateTime.UtcNow,
        };

        // Warmup
        int warmupTotal = Math.Min(profile.WarmupRuns, profile.WarmupPrompts.Count);
        int warmupIdx = 0;
        logger.LogDebug("Running {N} warmup prompts", profile.WarmupRuns);
        foreach (var prompt in profile.WarmupPrompts.Take(profile.WarmupRuns))
        {
            warmupIdx++;
            onPhase?.Invoke(warmupTotal > 1
                ? $"Warming up ({warmupIdx}/{warmupTotal})…"
                : "Warming up…");
            try
            {
                await server.CompleteAsync(new CompletionRequest
                {
                    Prompt = prompt,
                    NPredict = 64,
                    CachePrompt = false,
                }, ct);
            }
            catch (Exception ex)
            {
                logger.LogDebug("Warmup run failed: {Msg}", ex.Message);
            }
        }

        // Main benchmark runs
        var timings = new List<RunTimings>();
        int promptTotal = profile.BenchmarkPrompts.Count;
        int promptIdx = 0;
        foreach (var prompt in profile.BenchmarkPrompts)
        {
            ct.ThrowIfCancellationRequested();
            promptIdx++;
            onPhase?.Invoke($"Benchmarking prompt {promptIdx}/{promptTotal}…");
            var runTiming = await RunSinglePromptAsync(prompt, profile.MaxGenerateTokens, result, ct);
            if (runTiming is not null) timings.Add(runTiming);
        }

        // Long-form run so the repetition-loop check below has enough tokens to actually
        // catch a degenerate config — short benchmark prompts rarely run long enough to
        // reveal it even though real long-form usage frequently does. Opt-in via options,
        // and auto-enabled by the optimizer for Agentic runs once the TG target is met.
        if (includeRepetitionStressTest && !string.IsNullOrWhiteSpace(profile.StressTestPrompt))
        {
            ct.ThrowIfCancellationRequested();
            onPhase?.Invoke("Running repetition-loop stress test…");
            var stressTiming = await RunSinglePromptAsync(profile.StressTestPrompt, profile.StressTestMaxTokens, result, ct);
            if (stressTiming is not null) timings.Add(stressTiming);
        }

        // Agentic tool-use benchmarks
        double toolSuccessRate = 1.0;
        double agentLoopScore = 1.0;
        if (profile.Type == ProfileType.Agentic)
        {
            if (profile.ToolBenchmarks.Count > 0)
            {
                var (toolScore, toolTgRate) = await RunToolBenchmarksAsync(profile, result, ct, onPhase);
                toolSuccessRate = toolScore;
                result.ToolCallEffectiveTgRate = toolTgRate;
            }

            if (profile.AgentLoopCases.Count > 0)
            {
                var (loopScore, effectiveTgRate) = await RunAgentLoopBenchmarksAsync(profile, ct, onPhase);
                agentLoopScore = loopScore;
                result.AgentLoopEffectiveTgRate = effectiveTgRate;
            }
        }

        // Accuracy probes: short prompts with objectively checkable answers. Run separately
        // from the speed prompts so their tiny generations don't pollute the rate/TTFT
        // aggregation — they exist purely to catch quality degradation that stays fluent.
        double accuracyScore = 1.0;
        if (profile.AccuracyPrompts.Count > 0)
        {
            accuracyScore = await RunAccuracyPromptsAsync(profile, ct, onPhase);
        }
        result.AccuracyScore = accuracyScore;

        onPhase?.Invoke("Scoring output quality…");

        // Aggregate. Rates are weighted by tokens/time (not averaged per-run) because a
        // run that generates very few tokens in a tiny amount of time reports an inflated
        // per_second ratio that would otherwise dominate a plain average. Samples with an
        // implausible individual rate (server-reported timing anomaly, not real throughput —
        // see MaxPlausibleTgRate/MaxPlausiblePpRate) are dropped from their respective
        // aggregate before summing, so one corrupted sample can't produce a fictional result.
        var genTimings = timings.Where(t => !IsImplausibleRate(t.PredictedTokens, t.PredictedMs, MaxPlausibleTgRate)).ToList();
        var ppTimings = timings.Where(t => !IsImplausibleRate(t.PromptTokens, t.PromptMs, MaxPlausiblePpRate)).ToList();
        int excludedCount = timings.Count - Math.Min(genTimings.Count, ppTimings.Count);
        if (excludedCount > 0)
        {
            logger.LogWarning(
                "Discarded {N} benchmark sample(s) with an implausible server-reported rate (>{TgMax:N0}t/s or >{PpMax:N0}t/s) — likely a near-zero timing glitch, not real throughput",
                excludedCount, MaxPlausibleTgRate, MaxPlausiblePpRate);
            result.Notes = (result.Notes is null ? "" : result.Notes + " ") +
                $"Discarded {excludedCount} sample(s) with an implausible reported rate (timing anomaly).";
        }

        if (genTimings.Count > 0)
        {
            double predictedMs = genTimings.Sum(t => t.PredictedMs);
            result.GenerationRate = predictedMs > 0
                ? genTimings.Sum(t => t.PredictedTokens) / predictedMs * 1000.0
                : 0;
        }
        if (ppTimings.Count > 0)
        {
            double promptMs = ppTimings.Sum(t => t.PromptMs);
            result.PromptProcessingRate = promptMs > 0
                ? ppTimings.Sum(t => t.PromptTokens) / promptMs * 1000.0
                : 0;
        }
        if (timings.Count > 0)
        {
            result.TimeToFirstTokenMs = timings.Average(t => t.TimeToFirstTokenMs);
            result.AverageTotalMs = timings.Average(t => t.TotalMs);
        }

        result.ToolCallEffectiveTgRate = ClampRate(result.ToolCallEffectiveTgRate, MaxPlausibleTgRate);
        result.AgentLoopEffectiveTgRate = ClampRate(result.AgentLoopEffectiveTgRate, MaxPlausibleTgRate);

        result.SuccessfulRuns = timings.Count;
        result.FailedRuns = profile.BenchmarkPrompts.Count - timings.Count;
        result.ToolSuccessRate = toolSuccessRate;
        result.AgentLoopScore = agentLoopScore;
        result.QualityScore = await ScoreQualityAsync(timings, result, ct);

        return result;
    }

    // True when tokens/ms implies a rate beyond what real hardware can produce — a
    // server-reported near-zero timing sample, not genuine throughput.
    private static bool IsImplausibleRate(double tokens, double ms, double maxPlausibleRate)
        => ms > 0 && tokens / ms * 1000.0 > maxPlausibleRate;

    // Same anomaly, applied to the wall-clock-derived tool/agent-loop effective rates: zero
    // out a reading that exceeds the plausible ceiling rather than let a corrupted number
    // (e.g. from a near-instant cached response) feed the composite score or rescue check.
    private static double ClampRate(double rate, double maxPlausibleRate)
        => rate > maxPlausibleRate ? 0 : rate;

    private async Task<RunTimings?> RunSinglePromptAsync(
        string prompt, int maxTokens, BenchmarkResult result, CancellationToken ct)
    {
        try
        {
            var (response, _) = await server.CompleteAsync(new CompletionRequest
            {
                Prompt = prompt,
                NPredict = maxTokens,
                Temperature = 0.1f,
                CachePrompt = false,
            }, ct);

            var t = response.Timings;
            var timing = new RunTimings
            {
                PromptTokens = t.PromptN,
                PromptMs = t.PromptMs,
                PromptPerSecond = t.PromptPerSecond,
                PredictedTokens = t.PredictedN,
                PredictedMs = t.PredictedMs,
                PredictedPerSecond = t.PredictedPerSecond,
                TotalMs = t.TotalMs,
                // Server-reported prompt-eval time is the config-sensitive component of TTFT.
                // The wall-clock tuple value covers the whole non-streaming response (≈ total
                // time), which would double-count generation speed in the composite score.
                TimeToFirstTokenMs = t.PromptMs,
                GeneratedText = response.Content,
            };
            result.IndividualRuns.Add(timing);
            return timing;
        }
        catch (Exception ex)
        {
            logger.LogWarning("Benchmark run failed for prompt '{Prompt}': {Msg}",
                prompt[..Math.Min(40, prompt.Length)], ex.Message);
            return null;
        }
    }

    private async Task<(double score, double effectiveTgRate)> RunToolBenchmarksAsync(
        OptimizationProfile profile, BenchmarkResult result, CancellationToken ct, Action<string>? onPhase = null)
    {
        double score = 0;
        int total = profile.ToolBenchmarks.Count;
        int caseIdx = 0;
        long totalTokens = 0;
        double totalWallMs = 0;

        // Serialize tools in OpenAI format
        object? toolsPayload = null;
        if (profile.ToolDefinitions.Count > 0)
        {
            toolsPayload = profile.ToolDefinitions.Select(t => new
            {
                type = "function",
                function = new
                {
                    name = t.Name,
                    description = t.Description,
                    parameters = t.Parameters,
                }
            }).ToArray();
        }

        foreach (var benchCase in profile.ToolBenchmarks)
        {
            ct.ThrowIfCancellationRequested();
            caseIdx++;
            onPhase?.Invoke($"Testing tool calls ({caseIdx}/{total})…");
            try
            {
                var sw = Stopwatch.StartNew();
                var (response, _) = await server.ChatCompleteAsync(new ChatCompletionRequest
                {
                    Messages = [new ChatMessage { Role = "user", Content = benchCase.Prompt }],
                    Tools = toolsPayload,
                    MaxTokens = 512,
                    Temperature = 0.1f,
                }, ct);
                totalWallMs += sw.Elapsed.TotalMilliseconds;
                totalTokens += response.Usage?.CompletionTokens ?? 0;

                double caseScore = ScoreToolCall(response, benchCase, profile);
                score += caseScore;
                logger.LogDebug("Tool case scored {Score:F2}: {Prompt}",
                    caseScore, benchCase.Prompt[..Math.Min(50, benchCase.Prompt.Length)]);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Tool benchmark failed: {Msg}", ex.Message);
            }
        }

        double effectiveTgRate = totalWallMs > 0 ? totalTokens / totalWallMs * 1000.0 : 0.0;
        return (total > 0 ? score / total : 1.0, effectiveTgRate);
    }

    // Graded tool-call correctness. Every prompt in ToolBenchmarks clearly requires a tool,
    // so a plain-text answer earns nothing — the old "coherent text counts as success" rule
    // made this metric ~1.0 for every config, which meant it never influenced optimization.
    //   1.0  expected tool, arguments parse, all required parameters present
    //   0.6  a defined tool with valid arguments, but not the expected one
    //   0.25 called something, but an unknown tool or malformed/incomplete arguments
    //   0.0  no tool call at all
    private static double ScoreToolCall(
        ChatCompletionResponse response, ToolBenchmarkCase benchCase, OptimizationProfile profile)
    {
        var call = response.Choices
            .SelectMany(c => c.Message?.ToolCalls ?? [])
            .FirstOrDefault(t => t.Function is not null);
        if (call?.Function is null) return 0.0;

        var definition = profile.ToolDefinitions
            .FirstOrDefault(d => d.Name.Equals(call.Function.Name, StringComparison.OrdinalIgnoreCase));
        if (definition is null || !HasValidArguments(call.Function.Arguments, definition))
            return 0.25;

        bool expectedTool = benchCase.ExpectedTools.Length == 0
            || benchCase.ExpectedTools.Contains(call.Function.Name, StringComparer.OrdinalIgnoreCase);
        return expectedTool ? 1.0 : 0.6;
    }

    // Arguments must be a JSON object containing a non-empty value for every parameter
    // the tool definition marks as required.
    private static bool HasValidArguments(string arguments, ToolDefinition definition)
    {
        try
        {
            using var args = JsonDocument.Parse(string.IsNullOrWhiteSpace(arguments) ? "{}" : arguments);
            if (args.RootElement.ValueKind != JsonValueKind.Object) return false;

            var schema = JsonSerializer.SerializeToElement(definition.Parameters);
            if (!schema.TryGetProperty("required", out var required) ||
                required.ValueKind != JsonValueKind.Array)
                return true;

            foreach (var name in required.EnumerateArray())
            {
                if (name.ValueKind != JsonValueKind.String) continue;
                if (!args.RootElement.TryGetProperty(name.GetString()!, out var value)) return false;
                if (value.ValueKind == JsonValueKind.Null) return false;
                if (value.ValueKind == JsonValueKind.String && string.IsNullOrWhiteSpace(value.GetString()))
                    return false;
            }
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    // Runs every AgentLoopCase and returns (average case score, effective tokens/sec across
    // all round-trips). Unlike RunToolBenchmarksAsync this exercises a full multi-turn loop:
    // tool call -> fabricated tool result fed back -> either the next chained call or a final
    // free-text answer. The messages list keeps growing across turns within a case, so the
    // server's own slot-based KV cache reuse kicks in the same way it would in a real
    // harness session, instead of every turn re-processing the prompt from scratch.
    private async Task<(double score, double effectiveTgRate)> RunAgentLoopBenchmarksAsync(
        OptimizationProfile profile, CancellationToken ct, Action<string>? onPhase = null)
    {
        object? toolsPayload = profile.ToolDefinitions.Count > 0
            ? profile.ToolDefinitions.Select(t => new
            {
                type = "function",
                function = new { name = t.Name, description = t.Description, parameters = t.Parameters }
            }).ToArray()
            : null;

        double totalScore = 0;
        long totalTokens = 0;
        double totalWallMs = 0;
        int ran = 0;
        int caseTotal = profile.AgentLoopCases.Count;
        int caseIdx = 0;

        foreach (var loopCase in profile.AgentLoopCases)
        {
            ct.ThrowIfCancellationRequested();
            caseIdx++;
            onPhase?.Invoke($"Running agent-loop scenario ({caseIdx}/{caseTotal})…");
            try
            {
                var (caseScore, tokens, wallMs) = await RunAgentLoopCaseAsync(loopCase, toolsPayload, profile, ct);
                totalScore += caseScore;
                totalTokens += tokens;
                totalWallMs += wallMs;
                ran++;
                logger.LogDebug("Agent loop case scored {Score:F2}: {Prompt}",
                    caseScore, loopCase.Prompt[..Math.Min(50, loopCase.Prompt.Length)]);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Agent loop case failed: {Msg}", ex.Message);
            }
        }

        double score = ran > 0 ? totalScore / profile.AgentLoopCases.Count : 1.0;
        double effectiveTgRate = totalWallMs > 0 ? totalTokens / totalWallMs * 1000.0 : 0.0;
        return (score, effectiveTgRate);
    }

    // Runs one multi-turn case: ExpectedToolSequence.Length tool round-trips (each graded on
    // whether the expected tool was called with valid arguments), followed by a final
    // free-text turn graded against FinalAnswerContains. A case that fails to produce a tool
    // call mid-chain stops there — the remaining expected steps and the final turn score 0,
    // since a real harness would also be stuck at that point.
    //   Weighting: 0.7 * (fraction of expected tool steps completed correctly)
    //            + 0.3 * (final answer correct / coherent)
    private async Task<(double score, long tokens, double wallMs)> RunAgentLoopCaseAsync(
        AgentLoopCase loopCase, object? toolsPayload, OptimizationProfile profile, CancellationToken ct)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = loopCase.Prompt } };
        var sw = Stopwatch.StartNew();
        long tokens = 0;
        double stepCredit = 0;
        int steps = loopCase.ExpectedToolSequence.Length;
        bool aborted = false;

        for (int i = 0; i < steps && !aborted; i++)
        {
            ct.ThrowIfCancellationRequested();
            var (response, _) = await server.ChatCompleteAsync(new ChatCompletionRequest
            {
                Messages = messages,
                Tools = toolsPayload,
                MaxTokens = 512,
                Temperature = 0.1f,
            }, ct);
            tokens += response.Usage?.CompletionTokens ?? 0;

            var call = response.Choices
                .SelectMany(c => c.Message?.ToolCalls ?? [])
                .FirstOrDefault(t => t.Function is not null);
            if (call?.Function is null) { aborted = true; break; }

            var definition = profile.ToolDefinitions
                .FirstOrDefault(d => d.Name.Equals(call.Function.Name, StringComparison.OrdinalIgnoreCase));
            bool validArgs = definition is not null && HasValidArguments(call.Function.Arguments, definition);
            bool expectedTool = call.Function.Name.Equals(loopCase.ExpectedToolSequence[i], StringComparison.OrdinalIgnoreCase);
            stepCredit += !validArgs ? 0.0 : expectedTool ? 1.0 : 0.5;

            messages.Add(new ChatMessage
            {
                Role = "assistant",
                Content = response.Choices[0].Message?.Content,
                ToolCalls = [call],
            });
            messages.Add(new ChatMessage
            {
                Role = "tool",
                ToolCallId = call.Id,
                Content = i < loopCase.ToolResults.Length ? loopCase.ToolResults[i] : "OK",
            });
        }

        double finalScore = 0;
        if (!aborted)
        {
            // Final turn: no more tool calls expected — the model should synthesize an
            // answer from the injected tool results rather than looping further.
            var (finalResponse, _) = await server.ChatCompleteAsync(new ChatCompletionRequest
            {
                Messages = messages,
                Tools = toolsPayload,
                MaxTokens = 512,
                Temperature = 0.1f,
            }, ct);
            tokens += finalResponse.Usage?.CompletionTokens ?? 0;

            string finalText = finalResponse.FirstContent ?? "";
            finalScore = loopCase.FinalAnswerContains.Length == 0
                ? (finalText.Trim().Length > 0 && !finalResponse.HasToolCall ? 1.0 : 0.5)
                : (loopCase.FinalAnswerContains.Any(a =>
                    finalText.Contains(a, StringComparison.OrdinalIgnoreCase)) ? 1.0 : 0.0);
        }

        double stepScore = steps > 0 ? stepCredit / steps : 1.0;
        double caseScore = 0.7 * stepScore + 0.3 * finalScore;
        return (caseScore, tokens, sw.Elapsed.TotalMilliseconds);
    }

    // Runs the profile's accuracy probes and returns the fraction answered correctly.
    // Correct = the response contains any acceptable answer (case-insensitive).
    private async Task<double> RunAccuracyPromptsAsync(OptimizationProfile profile, CancellationToken ct, Action<string>? onPhase = null)
    {
        int correct = 0;
        int total = profile.AccuracyPrompts.Count;
        int idx = 0;
        foreach (var probe in profile.AccuracyPrompts)
        {
            ct.ThrowIfCancellationRequested();
            idx++;
            onPhase?.Invoke($"Checking accuracy ({idx}/{total})…");
            try
            {
                var (response, _) = await server.CompleteAsync(new CompletionRequest
                {
                    Prompt = probe.Prompt,
                    NPredict = 96,
                    Temperature = 0.0f,
                    CachePrompt = false,
                }, ct);

                if (probe.AcceptableAnswers.Any(a =>
                        response.Content.Contains(a, StringComparison.OrdinalIgnoreCase)))
                    correct++;
            }
            catch (Exception ex)
            {
                logger.LogWarning("Accuracy prompt failed: {Msg}", ex.Message);
            }
        }
        return profile.AccuracyPrompts.Count > 0
            ? (double)correct / profile.AccuracyPrompts.Count
            : 1.0;
    }

    private async Task<double> ScoreQualityAsync(
        List<RunTimings> timings, BenchmarkResult result, CancellationToken ct)
    {
        if (timings.Count == 0) return 0.0;

        // A degenerate repetition loop ("is is is is...") means the config produced
        // unusable output — that overrides every other quality signal below.
        if (timings.Any(t => HasRepetitionLoop(t.GeneratedText)))
        {
            result.Notes = (result.Notes is null ? "" : result.Notes + " ") +
                "Repetition loop detected in generated output — quality score zeroed.";
            return 0.0;
        }

        // Heuristic quality score based on:
        // 1. Completion rate (did we get outputs at all?)
        // 2. Minimum token count (did the model produce substantive output?)
        // 3. Error rate penalty
        // 4. Accuracy probes (objectively checkable answers)

        double completionRate = result.SuccessfulRuns /
            (double)Math.Max(1, result.SuccessfulRuns + result.FailedRuns);

        double avgPredicted = timings.Average(t => t.PredictedTokens);
        // 50+ tokens is considered a reasonable response
        double lengthScore = Math.Min(1.0, avgPredicted / 50.0);

        // Penalize very slow generation as it degrades perceived quality
        double speedPenalty = result.GenerationRate < 1.0 ? 0.5 : 1.0;

        // Graded repetition signal: the binary loop check above only fires on catastrophic
        // loops; near-loops (dropping distinct-trigram ratio) depress quality smoothly here,
        // giving the optimizer's anti-loop sampler escalation a measurable gradient. The
        // worst run dominates — one loopy output means the config can produce them.
        double repetitionHealth = timings.Min(t => RepetitionHealth(t.GeneratedText));
        if (repetitionHealth < 0.8)
            result.Notes = (result.Notes is null ? "" : result.Notes + " ") +
                $"Elevated repetition in generated output (health {repetitionHealth:P0}).";

        double fluency = completionRate * lengthScore * speedPenalty * repetitionHealth;

        // Accuracy carries most of the weight: fluency is near-1.0 for almost every config,
        // so without it the quality metric can't distinguish a healthy config from one that
        // quantization/expert-reduction has made subtly wrong.
        return 0.4 * fluency + 0.6 * result.AccuracyScore;
    }

    // Distinct-trigram ratio over the tail of the output, mapped to a 0–1 health score.
    // Natural prose scores ~1.0 (repeated trigrams are rare in a 200-word window); text
    // sliding into a loop drops sharply well before the binary detector below would fire.
    private static double RepetitionHealth(string text)
    {
        var words = text.Split((char[]?)null!, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < 24) return 1.0;   // too short to judge

        const int Window = 200;
        int start = Math.Max(0, words.Length - Window);
        var distinct = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int total = 0;
        for (int i = start; i + 2 < words.Length; i++)
        {
            // Space-joined is collision-free: words are whitespace-split so contain no spaces.
            distinct.Add($"{words[i]} {words[i + 1]} {words[i + 2]}");
            total++;
        }
        if (total == 0) return 1.0;

        // ratio ≥ 0.6 → healthy (1.0); ratio ≤ 0.3 → fully degenerate (0.0)
        double ratio = (double)distinct.Count / total;
        return Math.Clamp((ratio - 0.3) / 0.3, 0, 1);
    }

    // Detects decoding loops where a short word or phrase (1-3 words) repeats back-to-back
    // enough times that it can only be a stuck loop, not natural language repetition.
    private static bool HasRepetitionLoop(string text)
    {
        var words = text.Split((char[]?)null!, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < 12) return false;

        for (int n = 1; n <= 3; n++)
        {
            int run = 0;
            for (int i = n; i < words.Length; i++)
            {
                bool same = string.Equals(words[i], words[i - n], StringComparison.OrdinalIgnoreCase);
                run = same ? run + 1 : 0;
                if (run >= 10) return true;
            }
        }
        return false;
    }
}
