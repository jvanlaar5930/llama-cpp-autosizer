using System.Text.Json;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class BenchmarkService(
    LlamaServerService server,
    ILogger<BenchmarkService> logger)
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    public async Task<BenchmarkResult> RunAsync(
        LlamaSettings settings,
        string modelPath,
        OptimizationProfile profile,
        CancellationToken ct = default,
        bool includeRepetitionStressTest = false)
    {
        var result = new BenchmarkResult
        {
            Settings = settings.Clone(),
            ModelPath = modelPath,
            Timestamp = DateTime.UtcNow,
        };

        // Warmup
        logger.LogDebug("Running {N} warmup prompts", profile.WarmupRuns);
        foreach (var prompt in profile.WarmupPrompts.Take(profile.WarmupRuns))
        {
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
        foreach (var prompt in profile.BenchmarkPrompts)
        {
            ct.ThrowIfCancellationRequested();
            var runTiming = await RunSinglePromptAsync(prompt, profile.MaxGenerateTokens, result, ct);
            if (runTiming is not null) timings.Add(runTiming);
        }

        // Opt-in long-form run so the repetition-loop check below has enough tokens to
        // actually catch a degenerate config — short benchmark prompts rarely run long
        // enough to reveal it even though real long-form usage frequently does.
        if (includeRepetitionStressTest && !string.IsNullOrWhiteSpace(profile.StressTestPrompt))
        {
            ct.ThrowIfCancellationRequested();
            var stressTiming = await RunSinglePromptAsync(profile.StressTestPrompt, profile.StressTestMaxTokens, result, ct);
            if (stressTiming is not null) timings.Add(stressTiming);
        }

        // Agentic tool-use benchmarks
        double toolSuccessRate = 1.0;
        if (profile.Type == ProfileType.Agentic && profile.ToolBenchmarkPrompts.Count > 0)
        {
            toolSuccessRate = await RunToolBenchmarksAsync(profile, result, ct);
        }

        // Aggregate. Rates are weighted by tokens/time (not averaged per-run) because a
        // run that generates very few tokens in a tiny amount of time reports an inflated
        // per_second ratio that would otherwise dominate a plain average.
        if (timings.Count > 0)
        {
            double promptMs = timings.Sum(t => t.PromptMs);
            double predictedMs = timings.Sum(t => t.PredictedMs);
            result.PromptProcessingRate = promptMs > 0
                ? timings.Sum(t => t.PromptTokens) / promptMs * 1000.0
                : 0;
            result.GenerationRate = predictedMs > 0
                ? timings.Sum(t => t.PredictedTokens) / predictedMs * 1000.0
                : 0;
            result.TimeToFirstTokenMs = timings.Average(t => t.TimeToFirstTokenMs);
            result.AverageTotalMs = timings.Average(t => t.TotalMs);
        }

        result.SuccessfulRuns = timings.Count;
        result.FailedRuns = profile.BenchmarkPrompts.Count - timings.Count;
        result.ToolSuccessRate = toolSuccessRate;
        result.QualityScore = await ScoreQualityAsync(timings, result, ct);

        return result;
    }

    private async Task<RunTimings?> RunSinglePromptAsync(
        string prompt, int maxTokens, BenchmarkResult result, CancellationToken ct)
    {
        try
        {
            var (response, ttftMs) = await server.CompleteAsync(new CompletionRequest
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
                TimeToFirstTokenMs = ttftMs,
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

    private async Task<double> RunToolBenchmarksAsync(
        OptimizationProfile profile, BenchmarkResult result, CancellationToken ct)
    {
        int successes = 0;
        int total = profile.ToolBenchmarkPrompts.Count;

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

        foreach (var prompt in profile.ToolBenchmarkPrompts)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var (response, _) = await server.ChatCompleteAsync(new ChatCompletionRequest
                {
                    Messages = [new ChatMessage { Role = "user", Content = prompt }],
                    Tools = toolsPayload,
                    MaxTokens = 512,
                    Temperature = 0.1f,
                }, ct);

                // Success = model either used a tool OR gave a coherent response
                bool used = response.HasToolCall;
                bool coherent = !string.IsNullOrWhiteSpace(response.FirstContent);
                if (used || coherent) successes++;

                if (used)
                    logger.LogDebug("Tool call detected for: {Prompt}", prompt[..Math.Min(50, prompt.Length)]);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Tool benchmark failed: {Msg}", ex.Message);
            }
        }

        return total > 0 ? (double)successes / total : 1.0;
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

        // Simple heuristic quality score based on:
        // 1. Completion rate (did we get outputs at all?)
        // 2. Minimum token count (did the model produce substantive output?)
        // 3. Error rate penalty

        double completionRate = result.SuccessfulRuns /
            (double)Math.Max(1, result.SuccessfulRuns + result.FailedRuns);

        double avgPredicted = timings.Average(t => t.PredictedTokens);
        // 50+ tokens is considered a reasonable response
        double lengthScore = Math.Min(1.0, avgPredicted / 50.0);

        // Penalize very slow generation as it degrades perceived quality
        double speedPenalty = result.GenerationRate < 1.0 ? 0.5 : 1.0;

        return completionRate * lengthScore * speedPenalty;
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
