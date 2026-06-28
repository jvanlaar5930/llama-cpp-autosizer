using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class RecommendationService(
    LlamaServerService server,
    ILogger<RecommendationService> logger)
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    // Parameters the optimizer will explore in order of typical impact
    private static readonly string[] ExplorationOrder =
    [
        "GpuLayers", "FlashAttention", "CacheTypeK", "CacheTypeV",
        "BatchSize", "UBatchSize", "ContextSize", "Threads", "Mlock",
    ];

    /// <summary>
    /// Returns the next parameter change to try, using the LLM when possible
    /// and falling back to heuristic exploration.
    /// </summary>
    public async Task<ParameterChange?> GetNextRecommendationAsync(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware,
        CancellationToken ct = default)
    {
        // Try LLM first (it has context we can't express in heuristics)
        if (server.IsRunning && session.Iterations.Count >= 1)
        {
            try
            {
                var llmRec = await GetLlmRecommendationAsync(session, profile, hardware, ct);
                if (llmRec is not null)
                {
                    logger.LogInformation("LLM recommended: {Param} = {Val} — {Reason}",
                        llmRec.Parameter, llmRec.NewValue, llmRec.Reasoning);
                    return llmRec;
                }
            }
            catch (Exception ex)
            {
                logger.LogDebug("LLM recommendation failed, using heuristic: {Msg}", ex.Message);
            }
        }

        return GetHeuristicRecommendation(session, profile, hardware);
    }

    // -------------------------------------------------------------------------
    // LLM-based recommendation
    // -------------------------------------------------------------------------

    private record LlmSuggestion(
        [property: JsonPropertyName("parameter")] string Parameter,
        [property: JsonPropertyName("value")] JsonElement Value,
        [property: JsonPropertyName("reasoning")] string Reasoning
    );

    private async Task<ParameterChange?> GetLlmRecommendationAsync(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware,
        CancellationToken ct)
    {
        var best = session.BestResult;
        var current = session.Iterations.Last();
        var settings = current.Settings;

        var prompt = BuildRecommendationPrompt(session, profile, hardware);

        var (response, _) = await server.CompleteAsync(new CompletionRequest
        {
            Prompt = prompt,
            NPredict = 256,
            Temperature = 0.3f,
            Stop = ["\n\n", "```"],
        }, ct);

        return ParseLlmResponse(response.Content, current.Settings);
    }

    private static string BuildRecommendationPrompt(
        OptimizationSession session, OptimizationProfile profile, HardwareInfo hardware)
    {
        var best = session.BestResult!;
        var current = session.Iterations.Last();

        var history = string.Join("\n", session.Iterations.TakeLast(5).Select(i =>
            $"  iter={i.Number} score={i.Result.CompositeScore:F3} " +
            $"PP={i.Result.PromptProcessingRate:F0}t/s TG={i.Result.GenerationRate:F0}t/s " +
            $"TTFT={i.Result.TimeToFirstTokenMs:F0}ms change={i.AppliedChange?.Parameter ?? "baseline"}"));

        return $$"""
You are a llama.cpp performance optimizer. Analyze the benchmark history and suggest ONE parameter change.

HARDWARE:
- GPU VRAM: {{hardware.TotalVramMb}} MB (free: {{hardware.FreeVramMb}} MB)
- RAM: {{hardware.RamTotalMb}} MB (free: {{hardware.RamFreeMb}} MB)
- CPU threads: {{hardware.CpuThreads}}

PROFILE: {{profile.Name}} ({{profile.Description}})

CURRENT SETTINGS:
- ctx-size: {{current.Settings.ContextSize}}
- n-gpu-layers: {{current.Settings.GpuLayers}}
- batch-size: {{current.Settings.BatchSize}}
- ubatch-size: {{current.Settings.UBatchSize}}
- flash-attn: {{current.Settings.FlashAttention}}
- cache-type-k: {{current.Settings.CacheTypeK ?? "f16"}}
- cache-type-v: {{current.Settings.CacheTypeV ?? "f16"}}
- threads: {{current.Settings.Threads}}

BENCHMARK HISTORY (last 5 iterations):
{{history}}

BEST SCORE: {{best.CompositeScore:F3}}
BEST METRICS: PP={{best.PromptProcessingRate:F0}}t/s TG={{best.GenerationRate:F0}}t/s TTFT={{best.TimeToFirstTokenMs:F0}}ms

Suggest ONE change that is likely to improve the score for the {{profile.Name}} profile.
Valid parameters: GpuLayers (int), ContextSize (int, powers of 2), BatchSize (int), UBatchSize (int),
FlashAttention (true/false), CacheTypeK (f16/q8_0/q4_0/q5_0), CacheTypeV (f16/q8_0/q4_0/q5_0),
Threads (int, -1=auto), Mlock (true/false).

Respond with ONLY this JSON (no markdown, no explanation outside JSON):
{"parameter":"<name>","value":<value>,"reasoning":"<one sentence>"}
""";
    }

    private static ParameterChange? ParseLlmResponse(string content, LlamaSettings current)
    {
        // Find JSON in the response
        var start = content.IndexOf('{');
        var end = content.LastIndexOf('}');
        if (start < 0 || end <= start) return null;

        try
        {
            var json = content[start..(end + 1)];
            var suggestion = JsonSerializer.Deserialize<LlmSuggestion>(json);
            if (suggestion is null || string.IsNullOrEmpty(suggestion.Parameter)) return null;

            var (oldVal, newVal) = ApplyToSettings(current.Clone(), suggestion.Parameter, suggestion.Value);
            if (newVal is null) return null;

            return new ParameterChange
            {
                Parameter = suggestion.Parameter,
                OldValue = oldVal,
                NewValue = newVal,
                Reasoning = suggestion.Reasoning,
                Source = "llm",
            };
        }
        catch
        {
            return null;
        }
    }

    private static (object? oldVal, object? newVal) ApplyToSettings(
        LlamaSettings settings, string parameter, JsonElement value)
    {
        try
        {
            switch (parameter)
            {
                case "GpuLayers":
                    int ngl = value.GetInt32();
                    return (settings.GpuLayers, ngl != settings.GpuLayers ? ngl : (object?)null);
                case "ContextSize":
                    int ctx = value.GetInt32();
                    return (settings.ContextSize, ctx != settings.ContextSize ? ctx : (object?)null);
                case "BatchSize":
                    int bs = value.GetInt32();
                    return (settings.BatchSize, bs != settings.BatchSize ? bs : (object?)null);
                case "UBatchSize":
                    int ubs = value.GetInt32();
                    return (settings.UBatchSize, ubs != settings.UBatchSize ? ubs : (object?)null);
                case "FlashAttention":
                    bool fa = value.GetBoolean();
                    return (settings.FlashAttention, fa != settings.FlashAttention ? fa : (object?)null);
                case "CacheTypeK":
                    string ktk = value.GetString() ?? "f16";
                    return (settings.CacheTypeK ?? "f16", ktk != (settings.CacheTypeK ?? "f16") ? ktk : (object?)null);
                case "CacheTypeV":
                    string ktv = value.GetString() ?? "f16";
                    return (settings.CacheTypeV ?? "f16", ktv != (settings.CacheTypeV ?? "f16") ? ktv : (object?)null);
                case "Threads":
                    int t = value.GetInt32();
                    return (settings.Threads, t != settings.Threads ? t : (object?)null);
                case "Mlock":
                    bool ml = value.GetBoolean();
                    return (settings.Mlock, ml != settings.Mlock ? ml : (object?)null);
                default:
                    return (null, null);
            }
        }
        catch
        {
            return (null, null);
        }
    }

    // -------------------------------------------------------------------------
    // Heuristic recommendation
    // -------------------------------------------------------------------------

    public ParameterChange? GetHeuristicRecommendation(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware)
    {
        var tried = session.Iterations
            .Where(i => i.AppliedChange is not null)
            .Select(i => i.AppliedChange!.Parameter)
            .ToHashSet();

        var best = session.BestResult!;
        var bestSettings = session.BestSettings!;

        // If VRAM is available and GPU layers aren't maxed, try more GPU layers
        if (!tried.Contains("GpuLayers") && hardware.HasGpu)
        {
            var suggestedNgl = Math.Min(bestSettings.GpuLayers + 8, 100);
            if (suggestedNgl != bestSettings.GpuLayers)
                return Change("GpuLayers", bestSettings.GpuLayers, suggestedNgl,
                    "More GPU layers offloads computation to VRAM, improving TG speed");
        }

        // Try flash attention if not enabled
        if (!tried.Contains("FlashAttention") && !bestSettings.FlashAttention)
            return Change("FlashAttention", false, true,
                "Flash attention reduces VRAM usage and often improves throughput");

        // Try KV cache quantization to free VRAM for more context or layers
        if (!tried.Contains("CacheTypeK") && bestSettings.CacheTypeK is null or "f16")
            return Change("CacheTypeK", bestSettings.CacheTypeK ?? "f16", "q8_0",
                "q8_0 KV cache halves KV cache VRAM with minimal quality loss");

        if (!tried.Contains("CacheTypeV") && bestSettings.CacheTypeV is null or "f16")
            return Change("CacheTypeV", bestSettings.CacheTypeV ?? "f16", "q8_0",
                "q8_0 KV cache halves KV cache VRAM with minimal quality loss");

        // Tune batch size based on profile
        if (!tried.Contains("BatchSize"))
        {
            int newBs = profile.Type == ProfileType.Chat ? 256 : 1024;
            if (newBs != bestSettings.BatchSize)
                return Change("BatchSize", bestSettings.BatchSize, newBs,
                    profile.Type == ProfileType.Chat
                        ? "Smaller batch size reduces TTFT for chat workloads"
                        : "Larger batch size improves PP throughput for agentic workloads");
        }

        // Increase context for agentic profile
        if (!tried.Contains("ContextSize") && profile.Type == ProfileType.Agentic
            && bestSettings.ContextSize < profile.TargetContextSize)
        {
            int newCtx = Math.Min(bestSettings.ContextSize * 2, profile.TargetContextSize);
            return Change("ContextSize", bestSettings.ContextSize, newCtx,
                "Larger context enables longer agentic task chains");
        }

        // Try even more aggressive KV quant if first round helped
        if (tried.Contains("CacheTypeK") && bestSettings.CacheTypeK == "q8_0")
            return Change("CacheTypeK", "q8_0", "q4_0",
                "q4_0 further reduces KV cache VRAM at some quality cost — worth testing");

        // Adjust threads
        if (!tried.Contains("Threads"))
        {
            int suggestedThreads = Math.Max(1, hardware.CpuCores / 2);
            if (suggestedThreads != bestSettings.Threads)
                return Change("Threads", bestSettings.Threads, suggestedThreads,
                    "Tuning thread count to physical cores can reduce scheduling overhead");
        }

        return null; // Nothing left to try
    }

    private static ParameterChange Change(string param, object oldVal, object newVal, string reason)
        => new() { Parameter = param, OldValue = oldVal, NewValue = newVal, Reasoning = reason, Source = "heuristic" };

    /// <summary>Applies a ParameterChange to a cloned copy of the settings.</summary>
    public static LlamaSettings Apply(LlamaSettings settings, ParameterChange change)
    {
        var s = settings.Clone();
        switch (change.Parameter)
        {
            case "GpuLayers":       s.GpuLayers = Convert.ToInt32(change.NewValue); break;
            case "ContextSize":     s.ContextSize = Convert.ToInt32(change.NewValue); break;
            case "BatchSize":       s.BatchSize = Convert.ToInt32(change.NewValue); break;
            case "UBatchSize":      s.UBatchSize = Convert.ToInt32(change.NewValue); break;
            case "FlashAttention":  s.FlashAttention = Convert.ToBoolean(change.NewValue); break;
            case "CacheTypeK":      s.CacheTypeK = change.NewValue?.ToString(); break;
            case "CacheTypeV":      s.CacheTypeV = change.NewValue?.ToString(); break;
            case "Threads":         s.Threads = Convert.ToInt32(change.NewValue); break;
            case "Mlock":           s.Mlock = Convert.ToBoolean(change.NewValue); break;
        }
        return s;
    }
}
