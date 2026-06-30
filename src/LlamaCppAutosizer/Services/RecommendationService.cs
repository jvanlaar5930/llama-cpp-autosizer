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

    /// <summary>
    /// "Final push" prompt — used when 3 consecutive iterations haven't improved or when the
    /// normal pool is exhausted. Asks the LLM for a completely fresh angle; if it responds
    /// with DONE (or fails to suggest anything) returns null, signalling the loop should end.
    /// </summary>
    public async Task<ParameterChange?> GetFinalPushAsync(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware,
        CancellationToken ct = default)
    {
        if (!server.IsRunning)
            return GetHeuristicRecommendation(session, profile, hardware);

        try
        {
            var prompt = BuildFinalPushPrompt(session, profile, hardware);
            var (response, _) = await server.CompleteAsync(new CompletionRequest
            {
                Prompt = prompt,
                NPredict = 350,
                Temperature = 0.5f,
                Stop = ["\n\n", "```"],
            }, ct);

            var content = response.Content.Trim();

            // If the LLM responds with "DONE" and no JSON, it's saying nothing more to try
            bool hasDone = content.Contains("DONE", StringComparison.OrdinalIgnoreCase);
            bool hasJson = content.Contains('{') && content.Contains('}');
            if (hasDone && !hasJson) return null;

            var change = ParseLlmResponse(content, session.BestSettings!);
            if (change is null) return null;

            // Tag it so the UI can show it as a "final push" iteration
            return new ParameterChange
            {
                Parameter = change.Parameter,
                OldValue = change.OldValue,
                NewValue = change.NewValue,
                Reasoning = change.Reasoning,
                Source = "llm-push",
            };
        }
        catch (Exception ex)
        {
            logger.LogDebug("Final-push LLM call failed: {Msg}", ex.Message);
            // Fall back to heuristic so we don't waste an iteration slot
            return GetHeuristicRecommendation(session, profile, hardware);
        }
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
        var bestSettings = session.Best!.Settings;

        // Explicit per-iteration log: what was changed, did it help?
        double baselineScore = session.Iterations[0].Result.CompositeScore;
        var historyLines = session.Iterations.Select(i =>
        {
            if (i.Number == 0)
                return $"  [baseline] score={i.Result.CompositeScore:F3}  " +
                       $"TG={i.Result.GenerationRate:F0}t/s PP={i.Result.PromptProcessingRate:F0}t/s TTFT={i.Result.TimeToFirstTokenMs:F0}ms";
            double delta = i.Result.CompositeScore - baselineScore;
            string verdict = i.Result.CompositeScore >= best.CompositeScore ? "★ BEST"
                           : delta >= 0 ? $"+{delta:F3} vs baseline"
                           : $"{delta:F3} vs baseline (worse)";
            return $"  [iter {i.Number}] {i.AppliedChange?.Parameter ?? "?"}={i.AppliedChange?.NewValue}  " +
                   $"score={i.Result.CompositeScore:F3} ({verdict})  " +
                   $"TG={i.Result.GenerationRate:F0}t/s PP={i.Result.PromptProcessingRate:F0}t/s TTFT={i.Result.TimeToFirstTokenMs:F0}ms";
        });

        bool isMoe = LlamaSettings.IsMoeModel(session.ModelPath);
        string moeContext = isMoe
            ? $"\nMODEL TYPE: Mixture-of-Experts (MoE). Active experts/token: " +
              $"{(bestSettings.MoeExpertUsed.HasValue ? bestSettings.MoeExpertUsed.Value : "model default")}. " +
              $"Reducing active experts saves VRAM and speeds inference at some quality cost."
            : "";

        string moeParam = isMoe
            ? $"\n- MoeExpertUsed: integer, e.g. 4, 6, 8, 12 (fewer experts = less VRAM + faster; current={bestSettings.MoeExpertUsed?.ToString() ?? "model default"})"
            : "";

        return $$"""
You are a llama.cpp performance optimizer. Your task is to find the best server settings for this model on this hardware by trying one change at a time.
Study the results so far and suggest ONE parameter change most likely to improve the composite score.
You may suggest a different value for a parameter already tried if you think another value will do better.

HARDWARE:
- GPU: {{(hardware.HasGpu ? $"{hardware.Gpus.FirstOrDefault()?.Name ?? "GPU"} — {hardware.TotalVramMb} MB total, {hardware.FreeVramMb} MB free" : "none (CPU-only)")}}
- RAM: {{hardware.RamTotalMb}} MB total, {{hardware.RamFreeMb}} MB free
- CPU: {{hardware.CpuName}} — {{hardware.CpuCores}} cores / {{hardware.CpuThreads}} threads

PROFILE: {{profile.Name}} — {{profile.Description}}{{moeContext}}

BEST SETTINGS SO FAR (score={{best.CompositeScore:F3}} | TG={{best.GenerationRate:F0}}t/s PP={{best.PromptProcessingRate:F0}}t/s TTFT={{best.TimeToFirstTokenMs:F0}}ms):
- ctx-size: {{bestSettings.ContextSize}}
- n-gpu-layers: {{bestSettings.GpuLayers}} (-1 = all layers)
- batch-size: {{bestSettings.BatchSize}} / ubatch-size: {{bestSettings.UBatchSize}}
- flash-attn: {{bestSettings.FlashAttention}}
- cache-type-k: {{bestSettings.CacheTypeK ?? "f16"}} / cache-type-v: {{bestSettings.CacheTypeV ?? "f16"}}
- threads: {{bestSettings.Threads}} (-1 = auto)
- mlock: {{bestSettings.Mlock}}

FULL OPTIMIZATION HISTORY:
{{string.Join("\n", historyLines)}}

GOAL: Improve composite score above {{best.CompositeScore:F3}}.
RULE: If a parameter made things worse, try something else OR a different value for it.

Valid parameters and example values:
- GpuLayers: -1 (all layers), or a positive integer up to 200
- ContextSize: power of 2 — 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
- BatchSize: 64, 128, 256, 512, 1024, 2048
- UBatchSize: 32, 64, 128, 256, 512, 1024
- FlashAttention: true or false
- CacheTypeK: "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"
- CacheTypeV: "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"
- Threads: -1 (auto), or 1–{{hardware.CpuThreads}}
- Mlock: true or false{{moeParam}}

Respond with ONLY this JSON (no markdown, no extra text before or after):
{"parameter":"<name>","value":<value>,"reasoning":"<one sentence explaining why this should help>"}
""";
    }

    private static string BuildFinalPushPrompt(
        OptimizationSession session, OptimizationProfile profile, HardwareInfo hardware)
    {
        var best = session.BestResult!;
        var bestSettings = session.Best!.Settings;
        double baselineScore = session.Iterations[0].Result.CompositeScore;
        bool isMoe = LlamaSettings.IsMoeModel(session.ModelPath);

        var triedSummary = session.Iterations
            .Where(i => i.AppliedChange is not null)
            .Select(i => $"  {i.AppliedChange!.Parameter}={i.AppliedChange.NewValue} → score={i.Result.CompositeScore:F3} ({(i.IsBestSoFar ? "★ best" : i.Result.CompositeScore >= baselineScore ? "ok" : "worse")})")
            .ToList();

        string moeParam = isMoe
            ? $"\n- MoeExpertUsed: integer (e.g. 4, 6, 8, 12) — fewer experts saves VRAM and increases speed"
            : "";

        return $$"""
You are a llama.cpp performance optimizer. After {{triedSummary.Count}} attempts, the last 3 iterations produced no improvement.

HARDWARE:
- GPU: {{(hardware.HasGpu ? $"{hardware.Gpus.FirstOrDefault()?.Name ?? "GPU"} — {hardware.TotalVramMb} MB total, {hardware.FreeVramMb} MB free" : "none (CPU-only)")}}
- RAM: {{hardware.RamTotalMb}} MB total, {{hardware.RamFreeMb}} MB free
- CPU: {{hardware.CpuName}} — {{hardware.CpuCores}} cores / {{hardware.CpuThreads}} threads

PROFILE: {{profile.Name}} — {{profile.Description}}

BEST SETTINGS (score={{best.CompositeScore:F3}} vs baseline={{baselineScore:F3}} | TG={{best.GenerationRate:F0}}t/s PP={{best.PromptProcessingRate:F0}}t/s TTFT={{best.TimeToFirstTokenMs:F0}}ms):
- ctx-size: {{bestSettings.ContextSize}}
- n-gpu-layers: {{bestSettings.GpuLayers}} (-1 = all layers)
- batch-size: {{bestSettings.BatchSize}} / ubatch-size: {{bestSettings.UBatchSize}}
- flash-attn: {{bestSettings.FlashAttention}}
- cache-type-k: {{bestSettings.CacheTypeK ?? "f16"}} / cache-type-v: {{bestSettings.CacheTypeV ?? "f16"}}
- threads: {{bestSettings.Threads}} / mlock: {{bestSettings.Mlock}}

ALREADY TRIED:
{{string.Join("\n", triedSummary)}}

We need a fresh approach. Think carefully about:
- Whether a completely different GPU layer count might hit a sweet spot
- Context sizes not yet tried (both smaller for speed, larger if VRAM permits)
- Combinations we haven't explored (e.g. flash-attn + smaller ubatch)
- KV cache quantization levels not yet tried
- Thread count tuning for CPU-bound operations
- Mlock if the model fits in RAM
- Ubatch vs batch ratios{{moeParam}}

If you genuinely believe no further optimization is possible given what has been tried and the hardware constraints, respond with exactly:
DONE

Otherwise respond with ONLY this JSON (no markdown, no extra text):
{"parameter":"<name>","value":<value>,"reasoning":"<specific reason this new angle is worth trying>"}
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

    // All LLM-suggested values pass through here before being applied.
    // Clamps every parameter to a safe range so a hallucinated value can't
    // request absurd VRAM, zero threads, or a negative context window.
    private static (object? oldVal, object? newVal) ApplyToSettings(
        LlamaSettings settings, string parameter, JsonElement value)
    {
        try
        {
            switch (parameter)
            {
                case "GpuLayers":
                    // -1 = offload all (valid); positive values capped at 200
                    // (no known model has more layers than that)
                    int ngl = Math.Clamp(value.GetInt32(), -1, 200);
                    return (settings.GpuLayers, ngl != settings.GpuLayers ? ngl : (object?)null);

                case "ContextSize":
                    // 512 minimum (anything less is unusable), 262144 maximum (256k)
                    int ctx = Math.Clamp(value.GetInt32(), 512, 262144);
                    return (settings.ContextSize, ctx != settings.ContextSize ? ctx : (object?)null);

                case "BatchSize":
                    // 1–4096; common values are powers of 2
                    int bs = Math.Clamp(value.GetInt32(), 1, 4096);
                    return (settings.BatchSize, bs != settings.BatchSize ? bs : (object?)null);

                case "UBatchSize":
                    int ubs = Math.Clamp(value.GetInt32(), 1, 4096);
                    return (settings.UBatchSize, ubs != settings.UBatchSize ? ubs : (object?)null);

                case "FlashAttention":
                    bool fa = value.GetBoolean();
                    return (settings.FlashAttention, fa != settings.FlashAttention ? fa : (object?)null);

                case "CacheTypeK":
                    string ktk = value.GetString() ?? "f16";
                    // Only allow known-valid types; reject anything else silently
                    if (!LlamaSettings.ValidCacheTypes.Concat(LlamaSettings.TurboQuantCacheTypes).Contains(ktk))
                        return (null, null);
                    return (settings.CacheTypeK ?? "f16", ktk != (settings.CacheTypeK ?? "f16") ? ktk : (object?)null);

                case "CacheTypeV":
                    string ktv = value.GetString() ?? "f16";
                    if (!LlamaSettings.ValidCacheTypes.Concat(LlamaSettings.TurboQuantCacheTypes).Contains(ktv))
                        return (null, null);
                    return (settings.CacheTypeV ?? "f16", ktv != (settings.CacheTypeV ?? "f16") ? ktv : (object?)null);

                case "Threads":
                    // At least 1, at most the number of logical processors on the machine
                    int maxThreads = Math.Max(1, Environment.ProcessorCount);
                    int t = Math.Clamp(value.GetInt32(), 1, maxThreads);
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
        // Parameters tried so far (by name — for single-shot explorations)
        var tried = session.Iterations
            .Where(i => i.AppliedChange is not null)
            .Select(i => i.AppliedChange!.Parameter)
            .ToHashSet();

        var best = session.BestResult!;
        var bestSettings = session.BestSettings!;
        double baselineScore = session.Iterations[0].Result.CompositeScore;
        bool scoreImproved = best.CompositeScore > baselineScore;

        // If VRAM is available and GPU layers aren't already maxed (-1 = all), try more
        if (!tried.Contains("GpuLayers") && hardware.HasGpu && bestSettings.GpuLayers != -1)
        {
            var suggestedNgl = Math.Min(bestSettings.GpuLayers + 8, 200);
            if (suggestedNgl > bestSettings.GpuLayers)
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

        // Progressive context expansion — both profiles.
        // Step up by 2× each time, but only when score has improved from baseline,
        // and only if this specific target value hasn't already been tried.
        if (bestSettings.ContextSize < profile.TargetContextSize && scoreImproved)
        {
            int nextCtx = Math.Min(bestSettings.ContextSize * 2, profile.TargetContextSize);
            bool alreadyTriedThisValue = session.Iterations.Any(i =>
                i.AppliedChange?.Parameter == "ContextSize" &&
                Convert.ToInt32(i.AppliedChange.NewValue) == nextCtx);

            if (!alreadyTriedThisValue)
                return Change("ContextSize", bestSettings.ContextSize, nextCtx,
                    $"Score is improving — expanding context {bestSettings.ContextSize:N0} → {nextCtx:N0} tokens");
        }

        // Escalate KV quant if the first round helped
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

        // If context hasn't been expanded yet (score hasn't improved enough to trigger
        // the gate above) but we've run out of other ideas, try it anyway as a last resort.
        if (bestSettings.ContextSize < profile.TargetContextSize)
        {
            int nextCtx = Math.Min(bestSettings.ContextSize * 2, profile.TargetContextSize);
            bool alreadyTriedThisValue = session.Iterations.Any(i =>
                i.AppliedChange?.Parameter == "ContextSize" &&
                Convert.ToInt32(i.AppliedChange.NewValue) == nextCtx);

            if (!alreadyTriedThisValue)
                return Change("ContextSize", bestSettings.ContextSize, nextCtx,
                    $"Trying larger context {bestSettings.ContextSize:N0} → {nextCtx:N0} tokens as a final exploration");
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
