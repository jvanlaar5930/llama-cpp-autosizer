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
                if (llmRec is not null && RecreatesTestedConfig(session, llmRec))
                {
                    logger.LogInformation(
                        "LLM suggested {Param} = {Val}, but that configuration was already benchmarked — falling back to heuristic",
                        llmRec.Parameter, llmRec.NewValue);
                    llmRec = null;
                }
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
    /// True if applying <paramref name="change"/> to the current best settings would recreate
    /// a configuration this session has already benchmarked (or attempted). Such a change would
    /// waste an iteration re-measuring a known result — every recommendation path rejects it.
    /// </summary>
    private static bool RecreatesTestedConfig(OptimizationSession session, ParameterChange change)
        => session.BestSettings is not null
           && session.HasTestedConfiguration(Apply(session.BestSettings, change));

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

            if (RecreatesTestedConfig(session, change))
            {
                logger.LogInformation(
                    "Final-push suggestion {Change} recreates an already-benchmarked configuration — falling back to heuristic",
                    change.Describe());
                return GetHeuristicRecommendation(session, profile, hardware);
            }

            // Tag it so the UI can show it as a "final push" iteration
            return new ParameterChange
            {
                Parameter = change.Parameter,
                OldValue = change.OldValue,
                NewValue = change.NewValue,
                Reasoning = change.Reasoning,
                Source = "llm-push",
                Combined = change.Combined,
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
        [property: JsonPropertyName("reasoning")] string Reasoning,
        // Optional second change, only advertised in the final-push prompt — lets the LLM
        // explore combinations (e.g. flash-attn + smaller ubatch) one greedy step can't reach.
        [property: JsonPropertyName("parameter2")] string? Parameter2 = null,
        [property: JsonPropertyName("value2")] JsonElement? Value2 = null
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
            string changeDesc = i.AppliedChange?.Describe() ?? "?";
            if (i.Result.GenerationRate <= 0)
                return $"  [iter {i.Number}] {changeDesc}  FAILED (server would not start or benchmark errored — do not retry this)";
            double delta = i.Result.CompositeScore - baselineScore;
            string verdict = i.Result.CompositeScore >= best.CompositeScore ? "★ BEST"
                           : delta >= 0 ? $"+{delta:F3} vs baseline"
                           : $"{delta:F3} vs baseline (worse)";
            return $"  [iter {i.Number}] {changeDesc}  " +
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

        // Thinking-capable models (Qwen3/QwQ/DeepSeek-R1) start every run with reasoning
        // forced off for benchmark speed/consistency — CoT tokens would otherwise dominate
        // every timing. Surface it as an explorable parameter so the LLM can trade some of
        // that speed back for reasoning quality when the profile/target calls for it
        // (e.g. Agentic tool-call accuracy), rather than it being permanently locked off.
        bool isThinking = LlamaSettings.IsThinkingModel(session.ModelPath);
        string thinkingContext = isThinking
            ? $"\nMODEL TYPE: Thinking/reasoning model (chain-of-thought). Currently thinking=" +
              $"{(bestSettings.ThinkingEnabled ?? false)}. Disabled by default at baseline for benchmark " +
              $"speed — enabling it adds reasoning tokens before the answer, which costs speed but can " +
              $"improve answer/tool-call correctness."
            : "";

        string thinkingParam = isThinking
            ? "\n- ThinkingEnabled: true or false (enabling costs TG speed and TTFT but can improve reasoning/tool-call correctness)"
            : "";

        string guidance = string.IsNullOrWhiteSpace(session.UserGuidance)
            ? ""
            : $"\nUSER GUIDANCE (follow this even if it means a different tradeoff than the profile default): {session.UserGuidance}\n";

        return $$"""
You are a llama.cpp performance optimizer. Your task is to find the best server settings for this model on this hardware by trying one change at a time.
Study the results so far and suggest ONE parameter change most likely to improve the composite score.
You may suggest a different value for a parameter already tried if you think another value will do better.

HARDWARE:
- GPU: {{(hardware.HasGpu ? $"{hardware.Gpus.FirstOrDefault()?.Name ?? "GPU"} — {hardware.TotalVramMb} MB total, {hardware.FreeVramMb} MB free (measured with the current model loaded — this is real remaining headroom)" : "none (CPU-only)")}}
- RAM: {{hardware.RamTotalMb}} MB total, {{hardware.RamFreeMb}} MB free
- CPU: {{hardware.CpuName}} — {{hardware.CpuCores}} cores / {{hardware.CpuThreads}} threads

PROFILE: {{profile.Name}} — {{profile.Description}}{{moeContext}}{{thinkingContext}}
{{guidance}}

BEST SETTINGS SO FAR (score={{best.CompositeScore:F3}} | TG={{best.GenerationRate:F0}}t/s PP={{best.PromptProcessingRate:F0}}t/s TTFT={{best.TimeToFirstTokenMs:F0}}ms):
- ctx-size: {{bestSettings.ContextSize}}
- n-gpu-layers: {{bestSettings.GpuLayers}} (-1 = all layers)
- batch-size: {{bestSettings.BatchSize}} / ubatch-size: {{bestSettings.UBatchSize}}
- flash-attn: {{bestSettings.FlashAttention}}
- cache-type-k: {{bestSettings.CacheTypeK ?? "f16"}} / cache-type-v: {{bestSettings.CacheTypeV ?? "f16"}}
- threads: {{bestSettings.Threads}} (-1 = auto)
- mmap: {{bestSettings.Mmap}} / mlock: {{bestSettings.Mlock}}
- repeat-penalty: {{bestSettings.RepeatPenalty?.ToString() ?? "server default"}} / repeat-last-n: {{bestSettings.RepeatLastN?.ToString() ?? "server default"}} / dry-multiplier: {{bestSettings.DryMultiplier?.ToString() ?? "server default"}}{{(isThinking ? $"\n- thinking: {bestSettings.ThinkingEnabled ?? false}" : "")}}

FULL OPTIMIZATION HISTORY:
{{string.Join("\n", historyLines)}}

GOAL: Target generation speed is {{session.TargetTgSpeed:F0}}t/s.
- Below {{session.TargetTgSpeed:F0}}t/s: prioritize whatever raises TG speed most (GPU layers, flash-attn, mmap, KV cache quantization, batch size).
- At or above {{session.TargetTgSpeed * QualityPhaseBufferFactor:F0}}t/s (target met with buffer): switch to QUALITY TUNING and spend the spare speed, in this order: (1) revert aggressive KV cache quantization (q4/q5-class → q8_0; q8_0 → f16 only with a lot of headroom) — low-bit KV quant is a common cause of repetition loops and degraded tool calls; (2) if quality/tool/agent-loop scores are below ~0.9, strengthen anti-loop samplers (DryMultiplier, RepeatPenalty); (3) for agentic profiles on thinking-capable models, enable ThinkingEnabled; (4) grow ContextSize (in +16,384 steps) for more capability.
- Well above {{session.TargetTgSpeed:F0}}t/s (lots of headroom, e.g. {{session.TargetTgSpeed + LotsOfHeadroomAboveTarget:F0}}t/s+): it's fine to give back more speed for quality — e.g. restore MoE active-expert count toward the model default — as long as TG stays at/above {{session.TargetTgSpeed:F0}}t/s.
- If the quality score has cratered on the current best (a strong sign of a degenerate repetition loop like "is is is is..."), prioritize RepeatPenalty (e.g. 1.1–1.3), RepeatLastN (e.g. 256), or DryMultiplier (e.g. 0.8) over speed tuning — an unusable output is worse than a faster one.
RULE: If a parameter made things worse, try something else OR a different value for it. Below the speed target, treat cache-type-k/cache-type-v purely as a speed/VRAM lever; at/above the target, reverting quantized KV cache toward q8_0/f16 to protect quality is encouraged.
RULE: Never re-suggest a parameter=value pair from the history above unless the best settings have changed since it was tried — recreating an already-benchmarked configuration wastes an iteration and will be rejected.

Valid parameters and example values:
- GpuLayers: -1 (all layers), or a positive integer up to 200
- ContextSize: {{bestSettings.ContextSize + 16384}}, {{bestSettings.ContextSize + 32768}}, ... in steps of 16,384, up to 131072
- BatchSize: 64, 128, 256, 512, 1024, 2048
- UBatchSize: 32, 64, 128, 256, 512, 1024
- FlashAttention: true or false
- CacheTypeK: "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"
- CacheTypeV: "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"
- Threads: -1 (auto), or 1–{{hardware.CpuThreads}}
- Mmap: true or false (disabling forces the full model into RAM up front)
- Mlock: true or false
- RepeatPenalty: 1.0 (off) to 1.5, e.g. 1.1
- RepeatLastN: -1 (whole context), 0 (off), or a positive window size, e.g. 256
- DryMultiplier: 0 (off) to 2.0, e.g. 0.8{{moeParam}}{{thinkingParam}}

Respond with ONLY this JSON (no markdown, no extra text before or after):
{"parameter":"<name>","value":<value>,"reasoning":"<one hypothesis sentence: what you expect to happen and why — do not restate numbers from the history above>"}
""";
    }

    private static string BuildFinalPushPrompt(
        OptimizationSession session, OptimizationProfile profile, HardwareInfo hardware)
    {
        var best = session.BestResult!;
        var bestSettings = session.Best!.Settings;
        double baselineScore = session.Iterations[0].Result.CompositeScore;
        bool isMoe = LlamaSettings.IsMoeModel(session.ModelPath);
        bool isThinking = LlamaSettings.IsThinkingModel(session.ModelPath);

        var triedSummary = session.Iterations
            .Where(i => i.AppliedChange is not null)
            .Select(i => i.Result.GenerationRate <= 0
                ? $"  {i.AppliedChange!.Describe()} → FAILED (do not retry)"
                : $"  {i.AppliedChange!.Describe()} → score={i.Result.CompositeScore:F3} ({(i.IsBestSoFar ? "★ best" : i.Result.CompositeScore >= baselineScore ? "ok" : "worse")})")
            .ToList();

        string moeParam = isMoe
            ? $"\n- MoeExpertUsed: integer (e.g. 4, 6, 8, 12) — fewer experts saves VRAM and increases speed"
            : "";

        string thinkingParam = isThinking
            ? "\n- ThinkingEnabled: true or false (currently " + (bestSettings.ThinkingEnabled ?? false) +
              " — enabling reasoning costs speed but can improve answer/tool-call correctness; hasn't necessarily been tried yet)"
            : "";

        string guidance = string.IsNullOrWhiteSpace(session.UserGuidance)
            ? ""
            : $"\nUSER GUIDANCE (follow this even if it means a different tradeoff than the profile default): {session.UserGuidance}\n";

        return $$"""
You are a llama.cpp performance optimizer. After {{triedSummary.Count}} attempts, the last 3 iterations produced no improvement.

HARDWARE:
- GPU: {{(hardware.HasGpu ? $"{hardware.Gpus.FirstOrDefault()?.Name ?? "GPU"} — {hardware.TotalVramMb} MB total, {hardware.FreeVramMb} MB free (measured with the current model loaded — this is real remaining headroom)" : "none (CPU-only)")}}
- RAM: {{hardware.RamTotalMb}} MB total, {{hardware.RamFreeMb}} MB free
- CPU: {{hardware.CpuName}} — {{hardware.CpuCores}} cores / {{hardware.CpuThreads}} threads

PROFILE: {{profile.Name}} — {{profile.Description}}
{{guidance}}
BEST SETTINGS (score={{best.CompositeScore:F3}} vs baseline={{baselineScore:F3}} | TG={{best.GenerationRate:F0}}t/s PP={{best.PromptProcessingRate:F0}}t/s TTFT={{best.TimeToFirstTokenMs:F0}}ms):
- ctx-size: {{bestSettings.ContextSize}}
- n-gpu-layers: {{bestSettings.GpuLayers}} (-1 = all layers)
- batch-size: {{bestSettings.BatchSize}} / ubatch-size: {{bestSettings.UBatchSize}}
- flash-attn: {{bestSettings.FlashAttention}}
- cache-type-k: {{bestSettings.CacheTypeK ?? "f16"}} / cache-type-v: {{bestSettings.CacheTypeV ?? "f16"}}
- threads: {{bestSettings.Threads}} / mmap: {{bestSettings.Mmap}} / mlock: {{bestSettings.Mlock}}
- repeat-penalty: {{bestSettings.RepeatPenalty?.ToString() ?? "server default"}} / repeat-last-n: {{bestSettings.RepeatLastN?.ToString() ?? "server default"}} / dry-multiplier: {{bestSettings.DryMultiplier?.ToString() ?? "server default"}}{{(isThinking ? $"\n- thinking: {bestSettings.ThinkingEnabled ?? false}" : "")}}

ALREADY TRIED (do NOT repeat any of these parameter=value pairs — recreating an already-benchmarked configuration will be rejected):
{{string.Join("\n", triedSummary)}}

TARGET: {{session.TargetTgSpeed:F0}}t/s generation speed. Current TG is {{best.GenerationRate:F0}}t/s.
{{(best.GenerationRate >= session.TargetTgSpeed + LotsOfHeadroomAboveTarget
    ? "We're well above target — trade that spare speed for quality: revert quantized KV cache toward q8_0/f16, restore MoE active-expert count, enable thinking mode (agentic), or grow ContextSize — as long as TG stays at/above the target."
    : best.GenerationRate >= session.TargetTgSpeed * QualityPhaseBufferFactor
        ? "We're above target with buffer — QUALITY TUNING phase: prefer reverting aggressive KV cache quantization, strengthening anti-loop samplers if quality/tool scores dip, enabling thinking mode (agentic), or growing ContextSize (+16,384 steps, up to 131072) over further speed cuts."
        : "We're below target — prioritize whatever raises TG speed most.")}}

We need a fresh approach. Think carefully about:
- Whether a completely different GPU layer count might hit a sweet spot
- Toggling mmap on/off — hasn't necessarily been tried yet and can meaningfully shift TG speed
- Context sizes not yet tried, in +16,384 steps (larger if VRAM permits, up to 131072)
- Combinations we haven't explored (e.g. flash-attn + smaller ubatch) — you may change TWO parameters at once via the optional "parameter2"/"value2" fields
- KV cache quantization levels not yet tried (below target it's a speed/VRAM lever; above target, reverting toward q8_0/f16 protects quality)
- Thread count tuning for CPU-bound operations
- Mlock if the model fits in RAM
- Ubatch vs batch ratios
- If quality, tool, or agent-loop scores are weak (a sign of repetition loops or degraded reasoning): RepeatPenalty, RepeatLastN, or DryMultiplier escalation, or reverting KV quant{{moeParam}}{{thinkingParam}}

If you genuinely believe no further optimization is possible given what has been tried and the hardware constraints, respond with exactly:
DONE

Otherwise respond with ONLY this JSON (no markdown, no extra text; "parameter2"/"value2" are optional — include them only to combine two complementary changes):
{"parameter":"<name>","value":<value>,"parameter2":"<name>","value2":<value>,"reasoning":"<specific reason this new angle is worth trying>"}
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

            var primary = new ParameterChange
            {
                Parameter = suggestion.Parameter,
                OldValue = oldVal,
                NewValue = newVal,
                Reasoning = suggestion.Reasoning,
                Source = "llm",
            };

            // Optional second change: validate against the settings as they'd be after the
            // first, so a repeated parameter or a no-op second value is silently dropped.
            if (string.IsNullOrEmpty(suggestion.Parameter2) || suggestion.Value2 is null)
                return primary;

            var afterFirst = Apply(current, primary);
            var (oldVal2, newVal2) = ApplyToSettings(afterFirst, suggestion.Parameter2, suggestion.Value2.Value);
            if (newVal2 is null) return primary;

            return new ParameterChange
            {
                Parameter = primary.Parameter,
                OldValue = primary.OldValue,
                NewValue = primary.NewValue,
                Reasoning = primary.Reasoning,
                Source = "llm",
                Combined = new ParameterChange
                {
                    Parameter = suggestion.Parameter2,
                    OldValue = oldVal2,
                    NewValue = newVal2,
                    Reasoning = suggestion.Reasoning,
                    Source = "llm",
                },
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

                case "Mmap":
                    bool mm = value.GetBoolean();
                    return (settings.Mmap, mm != settings.Mmap ? mm : (object?)null);

                case "MoeExpertUsed":
                    // 1–256; 0 means "model default" (treated as null = don't pass the arg)
                    int experts = Math.Clamp(value.GetInt32(), 0, 256);
                    int? newExperts = experts == 0 ? null : experts;
                    return (settings.MoeExpertUsed, newExperts != settings.MoeExpertUsed ? newExperts : (object?)null);

                case "ThinkingEnabled":
                    bool thinking = value.GetBoolean();
                    return (settings.ThinkingEnabled, thinking != settings.ThinkingEnabled ? thinking : (object?)null);

                case "RepeatPenalty":
                    // 1.0 = off, up to 1.5 (higher starts actively hurting coherence)
                    float rp = Math.Clamp(value.GetSingle(), 1.0f, 1.5f);
                    return (settings.RepeatPenalty, rp != settings.RepeatPenalty ? rp : (object?)null);

                case "RepeatLastN":
                    // -1 = whole context, 0 = off, otherwise a token window
                    int rln = Math.Clamp(value.GetInt32(), -1, 8192);
                    return (settings.RepeatLastN, rln != settings.RepeatLastN ? rln : (object?)null);

                case "DryMultiplier":
                    // 0 = off, up to 2.0 (higher becomes overly aggressive)
                    float dry = Math.Clamp(value.GetSingle(), 0f, 2.0f);
                    return (settings.DryMultiplier, dry != settings.DryMultiplier ? dry : (object?)null);

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

    // How far above the target counts as "a lot" of headroom — worth actively giving back
    // some speed for quality (e.g. restoring MoE experts), rather than just growing context.
    private const double LotsOfHeadroomAboveTarget = 15.0;
    // Quality-tuning phase starts at target × this factor, so a config sitting exactly on
    // the target doesn't flip between speed and quality phases on benchmark noise.
    private const double QualityPhaseBufferFactor = 1.1;
    private const int ContextStep = 16384;
    // Hard ceiling for context growth regardless of profile default — most models top out
    // well below this, and a failed start here just teaches the optimizer where the wall is.
    private const int ContextCeiling = 131072;

    public ParameterChange? GetHeuristicRecommendation(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware)
    {
        var change = GetHeuristicCandidate(session, profile, hardware);
        if (change is not null && RecreatesTestedConfig(session, change))
        {
            logger.LogInformation(
                "Heuristic suggestion {Param} = {Val} recreates an already-benchmarked configuration — nothing new to try",
                change.Parameter, change.NewValue);
            return null;
        }
        return change;
    }

    private ParameterChange? GetHeuristicCandidate(
        OptimizationSession session,
        OptimizationProfile profile,
        HardwareInfo hardware)
    {
        // Parameters tried so far (by name — for single-shot explorations)
        var tried = session.Iterations
            .Where(i => i.AppliedChange is not null)
            .SelectMany(i => i.AppliedChange!.ParameterNames())
            .ToHashSet();

        var best = session.BestResult!;
        var bestSettings = session.BestSettings!;
        double baselineScore = session.Iterations[0].Result.CompositeScore;
        bool scoreImproved = best.CompositeScore > baselineScore;
        bool isMoe = LlamaSettings.IsMoeModel(session.ModelPath);

        // A near-zero quality score is the signature of a degenerate repetition loop
        // ("is is is is..."). Fix that before chasing any further speed gains — an
        // unusable response is worse than a faster one.
        if (best.QualityScore < OptimizationSession.MinHealthyQuality)
        {
            if (!tried.Contains("RepeatPenalty") && bestSettings.RepeatPenalty is null or <= 1.0f)
                return Change("RepeatPenalty", bestSettings.RepeatPenalty?.ToString() ?? "server default", 1.1f,
                    "Quality score has cratered, likely a repetition loop — raising repeat-penalty to discourage it");

            if (!tried.Contains("DryMultiplier") && bestSettings.DryMultiplier is null or <= 0f)
                return Change("DryMultiplier", bestSettings.DryMultiplier?.ToString() ?? "server default", 0.8f,
                    "Quality score is still cratered — enabling the DRY sampler to break repeat loops");

            if (!tried.Contains("RepeatLastN") && bestSettings.RepeatLastN is null or 0)
                return Change("RepeatLastN", bestSettings.RepeatLastN?.ToString() ?? "server default", 256,
                    "Quality score is still cratered — widening the repeat-penalty lookback window");
        }

        // User-configurable per run (see OptimizationSession.TargetTgSpeed); shared with
        // OptimizationSession.Best, which anchors future suggestions off the most capable
        // config that still clears this bar rather than the highest composite score.
        double targetTgSpeed = session.TargetTgSpeed;
        double tgHeadroom = best.GenerationRate - targetTgSpeed;
        // Quality phase entered with a buffer above the target so benchmark noise doesn't
        // flip the optimizer between speed-chasing and quality-tuning across iterations.
        bool inQualityPhase = best.GenerationRate >= targetTgSpeed * QualityPhaseBufferFactor;
        bool lotsOfSpeedHeadroom = tgHeadroom >= LotsOfHeadroomAboveTarget;

        // ── Speed target met (with buffer): spend the headroom on quality instead of chasing
        // more raw speed. Tooling quality comes first — reverting aggressive KV quant and
        // strengthening anti-loop samplers directly target the repetition loops and degraded
        // tool calls that low-bit quants produce — then capability (experts, context).
        if (inQualityPhase)
        {
            // 1. Undo aggressive KV-cache quantization: q4/q5-class → q8_0; with a lot of
            //    headroom, q8_0 → f16. This is the only path that ever reverses KV quant.
            foreach (var (param, current) in new[]
                     { ("CacheTypeK", bestSettings.CacheTypeK), ("CacheTypeV", bestSettings.CacheTypeV) })
            {
                string? revert = KvQuantRevertTarget(current, lotsOfSpeedHeadroom);
                if (revert is not null && !AlreadyTriedValue(session, param, revert))
                    return Change(param, current ?? "f16", revert,
                        $"TG={best.GenerationRate:F0}t/s clears the {targetTgSpeed:F0}t/s target — " +
                        $"reverting {param} {current} → {revert} to protect output quality");
            }

            // 2. Strengthen anti-loop samplers, but only when there's an actual loop/quality
            //    signal — baseline already ships sane defaults (1.1/256/0.8).
            bool loopSignal =
                (best.Notes?.Contains("Repetition loop", StringComparison.OrdinalIgnoreCase) ?? false)
                || best.QualityScore < 0.9
                || (profile.Type == ProfileType.Agentic && best.AgentLoopScore < 0.9);
            if (loopSignal)
            {
                if ((bestSettings.DryMultiplier ?? 0f) < 1.05f && !AlreadyTriedValue(session, "DryMultiplier", 1.05f))
                    return Change("DryMultiplier", bestSettings.DryMultiplier?.ToString() ?? "server default", 1.05f,
                        "Quality/agent-loop score shows loop symptoms — strengthening the DRY sampler");

                if ((bestSettings.RepeatPenalty ?? 1.0f) < 1.15f && !AlreadyTriedValue(session, "RepeatPenalty", 1.15f))
                    return Change("RepeatPenalty", bestSettings.RepeatPenalty?.ToString() ?? "server default", 1.15f,
                        "Loop symptoms persist — raising repeat-penalty a step");
            }

            // 3. Thinking models on agentic workloads: spend banked speed on reasoning,
            //    which measurably improves tool-call and agent-loop correctness.
            if (profile.Type == ProfileType.Agentic
                && LlamaSettings.IsThinkingModel(session.ModelPath)
                && bestSettings.ThinkingEnabled != true
                && !AlreadyTriedValue(session, "ThinkingEnabled", true))
                return Change("ThinkingEnabled", bestSettings.ThinkingEnabled ?? false, true,
                    $"TG={best.GenerationRate:F0}t/s clears the {targetTgSpeed:F0}t/s target — " +
                    "enabling thinking mode to improve tool-call correctness");

            // 4. A lot of headroom → actively give some back for quality (more MoE experts).
            if (lotsOfSpeedHeadroom && isMoe && bestSettings.MoeExpertUsed.HasValue)
            {
                bool alreadyRestoredDefault = AppliedChanges(session).Any(c =>
                    c.Parameter == "MoeExpertUsed" && c.NewValue is null);
                if (!alreadyRestoredDefault)
                    return Change("MoeExpertUsed", bestSettings.MoeExpertUsed.Value, null!,
                        $"TG={best.GenerationRate:F0}t/s is well above the {targetTgSpeed:F0}t/s target — " +
                        "restoring full expert count for better quality");
            }

            // 5. Grow context — free capability that doesn't cost the speed already banked.
            int? qualityCtx = NextContextStep(session, bestSettings);
            if (qualityCtx is not null)
                return Change("ContextSize", bestSettings.ContextSize, qualityCtx.Value,
                    $"TG={best.GenerationRate:F0}t/s is at/above the {targetTgSpeed:F0}t/s target — " +
                    $"growing context {bestSettings.ContextSize:N0} → {qualityCtx.Value:N0} tokens for more capability");
        }

        // ── Below target (or no further quality moves available above it): keep chasing the
        // best available balance of speed and quality, as before.

        // If VRAM is available and GPU layers aren't already maxed (-1 = all), try more
        if (!tried.Contains("GpuLayers") && hardware.HasGpu && bestSettings.GpuLayers != -1)
        {
            var suggestedNgl = Math.Min(bestSettings.GpuLayers + 8, 200);
            if (suggestedNgl > bestSettings.GpuLayers)
                return Change("GpuLayers", bestSettings.GpuLayers, suggestedNgl,
                    "More GPU layers offloads computation to VRAM, improving TG speed");
        }

        // MoE models: reducing active experts per token frees VRAM and cuts compute.
        // Try this early — it's one of the highest-impact levers for MoE architectures
        // and the heuristic path is the only one guaranteed to explore it if the LLM
        // recommender isn't producing usable suggestions.
        if (isMoe && !tried.Contains("MoeExpertUsed"))
        {
            int suggestedExperts = bestSettings.MoeExpertUsed.HasValue
                ? Math.Max(1, bestSettings.MoeExpertUsed.Value - 2)
                : 4;
            if (suggestedExperts != (bestSettings.MoeExpertUsed ?? -1))
                return Change("MoeExpertUsed", bestSettings.MoeExpertUsed?.ToString() ?? "model default", suggestedExperts,
                    "Fewer active experts per token reduces compute and VRAM with some quality tradeoff");
        }

        // Try flash attention if not enabled
        if (!tried.Contains("FlashAttention") && !bestSettings.FlashAttention)
            return Change("FlashAttention", false, true,
                "Flash attention reduces VRAM usage and often improves throughput");

        // Toggle mmap — disabling forces the full model into RAM up front (can help steady-state
        // TG if RAM allows it); re-enabling lets the OS page cache manage it (helps if RAM is tight).
        if (!tried.Contains("Mmap"))
            return Change("Mmap", bestSettings.Mmap, !bestSettings.Mmap,
                bestSettings.Mmap
                    ? "Disabling mmap forces the full model into RAM up front, which can improve steady-state TG speed"
                    : "Re-enabling mmap lets the OS page cache manage the model file, which can help if RAM is tight");

        // Try KV cache quantization to free VRAM for more context or layers
        if (!tried.Contains("CacheTypeK") && bestSettings.CacheTypeK is null or "f16")
            return Change("CacheTypeK", bestSettings.CacheTypeK ?? "f16", "q8_0",
                "q8_0 KV cache halves KV cache VRAM with minimal quality loss");

        if (!tried.Contains("CacheTypeV") && bestSettings.CacheTypeV is null or "f16")
            return Change("CacheTypeV", bestSettings.CacheTypeV ?? "f16", "q8_0",
                "q8_0 KV cache halves KV cache VRAM with minimal quality loss");

        // MoE: escalate to fewer experts still if the first reduction helped
        if (isMoe && tried.Contains("MoeExpertUsed") && bestSettings.MoeExpertUsed is > 2)
        {
            int nextExperts = Math.Max(1, bestSettings.MoeExpertUsed.Value - 2);
            bool alreadyTried = AppliedChanges(session).Any(c =>
                c.Parameter == "MoeExpertUsed" &&
                c.NewValue is not null &&
                Convert.ToInt32(c.NewValue) == nextExperts);
            if (!alreadyTried)
                return Change("MoeExpertUsed", bestSettings.MoeExpertUsed.Value, nextExperts,
                    "Reducing active experts further — the first reduction improved the score");
        }

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

        // Progressive context expansion — both profiles. +16,384 tokens per step, only while
        // score has been improving, and only if this specific target value hasn't been tried.
        if (scoreImproved)
        {
            int? nextCtx = NextContextStep(session, bestSettings);
            if (nextCtx is not null)
                return Change("ContextSize", bestSettings.ContextSize, nextCtx.Value,
                    $"Score is improving — expanding context {bestSettings.ContextSize:N0} → {nextCtx.Value:N0} tokens");
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
        {
            int? nextCtx = NextContextStep(session, bestSettings);
            if (nextCtx is not null)
                return Change("ContextSize", bestSettings.ContextSize, nextCtx.Value,
                    $"Trying larger context {bestSettings.ContextSize:N0} → {nextCtx.Value:N0} tokens as a final exploration");
        }

        return null; // Nothing left to try
    }

    // Next context-size step (+16,384 tokens, capped at ContextCeiling), or null if already
    // at the ceiling or this exact value has already been tried.
    private static int? NextContextStep(OptimizationSession session, LlamaSettings bestSettings)
    {
        if (bestSettings.ContextSize >= ContextCeiling) return null;

        int nextCtx = Math.Min(bestSettings.ContextSize + ContextStep, ContextCeiling);
        if (nextCtx <= bestSettings.ContextSize) return null;

        bool alreadyTried = AppliedChanges(session).Any(c =>
            c.Parameter == "ContextSize" &&
            c.NewValue is not null &&
            Convert.ToInt32(c.NewValue) == nextCtx);

        return alreadyTried ? null : nextCtx;
    }

    // Quality-phase KV cache revert target: low-bit quants step up to q8_0; q8_0 itself only
    // goes back to f16 when there's a lot of speed headroom to spend. null = nothing to revert.
    private static string? KvQuantRevertTarget(string? current, bool lotsOfSpeedHeadroom) => current switch
    {
        null or "f16" or "bf16" => null,
        "q8_0" or "turbo8" => lotsOfSpeedHeadroom ? "f16" : null,
        _ => "q8_0",
    };

    // True if this exact parameter=value pair has been applied at some point this session
    // (compared as strings — values arrive as int/float/bool/string depending on the source).
    private static bool AlreadyTriedValue(OptimizationSession session, string parameter, object? value) =>
        AppliedChanges(session).Any(c =>
            c.Parameter == parameter &&
            string.Equals(c.NewValue?.ToString(), value?.ToString(), StringComparison.OrdinalIgnoreCase));

    // Every individual parameter change applied this session, with combined changes flattened.
    private static IEnumerable<ParameterChange> AppliedChanges(OptimizationSession session) =>
        session.Iterations
            .Where(i => i.AppliedChange is not null)
            .SelectMany(i => i.AppliedChange!.Chain());

    private static ParameterChange Change(string param, object oldVal, object newVal, string reason)
        => new() { Parameter = param, OldValue = oldVal, NewValue = newVal, Reasoning = reason, Source = "heuristic" };

    /// <summary>Applies a ParameterChange (and any combined changes) to a cloned copy of the settings.</summary>
    public static LlamaSettings Apply(LlamaSettings settings, ParameterChange change)
    {
        var s = settings.Clone();
        foreach (var c in change.Chain())
            ApplyInPlace(s, c);
        return s;
    }

    private static void ApplyInPlace(LlamaSettings s, ParameterChange change)
    {
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
            case "Mmap":            s.Mmap = Convert.ToBoolean(change.NewValue); break;
            case "MoeExpertUsed":   s.MoeExpertUsed = change.NewValue is null ? null : Convert.ToInt32(change.NewValue); break;
            case "ThinkingEnabled": s.ThinkingEnabled = change.NewValue is null ? null : Convert.ToBoolean(change.NewValue); break;
            case "RepeatPenalty":   s.RepeatPenalty = change.NewValue is null ? null : Convert.ToSingle(change.NewValue); break;
            case "RepeatLastN":     s.RepeatLastN = change.NewValue is null ? null : Convert.ToInt32(change.NewValue); break;
            case "DryMultiplier":   s.DryMultiplier = change.NewValue is null ? null : Convert.ToSingle(change.NewValue); break;
        }
    }
}
