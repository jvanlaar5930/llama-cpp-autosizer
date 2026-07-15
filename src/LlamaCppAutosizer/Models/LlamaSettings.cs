namespace LlamaCppAutosizer.Models;

/// <summary>
/// All llama-server CLI parameters that the optimizer can tune.
/// </summary>
public class LlamaSettings
{
    // Core sizing
    public int ContextSize { get; set; } = 4096;
    public int GpuLayers { get; set; } = -1;       // -1 = offload all
    public int BatchSize { get; set; } = 512;
    public int UBatchSize { get; set; } = 512;

    // Threading
    public int Threads { get; set; } = -1;          // -1 = auto
    public int ThreadsBatch { get; set; } = -1;

    // Memory / attention
    public bool FlashAttention { get; set; } = false;
    public bool Mmap { get; set; } = true;
    public bool Mlock { get; set; } = false;

    // KV cache quantization (null = f16, options: q8_0, q4_0, q5_0, q4_1, q5_1, iq4_nl)
    public string? CacheTypeK { get; set; }
    public string? CacheTypeV { get; set; }

    // MoE (Mixture of Experts) — only meaningful for MoE architectures
    // Reduces active experts per token; null = use the model's built-in default
    public int? MoeExpertUsed { get; set; }

    // Thinking / chain-of-thought mode — only meaningful for thinking-capable models
    // (Qwen3, DeepSeek-R1, QwQ). null = model default (usually enabled).
    // false = disable thinking (shorter, faster responses); true = force enable.
    public bool? ThinkingEnabled { get; set; }

    // Concurrency / defrag
    public int ParallelSlots { get; set; } = 1;
    public float DefragThreshold { get; set; } = -1f;   // -1 = disabled

    // RoPE overrides (null = use model defaults)
    public string? RopeScaling { get; set; }
    public float? RopeFreqBase { get; set; }
    public float? RopeFreqScale { get; set; }

    // Anti-repetition sampling defaults (null = use llama-server's built-in default).
    // These only take effect for clients that don't override sampling params per-request.
    public float? RepeatPenalty { get; set; }
    public int? RepeatLastN { get; set; }
    // DRY sampler strength — llama.cpp's dedicated repeat-loop breaker ("is is is is...").
    // 0 disables it; null leaves the server default (usually already enabled).
    public float? DryMultiplier { get; set; }

    // Path to a custom jinja chat template file (null = use the model's built-in template)
    public string? ChatTemplateFile { get; set; }

    // Extra raw CLI args the user wants to inject
    public string? ExtraArgs { get; set; }

    // Display name for this configuration (set by optimizer)
    public string? Label { get; set; }

    public string[] ToServerArgs(string modelPath, int port = 8080)
    {
        var args = new List<string>
        {
            "--model", modelPath,
            "--ctx-size", ContextSize.ToString(),
            "--n-gpu-layers", GpuLayers.ToString(),
            "--batch-size", BatchSize.ToString(),
            "--ubatch-size", UBatchSize.ToString(),
            "--parallel", ParallelSlots.ToString(),
            "--port", port.ToString(),
            "--metrics",
        };

        if (Threads >= 0) { args.Add("--threads"); args.Add(Threads.ToString()); }
        if (ThreadsBatch >= 0) { args.Add("--threads-batch"); args.Add(ThreadsBatch.ToString()); }
        // Pass an explicit value so llama-server doesn't misinterpret the next arg
        // as the value for --flash-attn (it now accepts [on|off|auto]).
        if (FlashAttention) { args.Add("--flash-attn"); args.Add("on"); }
        if (!Mmap) args.Add("--no-mmap");
        if (Mlock) args.Add("--mlock");
        if (CacheTypeK is not null) { args.Add("--cache-type-k"); args.Add(CacheTypeK); }
        if (CacheTypeV is not null) { args.Add("--cache-type-v"); args.Add(CacheTypeV); }
        if (MoeExpertUsed.HasValue)
        {
            // The GGUF key is arch-prefixed (e.g. qwen3moe.expert_used_count) — an unprefixed
            // or wrong key is silently ignored by llama-server, making the override a no-op.
            string arch = Services.GgufMetadata.GetArchitecture(modelPath) ?? "llama";
            args.Add("--override-kv");
            args.Add($"{arch}.expert_used_count=int:{MoeExpertUsed.Value}");
        }
        if (ThinkingEnabled.HasValue)
        {
            // --override-kv can't toggle thinking (enable_thinking is a chat-template variable,
            // not GGUF metadata). Pass it as a template kwarg; when disabling, also zero the
            // reasoning budget — some models keep emitting thinking blocks on kwargs alone.
            args.Add("--chat-template-kwargs");
            args.Add($"{{\"enable_thinking\":{ThinkingEnabled.Value.ToString().ToLower()}}}");
            if (!ThinkingEnabled.Value) { args.Add("--reasoning-budget"); args.Add("0"); }
        }
        if (DefragThreshold >= 0) { args.Add("--defrag-thold"); args.Add(DefragThreshold.ToString("F2")); }
        if (RopeScaling is not null) { args.Add("--rope-scaling"); args.Add(RopeScaling); }
        if (RopeFreqBase.HasValue) { args.Add("--rope-freq-base"); args.Add(RopeFreqBase.Value.ToString()); }
        if (RopeFreqScale.HasValue) { args.Add("--rope-freq-scale"); args.Add(RopeFreqScale.Value.ToString()); }
        if (RepeatPenalty.HasValue) { args.Add("--repeat-penalty"); args.Add(RepeatPenalty.Value.ToString()); }
        if (RepeatLastN.HasValue) { args.Add("--repeat-last-n"); args.Add(RepeatLastN.Value.ToString()); }
        if (DryMultiplier.HasValue) { args.Add("--dry-multiplier"); args.Add(DryMultiplier.Value.ToString()); }
        if (!string.IsNullOrWhiteSpace(ChatTemplateFile)) { args.Add("--chat-template-file"); args.Add(ChatTemplateFile); }

        if (!string.IsNullOrWhiteSpace(ExtraArgs))
        {
            // Simple split on whitespace; user should quote paths manually
            args.AddRange(ExtraArgs.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }

        return [.. args];
    }

    public LlamaSettings Clone()
    {
        var c = (LlamaSettings)MemberwiseClone();
        return c;
    }

    /// <summary>
    /// Canonical identity of every setting that affects server behavior (Label excluded).
    /// Two settings with equal fingerprints launch an identical server, so benchmarking
    /// both is pure waste — the optimizer uses this to refuse duplicate configurations.
    /// </summary>
    public string Fingerprint() => string.Join("|",
        ContextSize, GpuLayers, BatchSize, UBatchSize,
        Threads, ThreadsBatch, FlashAttention, Mmap, Mlock,
        CacheTypeK ?? "f16", CacheTypeV ?? "f16",
        MoeExpertUsed?.ToString() ?? "default",
        ThinkingEnabled?.ToString() ?? "default",
        ParallelSlots, DefragThreshold,
        RopeScaling ?? "", RopeFreqBase?.ToString() ?? "", RopeFreqScale?.ToString() ?? "",
        RepeatPenalty?.ToString() ?? "", RepeatLastN?.ToString() ?? "", DryMultiplier?.ToString() ?? "",
        ChatTemplateFile ?? "", ExtraArgs ?? "");

    /// <summary>Human-readable summary of key parameters.</summary>
    public string Summary()
    {
        var s = $"ctx={ContextSize} ngl={GpuLayers} batch={BatchSize}/{UBatchSize} " +
                $"fa={FlashAttention} kv={CacheTypeK ?? "f16"}/{CacheTypeV ?? "f16"} " +
                $"threads={Threads}/{ThreadsBatch}";
        if (MoeExpertUsed.HasValue) s += $" experts={MoeExpertUsed.Value}";
        return s;
    }

    /// <summary>Detects models that support chain-of-thought "thinking" mode.</summary>
    public static bool IsThinkingModel(string? modelPath)
    {
        if (string.IsNullOrEmpty(modelPath)) return false;
        var name = Path.GetFileName(modelPath).ToLowerInvariant();
        // Qwen3.x family (NOT Qwen2.5 — it doesn't have thinking mode)
        if (System.Text.RegularExpressions.Regex.IsMatch(name, @"qwen3[\.\-_]")) return true;
        // QwQ (thinking model from Qwen team)
        if (name.StartsWith("qwq") || name.Contains("-qwq")) return true;
        // DeepSeek-R1 and its distillations
        if (name.Contains("deepseek-r1") || name.Contains("deepseek_r1")) return true;
        return false;
    }

    /// <summary>Detects likely MoE architectures from the model filename.</summary>
    public static bool IsMoeModel(string? modelPath)
    {
        if (string.IsNullOrEmpty(modelPath)) return false;
        var name = Path.GetFileName(modelPath).ToLowerInvariant();

        // Explicit MoE names
        if (name.Contains("mixtral") ||
            name.Contains("moe") ||
            name.Contains("dbrx") ||
            name.Contains("grok") ||
            name.Contains("deepseek") ||   // DeepSeek-V2/V3/R1 are all MoE
            name.Contains("arctic") ||
            name.Contains("jamba") ||
            name.Contains("olmoe")) return true;

        // Qwen "Active-NB" notation: e.g. Qwen3.6-35B-A3B, Qwen2-57B-A14B
        // Pattern: <total>B-A<active>B — the A<n>B suffix only appears on MoE variants
        if (System.Text.RegularExpressions.Regex.IsMatch(name, @"\d+b-a\d+b"))
            return true;

        return false;
    }

    // Standard llama.cpp types + turbo* types from the TurboQuant fork
    public static readonly string[] ValidCacheTypes =
        ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];

    // Cache types only available in the llama-cpp-turboquant fork's llama-server
    public static readonly string[] TurboQuantCacheTypes =
        ["turbo4", "turbo3", "turbo2", "turbo1", "turbo8"];
}
