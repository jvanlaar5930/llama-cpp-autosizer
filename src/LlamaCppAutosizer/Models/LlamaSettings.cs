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

    // Concurrency / defrag
    public int ParallelSlots { get; set; } = 1;
    public float DefragThreshold { get; set; } = -1f;   // -1 = disabled

    // RoPE overrides (null = use model defaults)
    public string? RopeScaling { get; set; }
    public float? RopeFreqBase { get; set; }
    public float? RopeFreqScale { get; set; }

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
            "--log-disable",
        };

        if (Threads >= 0) { args.Add("--threads"); args.Add(Threads.ToString()); }
        if (ThreadsBatch >= 0) { args.Add("--threads-batch"); args.Add(ThreadsBatch.ToString()); }
        if (FlashAttention) args.Add("--flash-attn");
        if (!Mmap) args.Add("--no-mmap");
        if (Mlock) args.Add("--mlock");
        if (CacheTypeK is not null) { args.Add("--cache-type-k"); args.Add(CacheTypeK); }
        if (CacheTypeV is not null) { args.Add("--cache-type-v"); args.Add(CacheTypeV); }
        if (DefragThreshold >= 0) { args.Add("--defrag-thold"); args.Add(DefragThreshold.ToString("F2")); }
        if (RopeScaling is not null) { args.Add("--rope-scaling"); args.Add(RopeScaling); }
        if (RopeFreqBase.HasValue) { args.Add("--rope-freq-base"); args.Add(RopeFreqBase.Value.ToString()); }
        if (RopeFreqScale.HasValue) { args.Add("--rope-freq-scale"); args.Add(RopeFreqScale.Value.ToString()); }

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

    /// <summary>Human-readable summary of key parameters.</summary>
    public string Summary()
        => $"ctx={ContextSize} ngl={GpuLayers} batch={BatchSize}/{UBatchSize} " +
           $"fa={FlashAttention} kv={CacheTypeK ?? "f16"}/{CacheTypeV ?? "f16"} " +
           $"threads={Threads}/{ThreadsBatch}";

    // Standard llama.cpp types + turbo* types from the TurboQuant fork
    public static readonly string[] ValidCacheTypes =
        ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"];

    // Cache types only available in the llama-cpp-turboquant fork's llama-server
    public static readonly string[] TurboQuantCacheTypes =
        ["turbo4", "turbo3", "turbo2", "turbo1", "turbo8"];
}
