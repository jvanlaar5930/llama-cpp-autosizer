namespace LlamaCppAutosizer.Models;

public class GpuInfo
{
    public string Name { get; init; } = "Unknown";
    public long VramTotalMb { get; init; }
    public long VramFreeMb { get; init; }
    public string Vendor { get; init; } = "Unknown";   // NVIDIA, AMD, Intel
    public int Index { get; init; }
}

public class HardwareInfo
{
    public List<GpuInfo> Gpus { get; init; } = [];
    public long RamTotalMb { get; init; }
    public long RamFreeMb { get; init; }
    public int CpuCores { get; init; }
    public int CpuThreads { get; init; }
    public string CpuName { get; init; } = "Unknown";

    public long TotalVramMb => Gpus.Sum(g => g.VramTotalMb);
    public long FreeVramMb => Gpus.Sum(g => g.VramFreeMb);
    public bool HasGpu => Gpus.Count > 0;

    /// <summary>
    /// Rough estimate of how many 1GB layers we can fit in VRAM,
    /// given a model's estimated bytes-per-layer.
    /// </summary>
    public int EstimateMaxGpuLayers(long modelFileSizeMb, int totalModelLayers)
    {
        if (!HasGpu || TotalVramMb == 0) return 0;
        // Reserve ~1GB for KV cache and overhead
        long usableVram = Math.Max(0, FreeVramMb - 1024);
        double mbPerLayer = (double)modelFileSizeMb / totalModelLayers;
        return (int)Math.Min(totalModelLayers, usableVram / mbPerLayer);
    }

    /// <summary>Suggest a safe starting context size given available RAM + VRAM.</summary>
    public int SuggestContextSize(long modelFileSizeMb, bool gpuOffload)
    {
        // KV cache grows roughly 0.5 MB per 1k context at bf16 for a 7B model
        // This is a rough heuristic; actual size depends on num heads and head dim
        long availableMb = gpuOffload ? FreeVramMb : RamFreeMb;
        availableMb -= modelFileSizeMb; // subtract model weight footprint
        availableMb = Math.Max(0, availableMb);

        // Rough: ~0.25 MB per 1k tokens (aggressive; halve for safety)
        long ctx = (availableMb / 256) * 1024;  // in tokens
        ctx = Math.Min(ctx, 131072);            // cap at 128k
        ctx = Math.Max(ctx, 2048);              // floor at 2k

        // Round to nearest power of 2 bucket
        int[] buckets = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
        return buckets.LastOrDefault(b => b <= ctx, 2048);
    }
}
