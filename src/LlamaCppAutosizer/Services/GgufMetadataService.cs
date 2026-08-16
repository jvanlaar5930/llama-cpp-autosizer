using System.Text;

namespace LlamaCppAutosizer.Services;

/// <summary>
/// Architecture facts read straight out of a GGUF file's header, so model traits don't have
/// to be guessed from the filename. Null fields mean the key wasn't present in the header.
/// </summary>
public record GgufMetadata
{
    public string? Architecture { get; init; }
    public int? ExpertCount { get; init; }
    public int? ExpertUsedCount { get; init; }
    public int? BlockCount { get; init; }
    public int? ContextLength { get; init; }

    // Attention shape — needed to size the KV cache exactly rather than guessing.
    public int? EmbeddingLength { get; init; }
    public int? HeadCount { get; init; }
    public int? KeyLength { get; init; }
    public int? ValueLength { get; init; }
    public int? ExpertFeedForwardLength { get; init; }

    /// <summary>KV heads per layer. Some architectures vary this per layer (Gemma uses fewer
    /// KV heads on its global-attention layers), so it is stored expanded to one entry per block.</summary>
    public int[]? HeadCountKvPerLayer { get; init; }

    // Sliding-window attention. Layers marked SWA only ever hold `SlidingWindow` tokens of KV,
    // which is what makes very long contexts affordable on models like Gemma.
    public int? SlidingWindow { get; init; }
    public int? KeyLengthSwa { get; init; }
    public int? ValueLengthSwa { get; init; }
    /// <summary>One entry per block; true = sliding-window layer, false = full/global attention.</summary>
    public bool[]? SlidingWindowPattern { get; init; }

    public bool IsMoe => ExpertCount is > 1;
    public bool HasSlidingWindowAttention => SlidingWindow is > 0 && SlidingWindowPattern is { Length: > 0 };

    public int KvHeadsAt(int layer) =>
        HeadCountKvPerLayer is { Length: > 0 } h ? h[Math.Clamp(layer, 0, h.Length - 1)] : HeadCount ?? 0;

    public bool IsSlidingWindowLayer(int layer) =>
        SlidingWindowPattern is { Length: > 0 } p && p[Math.Clamp(layer, 0, p.Length - 1)];

    /// <summary>
    /// Bytes the KV cache will occupy at the given context length, summed layer by layer so that
    /// per-layer KV-head counts and sliding-window layers are accounted for. This is the number
    /// that decides how much context actually fits, and for SWA models it is dramatically smaller
    /// than the usual "2 × layers × heads × dim × ctx" estimate.
    /// </summary>
    public long KvCacheBytes(int contextTokens, string? cacheTypeK, string? cacheTypeV, int batchSize = 512)
    {
        if (BlockCount is not > 0 || contextTokens <= 0) return 0;

        double kBytes = BytesPerElement(cacheTypeK);
        double vBytes = BytesPerElement(cacheTypeV);
        int globalK = KeyLength ?? 128;
        int globalV = ValueLength ?? globalK;
        int swaK = KeyLengthSwa ?? globalK;
        int swaV = ValueLengthSwa ?? globalV;

        double total = 0;
        for (int layer = 0; layer < BlockCount.Value; layer++)
        {
            bool swa = IsSlidingWindowLayer(layer);
            int heads = KvHeadsAt(layer);
            if (heads <= 0) continue;

            // An SWA layer's cache is sized to its window, not the context. The extra batch
            // slack mirrors llama.cpp, which keeps room for the in-flight batch on top.
            long tokens = swa && SlidingWindow is > 0
                ? Math.Min(contextTokens, SlidingWindow.Value + batchSize)
                : contextTokens;

            total += heads * ((swa ? swaK : globalK) * kBytes + (swa ? swaV : globalV) * vBytes) * tokens;
        }
        return (long)total;
    }

    /// <summary>
    /// Approximate share of one block's weights taken by MoE expert FFN tensors. Used to decide
    /// how many layers' experts can stay in VRAM. Returns 0 for dense models.
    /// </summary>
    public double ExpertShareOfLayer()
    {
        if (!IsMoe || ExpertCount is not > 0 || ExpertFeedForwardLength is not > 0
            || EmbeddingLength is not > 0 || HeadCount is not > 0)
            return 0;

        int embed = EmbeddingLength.Value;
        // gate + up + down per expert
        double expertParams = (double)ExpertCount.Value * 3 * ExpertFeedForwardLength.Value * embed;

        int kLen = KeyLength ?? 128;
        int vLen = ValueLength ?? kLen;
        int kvHeads = KvHeadsAt(0);
        double attnParams = (double)embed * HeadCount.Value * kLen   // q
                          + (double)embed * kvHeads * kLen           // k
                          + (double)embed * kvHeads * vLen           // v
                          + (double)HeadCount.Value * vLen * embed;  // o

        double denom = expertParams + attnParams;
        return denom > 0 ? expertParams / denom : 0;
    }

    /// <summary>Bytes per element for a llama.cpp KV cache type (block-quantized sizes included).</summary>
    public static double BytesPerElement(string? cacheType) => cacheType switch
    {
        null or "" or "f16" or "bf16" => 2.0,
        "q8_0"   => 34.0 / 32,
        "q5_1"   => 24.0 / 32,
        "q5_0"   => 22.0 / 32,
        "q4_1"   => 20.0 / 32,
        "q4_0" or "iq4_nl" => 18.0 / 32,
        // TurboQuant fork types and anything unrecognised: assume f16 so the estimate errs
        // toward reserving too much VRAM rather than too little.
        _ => 2.0,
    };
}

/// <summary>
/// Minimal reader for the GGUF header's key/value metadata block (spec v2/v3).
/// Only the handful of architecture keys the optimizer cares about are kept; tensor data is
/// never touched, so this reads a few KB off the front of a multi-GB file.
/// </summary>
public static class GgufMetadataService
{
    // Header reads hit the disk for every settings screen render, so results are memoized.
    // Keyed by path + last-write time so requantizing a model in place doesn't serve stale data.
    private static readonly Dictionary<string, GgufMetadata?> Cache = [];
    private static readonly Lock CacheLock = new();

    public static GgufMetadata? Read(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath)) return null;

        string key;
        try { key = $"{modelPath}|{File.GetLastWriteTimeUtc(modelPath).Ticks}"; }
        catch { return null; }

        lock (CacheLock)
        {
            if (Cache.TryGetValue(key, out var cached)) return cached;
        }

        var parsed = TryParse(modelPath);

        lock (CacheLock)
        {
            Cache[key] = parsed;
        }
        return parsed;
    }

    private static GgufMetadata? TryParse(string path)
    {
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

            if (new string(br.ReadChars(4)) != "GGUF") return null;

            uint version = br.ReadUInt32();
            if (version is < 2 or > 3) return null;   // only v2/v3 share this header layout

            br.ReadUInt64();                          // tensor count — not needed
            ulong kvCount = br.ReadUInt64();
            if (kvCount > 100_000) return null;       // implausible; treat as a malformed header

            string? arch = null;
            var values = new Dictionary<string, long>();
            var intArrays = new Dictionary<string, long[]>();
            var boolArrays = new Dictionary<string, bool[]>();

            for (ulong i = 0; i < kvCount; i++)
            {
                string k = ReadString(br);
                uint type = br.ReadUInt32();

                if (k == "general.architecture" && type == 8)
                {
                    arch = ReadString(br);
                    continue;
                }

                // Architecture-scoped keys are prefixed with the arch name (e.g. "qwen35moe.expert_count"),
                // so match on the suffix rather than reconstructing the full key.
                if (IsWantedKey(k))
                {
                    // A few keys are per-layer arrays on some architectures and plain scalars on
                    // others (Gemma varies KV heads per layer; most models don't), so both shapes
                    // have to be accepted for the same key.
                    if (type == 9)
                    {
                        if (TryReadArray(br, out long[]? ints, out bool[]? bools))
                        {
                            if (ints is not null) intArrays[k] = ints;
                            if (bools is not null) boolArrays[k] = bools;
                        }
                        continue;
                    }
                    if (type == 7) { boolArrays[k] = [br.ReadByte() != 0]; continue; }
                    if (TryReadInt(br, type, out long v)) { values[k] = v; continue; }
                    continue;
                }

                SkipValue(br, type);
            }

            int? blockCount = FindInt(values, ".block_count");

            return new GgufMetadata
            {
                Architecture = arch,
                ExpertCount = FindInt(values, ".expert_count"),
                ExpertUsedCount = FindInt(values, ".expert_used_count"),
                BlockCount = blockCount,
                ContextLength = FindInt(values, ".context_length"),
                EmbeddingLength = FindInt(values, ".embedding_length"),
                HeadCount = FindInt(values, ".attention.head_count"),
                KeyLength = FindInt(values, ".attention.key_length"),
                ValueLength = FindInt(values, ".attention.value_length"),
                KeyLengthSwa = FindInt(values, ".attention.key_length_swa"),
                ValueLengthSwa = FindInt(values, ".attention.value_length_swa"),
                SlidingWindow = FindInt(values, ".attention.sliding_window"),
                ExpertFeedForwardLength = FindInt(values, ".expert_feed_forward_length"),
                HeadCountKvPerLayer = ExpandHeadCountKv(values, intArrays, blockCount),
                SlidingWindowPattern = ExpandSwaPattern(values, boolArrays, blockCount),
            };
        }
        catch
        {
            // Unreadable/truncated/locked file — callers fall back to filename heuristics.
            return null;
        }
    }

    private static readonly string[] WantedSuffixes =
    [
        ".expert_count", ".expert_used_count", ".block_count", ".context_length",
        ".embedding_length", ".expert_feed_forward_length",
        ".attention.head_count", ".attention.head_count_kv",
        ".attention.key_length", ".attention.value_length",
        ".attention.key_length_swa", ".attention.value_length_swa",
        ".attention.sliding_window", ".attention.sliding_window_pattern",
    ];

    private static bool IsWantedKey(string key)
    {
        foreach (var s in WantedSuffixes)
            if (key.EndsWith(s, StringComparison.Ordinal)) return true;
        return false;
    }

    // Reads an array value as either ints or bools, whichever the element type says.
    // Anything else (strings, floats) is consumed and reported as neither.
    private static bool TryReadArray(BinaryReader br, out long[]? ints, out bool[]? bools)
    {
        ints = null; bools = null;
        uint elemType = br.ReadUInt32();
        ulong n = br.ReadUInt64();
        if (n > 100_000) throw new InvalidDataException("implausible GGUF array length");

        if (elemType == 7)
        {
            var b = new bool[n];
            for (ulong i = 0; i < n; i++) b[i] = br.ReadByte() != 0;
            bools = b;
            return true;
        }

        if (elemType is 0 or 1 or 2 or 3 or 4 or 5 or 10 or 11)
        {
            var a = new long[n];
            for (ulong i = 0; i < n; i++) { TryReadInt(br, elemType, out long v); a[i] = v; }
            ints = a;
            return true;
        }

        for (ulong i = 0; i < n; i++) SkipValue(br, elemType);
        return false;
    }

    // KV-head count is a scalar on most architectures and a per-layer array on a few.
    // Normalise to one entry per block so callers never have to care which.
    private static int[]? ExpandHeadCountKv(
        Dictionary<string, long> scalars, Dictionary<string, long[]> arrays, int? blockCount)
    {
        foreach (var (k, v) in arrays)
            if (k.EndsWith(".attention.head_count_kv", StringComparison.Ordinal) && v.Length > 0)
                return [.. v.Select(x => (int)x)];

        int? scalar = FindInt(scalars, ".attention.head_count_kv");
        if (scalar is null || blockCount is not > 0) return null;
        return [.. Enumerable.Repeat(scalar.Value, blockCount.Value)];
    }

    // The SWA pattern appears either as a per-layer bool array (true = sliding-window layer) or
    // as an integer stride N meaning "every Nth layer is global attention, the rest are SWA".
    private static bool[]? ExpandSwaPattern(
        Dictionary<string, long> scalars, Dictionary<string, bool[]> boolArrays, int? blockCount)
    {
        foreach (var (k, v) in boolArrays)
            if (k.EndsWith(".attention.sliding_window_pattern", StringComparison.Ordinal) && v.Length > 0)
                return v;

        int? stride = FindInt(scalars, ".attention.sliding_window_pattern");
        if (stride is not > 1 || blockCount is not > 0) return null;
        return [.. Enumerable.Range(0, blockCount.Value).Select(i => (i + 1) % stride.Value != 0)];
    }

    private static int? FindInt(Dictionary<string, long> values, string suffix)
    {
        foreach (var (k, v) in values)
            if (k.EndsWith(suffix, StringComparison.Ordinal) && v is > 0 and <= int.MaxValue)
                return (int)v;
        return null;
    }

    private static string ReadString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        if (len > 1_000_000) throw new InvalidDataException("implausible GGUF string length");
        return Encoding.UTF8.GetString(br.ReadBytes((int)len));
    }

    private static bool TryReadInt(BinaryReader br, uint type, out long value)
    {
        switch (type)
        {
            case 0: value = br.ReadByte(); return true;
            case 1: value = br.ReadSByte(); return true;
            case 2: value = br.ReadUInt16(); return true;
            case 3: value = br.ReadInt16(); return true;
            case 4: value = br.ReadUInt32(); return true;
            case 5: value = br.ReadInt32(); return true;
            case 10: value = (long)br.ReadUInt64(); return true;
            case 11: value = br.ReadInt64(); return true;
            default:
                SkipValue(br, type);
                value = 0;
                return false;
        }
    }

    private static void SkipValue(BinaryReader br, uint type)
    {
        switch (type)
        {
            case 0: case 1: case 7: br.BaseStream.Seek(1, SeekOrigin.Current); break;
            case 2: case 3: br.BaseStream.Seek(2, SeekOrigin.Current); break;
            case 4: case 5: case 6: br.BaseStream.Seek(4, SeekOrigin.Current); break;
            case 10: case 11: case 12: br.BaseStream.Seek(8, SeekOrigin.Current); break;
            case 8: ReadString(br); break;
            case 9:
            {
                uint elemType = br.ReadUInt32();
                ulong n = br.ReadUInt64();
                // Strings are variable-width, so the array has to be walked element by element;
                // fixed-width types can be skipped in one seek.
                if (elemType == 8)
                {
                    for (ulong i = 0; i < n; i++) ReadString(br);
                }
                else
                {
                    int width = FixedWidth(elemType);
                    br.BaseStream.Seek(width * (long)n, SeekOrigin.Current);
                }
                break;
            }
            default: throw new InvalidDataException($"unknown GGUF value type {type}");
        }
    }

    private static int FixedWidth(uint type) => type switch
    {
        0 or 1 or 7 => 1,
        2 or 3 => 2,
        4 or 5 or 6 => 4,
        10 or 11 or 12 => 8,
        _ => throw new InvalidDataException($"unknown GGUF array element type {type}"),
    };
}
