using System.Collections.Concurrent;
using System.Text;

namespace LlamaCppAutosizer.Services;

/// <summary>
/// Minimal GGUF header reader — extracts <c>general.architecture</c> so CLI metadata
/// overrides can use the model's arch-specific key prefix (e.g.
/// <c>qwen3moe.expert_used_count</c>). Reads only the metadata KV section, never tensor data.
/// </summary>
public static class GgufMetadata
{
    private static readonly ConcurrentDictionary<string, string?> ArchCache = new(StringComparer.OrdinalIgnoreCase);

    // GGUF value type ids (spec: ggml-org/ggml docs/gguf.md)
    private const uint TypeUint8 = 0, TypeInt8 = 1, TypeUint16 = 2, TypeInt16 = 3,
        TypeUint32 = 4, TypeInt32 = 5, TypeFloat32 = 6, TypeBool = 7,
        TypeString = 8, TypeArray = 9, TypeUint64 = 10, TypeInt64 = 11, TypeFloat64 = 12;

    // Sanity cap: no metadata string/array in a well-formed model header comes close to this.
    private const long MaxStringBytes = 16 * 1024 * 1024;

    /// <summary>
    /// The model's <c>general.architecture</c> value (e.g. "llama", "qwen3moe"), or null if
    /// the file can't be read or isn't a parseable GGUF. Cached per path.
    /// </summary>
    public static string? GetArchitecture(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath)) return null;
        return ArchCache.GetOrAdd(modelPath, static path =>
        {
            try { return ReadArchitecture(path); }
            catch { return null; }
        });
    }

    private static string? ReadArchitecture(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

        if (reader.ReadUInt32() != 0x46554747) return null; // "GGUF" little-endian
        uint version = reader.ReadUInt32();
        if (version is < 2 or > 3) return null;             // v1 used 32-bit counts; unsupported

        _ = reader.ReadUInt64();                            // tensor count
        ulong kvCount = reader.ReadUInt64();

        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadString(reader);
            uint valueType = reader.ReadUInt32();

            if (key == "general.architecture" && valueType == TypeString)
                return ReadString(reader);

            SkipValue(reader, valueType);
        }
        return null;
    }

    private static string ReadString(BinaryReader reader)
    {
        long len = checked((long)reader.ReadUInt64());
        if (len is < 0 or > MaxStringBytes)
            throw new InvalidDataException($"GGUF string length {len} out of range");
        return Encoding.UTF8.GetString(reader.ReadBytes((int)len));
    }

    private static void SkipValue(BinaryReader reader, uint valueType)
    {
        switch (valueType)
        {
            case TypeUint8 or TypeInt8 or TypeBool:
                reader.BaseStream.Seek(1, SeekOrigin.Current); break;
            case TypeUint16 or TypeInt16:
                reader.BaseStream.Seek(2, SeekOrigin.Current); break;
            case TypeUint32 or TypeInt32 or TypeFloat32:
                reader.BaseStream.Seek(4, SeekOrigin.Current); break;
            case TypeUint64 or TypeInt64 or TypeFloat64:
                reader.BaseStream.Seek(8, SeekOrigin.Current); break;
            case TypeString:
                long len = checked((long)reader.ReadUInt64());
                if (len is < 0 or > MaxStringBytes)
                    throw new InvalidDataException($"GGUF string length {len} out of range");
                reader.BaseStream.Seek(len, SeekOrigin.Current);
                break;
            case TypeArray:
                uint elemType = reader.ReadUInt32();
                ulong count = reader.ReadUInt64();
                for (ulong i = 0; i < count; i++) SkipValue(reader, elemType);
                break;
            default:
                throw new InvalidDataException($"Unknown GGUF value type {valueType}");
        }
    }
}
