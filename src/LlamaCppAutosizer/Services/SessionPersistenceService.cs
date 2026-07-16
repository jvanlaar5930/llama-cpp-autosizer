using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class SessionPersistenceService(AppSettingsService appSettings, ILogger<SessionPersistenceService> logger)
{
    private string SessionsDir => appSettings.Current.SessionsDirectory;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(), new ObjectPrimitiveConverter() },
    };

    // ParameterChange.OldValue/NewValue are typed `object?`. Without this, deserialized
    // sessions carry JsonElement there instead of the int/bool/string a live run produces,
    // and consumers like Convert.ToInt32 crash on it (JsonElement is not IConvertible) —
    // seen when prior-run iterations are seeded into a new session's recommendation logic.
    private sealed class ObjectPrimitiveConverter : JsonConverter<object?>
    {
        public override object? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            => reader.TokenType switch
            {
                JsonTokenType.True => true,
                JsonTokenType.False => false,
                JsonTokenType.String => reader.GetString(),
                // Box each branch explicitly — a bare ternary would promote every branch to
                // double (the common numeric type) before boxing, turning ints into doubles.
                JsonTokenType.Number => reader.TryGetInt32(out int i) ? (object)i
                                      : reader.TryGetInt64(out long l) ? (object)l
                                      : reader.GetDouble(),
                // Arrays/objects don't occur in these fields today; keep them as JsonElement.
                _ => JsonSerializer.Deserialize<JsonElement>(ref reader, options),
            };

        public override void Write(Utf8JsonWriter writer, object? value, JsonSerializerOptions options)
        {
            if (value is null) writer.WriteNullValue();
            else JsonSerializer.Serialize(writer, value, value.GetType(), options);
        }
    }

    public async Task SaveAsync(OptimizationSession session)
    {
        try
        {
            Directory.CreateDirectory(SessionsDir);
            string path = Path.Combine(SessionsDir, session.SessionFileName);

            await using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            await JsonSerializer.SerializeAsync(fs, session, JsonOpts);
            logger.LogDebug("Session saved to {File}", path);
        }
        catch (Exception ex)
        {
            logger.LogWarning("Failed to save session: {Msg}", ex.Message);
        }
    }

    public async Task<OptimizationSession?> LoadAsync(string path)
    {
        try
        {
            await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            return await JsonSerializer.DeserializeAsync<OptimizationSession>(fs, JsonOpts);
        }
        catch (Exception ex)
        {
            logger.LogError("Failed to load session from {Path}: {Msg}", path, ex.Message);
            return null;
        }
    }

    public List<string> ListSessions(string? directory = null)
    {
        directory ??= SessionsDir;
        if (!Directory.Exists(directory)) return [];
        return Directory.GetFiles(directory, "*.json", SearchOption.TopDirectoryOnly)
                        .OrderByDescending(f => File.GetLastWriteTime(f))
                        .ToList();
    }

    /// <summary>
    /// Loads every session in the directory and returns those that have at least one result.
    /// Failures are silently skipped.
    /// </summary>
    public async Task<List<OptimizationSession>> LoadAllAsync(string? directory = null)
    {
        var files = ListSessions(directory ?? SessionsDir);
        var results = new List<OptimizationSession>(files.Count);

        foreach (var file in files)
        {
            var session = await LoadAsync(file);
            if (session?.BestResult is not null)
                results.Add(session);
        }

        return results;
    }

    /// <summary>
    /// Loads earlier runs of the same model + profile (matched by model file name so a moved
    /// GGUF still counts), newest first. Incomplete sessions — e.g. a run cut short by a
    /// crash — are included: every iteration they managed to autosave is still a valid
    /// measurement worth reusing.
    /// </summary>
    public async Task<List<OptimizationSession>> LoadPriorSessionsAsync(
        string modelPath, ProfileType profile)
    {
        string modelName = Path.GetFileNameWithoutExtension(modelPath);
        var sessions = await LoadAllAsync();
        return sessions
            .Where(s => s.Profile == profile
                     && string.Equals(s.ModelName, modelName, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(s => s.StartedAt)
            .ToList();
    }

    /// <summary>
    /// Exports a CSV comparing the best result from every session.
    /// </summary>
    public async Task ExportComparisonCsvAsync(
        IEnumerable<OptimizationSession> sessions, string outputPath)
    {
        await using var sw = new StreamWriter(outputPath, append: false, System.Text.Encoding.UTF8);
        await sw.WriteLineAsync(
            "ModelName,Profile,Date,Score,PP_tok_s,TG_tok_s,TTFT_ms," +
            "Iterations,ContextSize,GpuLayers,BatchSize,FlashAttn,CacheTypeK,CacheTypeV," +
            "SettingsSummary");

        foreach (var s in sessions.OrderByDescending(s => s.BestResult!.CompositeScore))
        {
            var r = s.BestResult!;
            var st = s.BestSettings!;
            await sw.WriteLineAsync(string.Join(",",
                Csv(s.ModelName),
                Csv(s.Profile.ToString()),
                Csv(s.StartedAt.ToString("yyyy-MM-dd HH:mm")),
                r.CompositeScore.ToString("F4"),
                r.PromptProcessingRate.ToString("F1"),
                r.GenerationRate.ToString("F1"),
                r.TimeToFirstTokenMs.ToString("F0"),
                s.Iterations.Count.ToString(),
                st.ContextSize.ToString(),
                st.GpuLayers.ToString(),
                st.BatchSize.ToString(),
                st.FlashAttention.ToString(),
                Csv(st.CacheTypeK ?? "f16"),
                Csv(st.CacheTypeV ?? "f16"),
                Csv(st.Summary())));
        }
    }

    private static string Csv(string s)
        => s.Contains(',') || s.Contains('"') ? $"\"{s.Replace("\"", "\"\"")}\"" : s;

    /// <summary>Exports the best settings as a ready-to-use llama-server command.</summary>
    public string ExportAsCliCommand(OptimizationSession session, string serverExecutable)
    {
        if (session.BestSettings is null) return "# No results yet";

        var args = session.BestSettings.ToServerArgs(session.ModelPath);
        return $"{serverExecutable} {string.Join(" ", args)}";
    }

    /// <summary>Exports best settings as a JSON config file.</summary>
    public async Task ExportResultsAsync(OptimizationSession session, string outputPath)
    {
        var export = new
        {
            session.ModelName,
            session.Profile,
            GeneratedAt = DateTime.UtcNow,
            BestScore = session.BestResult?.CompositeScore,
            Metrics = session.BestResult is null ? null : new
            {
                session.BestResult.PromptProcessingRate,
                session.BestResult.GenerationRate,
                session.BestResult.TimeToFirstTokenMs,
            },
            Settings = session.BestSettings,
            Iterations = session.Iterations.Count,
            CompletionReason = session.CompletionReason,
        };

        await using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
        await JsonSerializer.SerializeAsync(fs, export, JsonOpts);
    }
}
