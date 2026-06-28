using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class SessionPersistenceService(ILogger<SessionPersistenceService> logger)
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
    };

    public async Task SaveAsync(OptimizationSession session)
    {
        try
        {
            var dir = Path.GetDirectoryName(session.SessionFile) ?? "sessions";
            Directory.CreateDirectory(dir);

            await using var fs = new FileStream(session.SessionFile, FileMode.Create, FileAccess.Write);
            await JsonSerializer.SerializeAsync(fs, session, JsonOpts);
            logger.LogDebug("Session saved to {File}", session.SessionFile);
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

    public List<string> ListSessions(string directory = "sessions")
    {
        if (!Directory.Exists(directory)) return [];
        return Directory.GetFiles(directory, "*.json", SearchOption.TopDirectoryOnly)
                        .OrderByDescending(f => File.GetLastWriteTime(f))
                        .ToList();
    }

    /// <summary>
    /// Loads every session in the directory and returns those that have at least one result.
    /// Failures are silently skipped.
    /// </summary>
    public async Task<List<OptimizationSession>> LoadAllAsync(string directory = "sessions")
    {
        var files = ListSessions(directory);
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
