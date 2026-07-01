namespace LlamaCppAutosizer.Models;

public class SavedProfile
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string Name { get; set; } = "";
    public string ModelPath { get; set; } = "";
    public LlamaSettings Settings { get; init; } = new();

    // Optional context about where this profile came from
    public ProfileType? OptimizationProfile { get; init; }
    public double? BenchmarkScore { get; init; }
    public double? BenchmarkPpRate { get; init; }
    public double? BenchmarkTgRate { get; init; }
    public double? BenchmarkTtftMs { get; init; }

    public string? Notes { get; set; }
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    public DateTime? LastRunAt { get; set; }

    public string ModelName => Path.GetFileNameWithoutExtension(ModelPath);

    public string ProfileFile => $"profiles/{Id:N}.json";

    /// <summary>Short one-line summary for list display.</summary>
    public string Summary()
    {
        var parts = new List<string>();
        if (OptimizationProfile.HasValue) parts.Add(OptimizationProfile.Value.ToString());
        if (BenchmarkTgRate.HasValue) parts.Add($"TG={BenchmarkTgRate:F0}t/s");
        if (BenchmarkPpRate.HasValue) parts.Add($"PP={BenchmarkPpRate:F0}t/s");
        if (BenchmarkScore.HasValue) parts.Add($"score={BenchmarkScore:F3}");
        return parts.Count > 0 ? string.Join("  ", parts) : Settings.Summary();
    }

    /// <summary>Creates a SavedProfile from a completed session.
    /// Pass <paramref name="chosenIteration"/> to use a specific starred iteration
    /// instead of the automatic best (highest score).</summary>
    public static SavedProfile FromSession(
        OptimizationSession session,
        string name,
        OptimizationIteration? chosenIteration = null)
    {
        var iter = chosenIteration ?? session.Best;
        var r = iter?.Result;
        return new SavedProfile
        {
            Name = name,
            ModelPath = session.ModelPath,
            Settings = (iter?.Settings ?? session.BestSettings!).Clone(),
            OptimizationProfile = session.Profile,
            BenchmarkScore = r?.CompositeScore,
            BenchmarkPpRate = r?.PromptProcessingRate,
            BenchmarkTgRate = r?.GenerationRate,
            BenchmarkTtftMs = r?.TimeToFirstTokenMs,
        };
    }

    /// <summary>Creates a SavedProfile from manually-edited settings.</summary>
    public static SavedProfile FromSettings(string name, string modelPath, LlamaSettings settings)
        => new() { Name = name, ModelPath = modelPath, Settings = settings.Clone() };
}
