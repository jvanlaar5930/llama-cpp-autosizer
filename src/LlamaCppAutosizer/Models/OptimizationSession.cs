namespace LlamaCppAutosizer.Models;

public class ParameterChange
{
    public string Parameter { get; init; } = "";
    public object? OldValue { get; init; }
    public object? NewValue { get; init; }
    public string Reasoning { get; init; } = "";
    public string Source { get; init; } = "heuristic";   // "heuristic" | "llm" | "user"
}

public class OptimizationIteration
{
    public int Number { get; init; }
    public LlamaSettings Settings { get; init; } = new();
    public BenchmarkResult Result { get; init; } = new();
    public ParameterChange? AppliedChange { get; init; }
    public bool IsBestSoFar { get; set; }
    public string? StatusMessage { get; set; }
}

public class OptimizationSession
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string ModelPath { get; set; } = "";
    public string ModelName => Path.GetFileNameWithoutExtension(ModelPath);
    public ProfileType Profile { get; set; }
    public HardwareInfo Hardware { get; set; } = new();
    public DateTime StartedAt { get; init; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }

    public List<OptimizationIteration> Iterations { get; init; } = [];

    // Once generation speed reaches this, further tuning should spend headroom on quality
    // instead of chasing more raw speed. Shared with RecommendationService's heuristic tiers.
    // User-configurable per run (see MainMenu's optimization setup prompt); 30 t/s default.
    public double TargetTgSpeed { get; set; } = 30.0;

    /// <summary>
    /// The iteration that future recommendations should branch from, and the one reported
    /// as the run's result.
    ///
    /// Below the speed target: highest composite score seen (as before) — chase the best
    /// available speed/quality balance.
    ///
    /// At/above the speed target: the *latest* iteration that still clears the target, even
    /// if its composite score is lower than an earlier, faster-but-lower-quality iteration.
    /// This lets quality-oriented moves (restoring MoE experts, growing context) stick as the
    /// new anchor and keep chaining forward, rather than snapping back to a faster config just
    /// because it scored higher on the speed-weighted composite metric.
    /// </summary>
    public OptimizationIteration? Best
    {
        get
        {
            // Skip start-failure placeholders (GenerationRate == 0 means the benchmark never ran).
            var valid = Iterations.Where(i => i.Result.GenerationRate > 0).ToList();
            if (valid.Count == 0) return Iterations.MaxBy(i => i.Result.CompositeScore);

            var meetingTarget = valid.Where(i => i.Result.GenerationRate >= TargetTgSpeed).ToList();
            return meetingTarget.Count > 0
                ? meetingTarget.MaxBy(i => i.Number)
                : valid.MaxBy(i => i.Result.CompositeScore);
        }
    }

    public BenchmarkResult? BestResult => Best?.Result;
    public LlamaSettings? BestSettings => Best?.Settings;

    public bool IsComplete { get; set; }
    public string? CompletionReason { get; set; }

    // Free-text steer from the user (e.g. "prioritize low VRAM usage"), passed to the
    // LLM recommender's prompts. Null/empty means no extra guidance was given.
    public string? UserGuidance { get; set; }

    // TurboQuant results if run
    public string? TurboQuantModelPath { get; set; }
    public BenchmarkResult? TurboQuantResult { get; set; }

    public string SessionFile =>
        $"sessions/{ModelName}_{Profile}_{StartedAt:yyyyMMdd_HHmmss}.json";

    public void AddIteration(OptimizationIteration iter)
    {
        Iterations.Add(iter);
        iter.IsBestSoFar = Best == iter;
    }
}
