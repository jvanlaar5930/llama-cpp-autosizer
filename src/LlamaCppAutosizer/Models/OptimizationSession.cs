namespace LlamaCppAutosizer.Models;

public class ParameterChange
{
    public string Parameter { get; init; } = "";
    public object? OldValue { get; init; }
    public object? NewValue { get; init; }
    public string Reasoning { get; init; } = "";
    public string Source { get; init; } = "heuristic";   // "heuristic" | "heuristic-rescue" | "llm" | "llm-push" | "llm-rescue" | "claude" | "claude-push" | "claude-rescue" | "user"

    // Optional second change applied atomically with this one — used by the final-push
    // recommender to explore parameter combinations a single greedy step can't reach.
    public ParameterChange? Combined { get; init; }

    /// <summary>This change followed by any combined changes, flattened.</summary>
    public IEnumerable<ParameterChange> Chain()
    {
        yield return this;
        if (Combined is not null)
            foreach (var c in Combined.Chain()) yield return c;
    }

    /// <summary>All parameter names touched by this change, including combined ones.</summary>
    public IEnumerable<string> ParameterNames() => Chain().Select(c => c.Parameter);

    /// <summary>"Param → value" display, chaining combined changes with " + ".</summary>
    public string Describe() =>
        $"{Parameter} → {NewValue ?? "default"}" + (Combined is null ? "" : $" + {Combined.Describe()}");
}

public class OptimizationIteration
{
    public int Number { get; init; }
    public LlamaSettings Settings { get; init; } = new();
    public BenchmarkResult Result { get; init; } = new();
    public ParameterChange? AppliedChange { get; init; }
    public bool IsBestSoFar { get; set; }
    public string? StatusMessage { get; set; }

    // End-of-run re-benchmark of the champion config. Excluded from Best ranking —
    // its measurements are folded into the champion iteration's result instead.
    public bool IsVerification { get; init; }
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

    // Iterations imported from earlier optimization runs of the same model + profile
    // (seeded at run setup, see MainMenu). They feed the duplicate guard and the
    // recommenders' "already tried" knowledge so a new run never re-benchmarks a
    // configuration a previous run already measured — but they are NOT part of this
    // run's history: never serialized, never ranked by Best, never shown in the UI.
    [System.Text.Json.Serialization.JsonIgnore]
    public List<OptimizationIteration> PriorIterations { get; init; } = [];

    // Once generation speed reaches this, further tuning should spend headroom on quality
    // instead of chasing more raw speed. Shared with RecommendationService's heuristic tiers.
    // User-configurable per run (see MainMenu's optimization setup prompt); 30 t/s default.
    public double TargetTgSpeed { get; set; } = 30.0;

    // Quality scores below this indicate degenerate output (e.g. a repetition loop) —
    // such configs are never preferred as Best while a healthy alternative exists, and
    // the heuristic recommender treats them as "fix quality before chasing speed".
    public const double MinHealthyQuality = 0.3;

    /// <summary>
    /// The iteration that future recommendations should branch from, and the one reported
    /// as the run's result.
    ///
    /// Below the speed target: highest composite score seen — chase the best available
    /// speed/quality balance.
    ///
    /// At/above the speed target: the most *capable* iteration that still clears the target —
    /// healthy output first, then largest context, then most MoE experts, then composite score.
    /// This lets quality-oriented moves (restoring MoE experts, growing context) stick as the
    /// new anchor even though they lower the speed-weighted composite score, while a move that
    /// merely lost speed with no capability gain does NOT displace the anchor (ranking by
    /// recency here caused the optimizer to anchor on worse configs and then oscillate by
    /// reverting them, re-benchmarking configurations it had already run).
    /// </summary>
    public OptimizationIteration? Best
    {
        get
        {
            // Skip start-failure placeholders (GenerationRate == 0 means the benchmark never
            // ran) and end-of-run verification re-benchmarks (their measurements are folded
            // into the champion iteration they verify).
            var valid = Iterations.Where(i => i.Result.GenerationRate > 0 && !i.IsVerification).ToList();
            if (valid.Count == 0) return Iterations.MaxBy(i => i.Result.CompositeScore);

            var meetingTarget = valid.Where(i => i.Result.GenerationRate >= TargetTgSpeed).ToList();
            if (meetingTarget.Count == 0) return valid.MaxBy(i => i.Result.CompositeScore);

            return meetingTarget
                .OrderBy(i => i.Result.QualityScore >= MinHealthyQuality) // degenerate output loses to any healthy config
                .ThenBy(i => i.Settings.ContextSize)
                .ThenBy(i => i.Settings.MoeExpertUsed ?? int.MaxValue)    // null = model default = full quality
                .ThenBy(i => i.Result.CompositeScore)
                .ThenBy(i => i.Number)
                .Last();
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

    // File name only — SessionPersistenceService combines this with the configured
    // sessions directory (see AppSettingsService) to get the full path.
    public string SessionFileName =>
        $"{ModelName}_{Profile}_{StartedAt:yyyyMMdd_HHmmss}.json";

    public void AddIteration(OptimizationIteration iter)
    {
        Iterations.Add(iter);
        iter.IsBestSoFar = Best == iter;
    }

    /// <summary>
    /// True if a configuration identical to <paramref name="candidate"/> has already been
    /// benchmarked (or attempted — start failures count too) in this session or in a seeded
    /// previous run (<see cref="PriorIterations"/>). Used to stop the optimizer from
    /// re-running settings it has already measured.
    /// </summary>
    public bool HasTestedConfiguration(LlamaSettings candidate)
    {
        var fp = candidate.Fingerprint();
        return Iterations.Concat(PriorIterations).Any(i => i.Settings.Fingerprint() == fp);
    }
}
