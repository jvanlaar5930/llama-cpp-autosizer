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

    public OptimizationIteration? Best =>
        Iterations.MaxBy(i => i.Result.CompositeScore);

    public BenchmarkResult? BestResult => Best?.Result;
    public LlamaSettings? BestSettings => Best?.Settings;

    public bool IsComplete { get; set; }
    public string? CompletionReason { get; set; }

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
