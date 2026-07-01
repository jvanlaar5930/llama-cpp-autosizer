namespace LlamaCppAutosizer.Models;

public class RunTimings
{
    public double PromptTokens { get; set; }
    public double PromptMs { get; set; }
    public double PromptPerSecond { get; set; }

    public double PredictedTokens { get; set; }
    public double PredictedMs { get; set; }
    public double PredictedPerSecond { get; set; }

    public double TotalMs { get; set; }

    /// <summary>Time from request sent to first response byte, in ms.</summary>
    public double TimeToFirstTokenMs { get; set; }

    /// <summary>Generated text, kept only long enough to check for degenerate repetition.</summary>
    public string GeneratedText { get; set; } = "";
}

public class BenchmarkResult
{
    public LlamaSettings Settings { get; init; } = new();
    public string ModelPath { get; init; } = "";

    // Aggregated from all successful runs
    public double PromptProcessingRate { get; set; }    // tokens/s (PP / prefill)
    public double GenerationRate { get; set; }           // tokens/s (TG / decode)
    public double TimeToFirstTokenMs { get; set; }       // ms
    public double AverageTotalMs { get; set; }

    // Resource usage at time of benchmark
    public long VramUsageMb { get; set; }
    public long RamUsageMb { get; set; }

    // Run statistics
    public int SuccessfulRuns { get; set; }
    public int FailedRuns { get; set; }
    public double ErrorRate => FailedRuns == 0 ? 0 : (double)FailedRuns / (SuccessfulRuns + FailedRuns);

    // Scored metrics (0–1 each)
    public double QualityScore { get; set; }
    public double ToolSuccessRate { get; set; }  // agentic only

    // Fraction of accuracy prompts answered correctly (1.0 when the profile defines none).
    // Folded into QualityScore so degraded configs (aggressive KV quant, too few MoE
    // experts) lose score even when their output is fluent.
    public double AccuracyScore { get; set; } = 1.0;

    // Final composite score for this profile (set by profile scorer)
    public double CompositeScore { get; set; }

    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string? Notes { get; set; }

    public List<RunTimings> IndividualRuns { get; init; } = [];

    public string FormatMetrics()
        => $"PP {PromptProcessingRate,7:F1} t/s  TG {GenerationRate,6:F1} t/s  " +
           $"TTFT {TimeToFirstTokenMs,6:F0} ms  Score {CompositeScore:F3}";
}
