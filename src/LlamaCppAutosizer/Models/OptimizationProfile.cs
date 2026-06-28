using System.Text.Json;

namespace LlamaCppAutosizer.Models;

public enum ProfileType { Chat, Agentic }

public record ScoringWeights(
    double TgSpeed,
    double PpSpeed,
    double TimeToFirstToken,
    double Quality,
    double ToolSuccess
);

public class ToolDefinition
{
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public object Parameters { get; set; } = new { type = "object", properties = new { } };
}

public record class OptimizationProfile
{
    public ProfileType Type { get; init; }
    public string Name { get; init; } = "";
    public string Description { get; init; } = "";
    public ScoringWeights Weights { get; init; } = new(0.4, 0.1, 0.3, 0.2, 0.0);

    // Benchmark parameters
    public int TargetContextSize { get; init; } = 8192;
    public int MaxGenerateTokens { get; init; } = 256;
    public int WarmupRuns { get; init; } = 2;

    public List<string> WarmupPrompts { get; init; } = [];
    public List<string> BenchmarkPrompts { get; init; } = [];
    public List<ToolDefinition> ToolDefinitions { get; init; } = [];
    public List<string> ToolBenchmarkPrompts { get; init; } = [];

    // -------------------------------------------------------------------------
    // Built-in profiles
    // -------------------------------------------------------------------------

    public static OptimizationProfile Chat() => new()
    {
        Type = ProfileType.Chat,
        Name = "Chat",
        Description = "Low latency, responsive generation — ideal for interactive conversations",
        Weights = new(TgSpeed: 0.35, PpSpeed: 0.10, TimeToFirstToken: 0.35, Quality: 0.20, ToolSuccess: 0.00),
        TargetContextSize = 8192,
        MaxGenerateTokens = 256,
        WarmupRuns = 2,
        WarmupPrompts =
        [
            "Hello!",
            "What is 2 + 2?",
        ],
        BenchmarkPrompts =
        [
            "Tell me a short story about a robot who learns to paint. Keep it under 150 words.",
            "Explain the concept of recursion in programming with a brief, clear example.",
            "What are three practical tips for improving sleep quality?",
            "Describe the water cycle in 3–4 sentences.",
            "Write a haiku about autumn leaves.",
        ],
    };

    public static OptimizationProfile Agentic() => new()
    {
        Type = ProfileType.Agentic,
        Name = "Agentic",
        Description = "High throughput, reliable structured outputs — ideal for tool-calling agent loops",
        Weights = new(TgSpeed: 0.15, PpSpeed: 0.25, TimeToFirstToken: 0.10, Quality: 0.20, ToolSuccess: 0.30),
        TargetContextSize = 32768,
        MaxGenerateTokens = 512,
        WarmupRuns = 1,
        WarmupPrompts =
        [
            "List any tools you have available.",
        ],
        BenchmarkPrompts =
        [
            "Analyze this Python snippet and identify any issues:\n\ndef fib(n):\n    if n == 0: return 0\n    return fib(n-1) + fib(n-2)\n\nBe concise.",
            "Break down a software release plan into 5 ordered steps. Be specific and concise.",
            "Given a list of tasks: [email clients, update docs, fix bug #42, review PR]. Prioritize them by urgency and explain briefly.",
        ],
        ToolDefinitions =
        [
            new()
            {
                Name = "search_web",
                Description = "Search the web for information.",
                Parameters = new
                {
                    type = "object",
                    properties = new
                    {
                        query = new { type = "string", description = "Search query" }
                    },
                    required = new[] { "query" }
                }
            },
            new()
            {
                Name = "read_file",
                Description = "Read the contents of a file from disk.",
                Parameters = new
                {
                    type = "object",
                    properties = new
                    {
                        path = new { type = "string", description = "Absolute file path" }
                    },
                    required = new[] { "path" }
                }
            },
            new()
            {
                Name = "run_code",
                Description = "Execute a code snippet and return stdout.",
                Parameters = new
                {
                    type = "object",
                    properties = new
                    {
                        language = new { type = "string", @enum = new[] { "python", "javascript", "bash" } },
                        code = new { type = "string" }
                    },
                    required = new[] { "language", "code" }
                }
            },
        ],
        ToolBenchmarkPrompts =
        [
            "Search the web for the current weather in Paris and tell me what you found.",
            "Read the file at /etc/hosts and summarize its contents.",
            "Search for how to reverse a linked list in Python, then run a quick example.",
        ],
    };

    // -------------------------------------------------------------------------
    // Scoring
    // -------------------------------------------------------------------------

    public double ScoreResult(BenchmarkResult result)
    {
        double tgScore = NormalizeRate(result.GenerationRate, lower: 5, upper: 80);
        double ppScore = NormalizeRate(result.PromptProcessingRate, lower: 50, upper: 2000);
        double ttftScore = NormalizeTtft(result.TimeToFirstTokenMs, worst: 5000, best: 200);
        double qualityScore = result.QualityScore;
        double toolScore = result.ToolSuccessRate;

        return Math.Clamp(
            Weights.TgSpeed * tgScore
          + Weights.PpSpeed * ppScore
          + Weights.TimeToFirstToken * ttftScore
          + Weights.Quality * qualityScore
          + Weights.ToolSuccess * toolScore,
            0, 1);
    }

    static double NormalizeRate(double actual, double lower, double upper)
        => Math.Clamp((actual - lower) / (upper - lower), 0, 1);

    static double NormalizeTtft(double actualMs, double worst, double best)
        => Math.Clamp((worst - actualMs) / (worst - best), 0, 1);

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    public string ToolsAsJson()
    {
        if (ToolDefinitions.Count == 0) return "[]";
        return JsonSerializer.Serialize(ToolDefinitions.Select(t => new
        {
            type = "function",
            function = new { t.Name, t.Description, parameters = t.Parameters }
        }), new JsonSerializerOptions { WriteIndented = false });
    }
}
