using System.Text.Json;

namespace LlamaCppAutosizer.Models;

public enum ProfileType { Chat, Agentic }

public record ScoringWeights(
    double TgSpeed,
    double PpSpeed,
    double TimeToFirstToken,
    double Quality,
    double ToolSuccess,
    double AgentLoop = 0.0
);

public class ToolDefinition
{
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public object Parameters { get; set; } = new { type = "object", properties = new { } };
}

public class ToolBenchmarkCase
{
    public string Prompt { get; set; } = "";
    // Tool names that count as the correct choice for this prompt; empty = any tool call.
    public string[] ExpectedTools { get; set; } = [];
}

// A short prompt with an objectively checkable answer. Degraded configs (aggressive KV
// cache quantization, too few MoE experts) stay fluent but start getting these wrong,
// which is exactly the signal the speed-oriented metrics can't see.
public class AccuracyPrompt
{
    public string Prompt { get; set; } = "";
    // Response is correct if it contains ANY of these (case-insensitive substring match).
    public string[] AcceptableAnswers { get; set; } = [];
}

// A full agent-style tool loop: initial prompt, N tool round-trips with a synthetic
// (fabricated) tool result injected after each call, then a final free-text turn. Unlike
// ToolBenchmarkCase (one shot, first call only) this exercises the parts of a real tooling
// harness that a single-turn test can't see: whether the model can consume an injected tool
// result and either chain the next call or synthesize a correct final answer, and whether
// conversation-history reuse across turns behaves like it would in a real session.
public class AgentLoopCase
{
    public string Prompt { get; set; } = "";
    // Tool name expected at each successive round-trip; length determines how many
    // tool-call turns are run before the final free-text turn.
    public string[] ExpectedToolSequence { get; set; } = [];
    // Fabricated tool output fed back for each step (same length as ExpectedToolSequence).
    public string[] ToolResults { get; set; } = [];
    // Substrings the final answer should contain (case-insensitive, any match). Empty means
    // just check that a coherent non-tool-call answer was produced.
    public string[] FinalAnswerContains { get; set; } = [];
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
    public List<ToolBenchmarkCase> ToolBenchmarks { get; init; } = [];
    public List<AccuracyPrompt> AccuracyPrompts { get; init; } = [];

    // Multi-turn tool-loop cases. Agentic-only — left empty by Chat() since chat mode
    // never exercises tool calling.
    public List<AgentLoopCase> AgentLoopCases { get; init; } = [];

    // Opt-in long-form generation used to surface degenerate repetition loops
    // ("is is is is...") that short benchmark prompts rarely run long enough to reach.
    public string? StressTestPrompt { get; init; } =
        "List and explain in detail 40 distinct, non-repeating tips for improving software code review quality. " +
        "Number each one and give a full paragraph of justification for each.";
    public int StressTestMaxTokens { get; init; } = 1536;

    // -------------------------------------------------------------------------
    // Built-in profiles
    // -------------------------------------------------------------------------

    public static OptimizationProfile Chat() => new()
    {
        Type = ProfileType.Chat,
        Name = "Chat",
        Description = "Low latency, responsive generation — ideal for interactive conversations",
        Weights = new(TgSpeed: 0.35, PpSpeed: 0.10, TimeToFirstToken: 0.35, Quality: 0.20, ToolSuccess: 0.00, AgentLoop: 0.00),
        TargetContextSize = 8192,
        MaxGenerateTokens = 256,
        WarmupRuns = 2,
        WarmupPrompts =
        [
            "Hello!",
            "What is 2 + 2?",
        ],
        // Each prompt includes a short prior-turn so TG is measured with a small amount
        // of KV cache already populated, closer to real conversational use.
        BenchmarkPrompts =
        [
            "User: Can you help me learn something new today?\nAssistant: Of course! What topic interests you — science, history, coding, or something else?\nUser: Tell me a short story about a robot who learns to paint. Keep it under 150 words.",
            "User: I'm brushing up on programming concepts.\nAssistant: Great — happy to help. Where would you like to start?\nUser: Explain the concept of recursion in programming with a brief, clear example.",
            "User: Give me some life advice.\nAssistant: Happy to share some practical wisdom. What area — health, productivity, or relationships?\nUser: What are three practical tips for improving sleep quality?",
            "User: Teach me something about nature.\nAssistant: Nature has so many fascinating systems. Want something about weather, ecosystems, or geology?\nUser: Describe the water cycle in 3–4 sentences.",
            "User: Can you write me some poetry?\nAssistant: I'd love to. Any particular form or subject in mind?\nUser: Write a haiku about autumn leaves.",
        ],
        AccuracyPrompts = DefaultAccuracyPrompts(),
    };

    public static OptimizationProfile Agentic() => new()
    {
        Type = ProfileType.Agentic,
        Name = "Agentic",
        Description = "High throughput, reliable structured outputs — ideal for tool-calling agent loops",
        Weights = new(TgSpeed: 0.10, PpSpeed: 0.20, TimeToFirstToken: 0.10, Quality: 0.15, ToolSuccess: 0.20, AgentLoop: 0.25),
        TargetContextSize = 32768,
        MaxGenerateTokens = 512,
        WarmupRuns = 1,
        WarmupPrompts =
        [
            "List any tools you have available.",
        ],
        // Prompts include a realistic multi-turn conversation preamble so that TG is
        // measured with a partially-filled KV cache, matching real agentic workloads.
        BenchmarkPrompts =
        [
            // Preamble: ~400-token coding-agent conversation, then the actual task
            "You are an expert AI coding assistant. Here is the conversation so far:\n\n" +
            "<USER>: I have a FastAPI app with SQLAlchemy and PostgreSQL. The report endpoint takes 30+ seconds.\n\n" +
            "<ASSISTANT>: This is almost certainly an N+1 query problem. Each row you iterate triggers a new SELECT. Fix it with eager loading:\n" +
            "```python\n# Bad\nfor order in orders:\n    items = order.items  # new query each time\n\n" +
            "# Good\norders = session.query(Order).options(joinedload(Order.items)).all()\n```\n" +
            "Can you share the endpoint code so I can give specific advice?\n\n" +
            "<USER>: Here it is:\n```python\nasync def generate_report(db, user_id, date_from, date_to):\n" +
            "    result = await db.execute(select(Order).where(Order.user_id == user_id))\n" +
            "    orders = result.scalars().all()\n    report = []\n" +
            "    for order in orders:\n        r = await db.execute(select(OrderItem).where(OrderItem.order_id == order.id))\n" +
            "        items = r.scalars().all()\n        report.append({'id': order.id, 'total': sum(i.price * i.qty for i in items)})\n" +
            "    return report\n```\n\n" +
            "<ASSISTANT>: Classic N+1. Here is the optimized single-query version with joinedload and a date filter:\n\n" +
            "Now add a composite index and a streaming fallback for large result sets. What database version are you on?\n\n" +
            "<USER>: PostgreSQL 15. Now I need you to:\n" +
            "1. Write the optimized query with proper eager loading and date filtering.\n" +
            "2. Add a covering index migration.\n" +
            "3. Explain why each change helps. Be thorough.",

            "You are an expert AI coding assistant. Previous context:\n\n" +
            "<USER>: Set up CI/CD for my Python monorepo with GitHub Actions.\n\n" +
            "<ASSISTANT>: Here is a matrix strategy that runs lint, test, and build in parallel across Python 3.11 and 3.12:\n" +
            "```yaml\njobs:\n  test:\n    strategy:\n      matrix:\n        python-version: ['3.11', '3.12']\n    steps:\n" +
            "      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: ${{ matrix.python-version }}\n" +
            "      - run: pip install -e '.[dev]'\n      - run: pytest --cov\n```\n\n" +
            "<USER>: Add deployment to AWS ECS with blue/green and auto-rollback on failed health check.\n\n" +
            "<ASSISTANT>: You need a separate deploy job gated on the test job, using the AWS CLI and ECS blue/green via CodeDeploy:\n\n" +
            "The rollback trigger is a CloudWatch alarm on the ALB target group unhealthy host count. Let me write the full workflow.\n\n" +
            "<USER>: Also add Dependabot config, branch protection rules via Terraform, and a release drafter. List every file you will create and what goes in each.",
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
        ToolBenchmarks =
        [
            new()
            {
                Prompt = "Search the web for the current weather in Paris and tell me what you found.",
                ExpectedTools = ["search_web"],
            },
            new()
            {
                Prompt = "Read the file at /etc/hosts and summarize its contents.",
                ExpectedTools = ["read_file"],
            },
            new()
            {
                Prompt = "Search for how to reverse a linked list in Python, then run a quick example.",
                ExpectedTools = ["search_web", "run_code"],
            },
        ],
        AccuracyPrompts = DefaultAccuracyPrompts(),
        AgentLoopCases =
        [
            new()
            {
                Prompt = "Search the web for the current population of Tokyo, then tell me the number you found.",
                ExpectedToolSequence = ["search_web"],
                ToolResults =
                [
                    "{\"results\":[{\"title\":\"Tokyo population 2024\",\"snippet\":\"Tokyo's metropolitan population is approximately 14 million people as of 2024.\"}]}",
                ],
                FinalAnswerContains = ["14 million"],
            },
            new()
            {
                Prompt = "Read the file at /etc/hosts, then tell me how many non-empty lines it contains.",
                ExpectedToolSequence = ["read_file"],
                ToolResults =
                [
                    "{\"content\":\"127.0.0.1 localhost\\n::1 localhost\\n10.0.0.5 devbox\\n\"}",
                ],
                FinalAnswerContains = ["3"],
            },
            new()
            {
                Prompt = "Search the web for how to reverse a linked list in Python, then run the example you find " +
                         "against the list [1, 2, 3] and tell me what it prints.",
                ExpectedToolSequence = ["search_web", "run_code"],
                ToolResults =
                [
                    "{\"results\":[{\"title\":\"Reverse a linked list - Python\",\"snippet\":" +
                    "\"def reverse(head):\\n    prev = None\\n    while head:\\n        head.next, prev, head = prev, head, head.next\\n    return prev\"}]}",
                    "{\"stdout\":\"[3, 2, 1]\"}",
                ],
                FinalAnswerContains = ["3, 2, 1", "[3, 2, 1]"],
            },
        ],
    };

    // Easy enough that any healthy config answers all of them; a config degraded by
    // aggressive KV quantization or expert reduction starts missing them first.
    private static List<AccuracyPrompt> DefaultAccuracyPrompts() =>
    [
        new()
        {
            Prompt = "What is 17 multiplied by 23? Reply with only the number.",
            AcceptableAnswers = ["391"],
        },
        new()
        {
            Prompt = "What is the capital city of Australia? Reply with only the city name.",
            AcceptableAnswers = ["Canberra"],
        },
        new()
        {
            Prompt = "Which number is larger: 0.9 or 0.11? Reply with only that number.",
            AcceptableAnswers = ["0.9"],
        },
        new()
        {
            Prompt = "Complete the sequence and reply with only the next number: 2, 4, 8, 16, ...",
            AcceptableAnswers = ["32"],
        },
        new()
        {
            Prompt = "What is 9 multiplied by 14? Reply with only the number.",
            AcceptableAnswers = ["126"],
        },
        new()
        {
            Prompt = "How many days are in a leap year? Reply with only the number.",
            AcceptableAnswers = ["366"],
        },
        new()
        {
            Prompt = "Which planet is closest to the sun? Reply with only the planet name.",
            AcceptableAnswers = ["Mercury"],
        },
        new()
        {
            Prompt = "Alphabetically, which word comes first: zebra or apple? Reply with only that word.",
            AcceptableAnswers = ["apple"],
        },
        new()
        {
            Prompt = "Complete the sequence and reply with only the next number: 1, 1, 2, 3, 5, 8, ...",
            AcceptableAnswers = ["13"],
        },
        new()
        {
            Prompt = "How many hours are in three days? Reply with only the number.",
            AcceptableAnswers = ["72"],
        },
    ];

    // -------------------------------------------------------------------------
    // Scoring
    // -------------------------------------------------------------------------

    public double ScoreResult(BenchmarkResult result)
    {
        double tgScore = NormalizeRate(result.GenerationRate, lower: 5, upper: 300);
        double ppScore = NormalizeRate(result.PromptProcessingRate, lower: 50, upper: 10000);
        // TTFT here is the server-reported prompt-eval time (see BenchmarkService), typically
        // 50–500 ms on GPU — bounds sized to that, not to full-response wall time.
        double ttftScore = NormalizeTtft(result.TimeToFirstTokenMs, worst: 3000, best: 50);
        double qualityScore = result.QualityScore;
        double toolScore = result.ToolSuccessRate;
        double agentLoopScore = result.AgentLoopScore;

        return Math.Clamp(
            Weights.TgSpeed * tgScore
          + Weights.PpSpeed * ppScore
          + Weights.TimeToFirstToken * ttftScore
          + Weights.Quality * qualityScore
          + Weights.ToolSuccess * toolScore
          + Weights.AgentLoop * agentLoopScore,
            0, 1);
    }

    // Log-scale so resolution is preserved across the whole range: on fast hardware a linear
    // scale with a low ceiling clamps every config to 1.0 and the optimizer goes blind to
    // real speed differences (which then read as "no improvement" and end the run early).
    static double NormalizeRate(double actual, double lower, double upper)
        => actual <= lower ? 0
         : Math.Clamp(Math.Log(actual / lower) / Math.Log(upper / lower), 0, 1);

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
