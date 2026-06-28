using LlamaCppAutosizer.Services;
using LlamaCppAutosizer.UI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Spectre.Console;

var services = new ServiceCollection();

// Logging (suppress to avoid cluttering the TUI; errors still show)
services.AddLogging(b => b
    .SetMinimumLevel(LogLevel.Warning)
    .AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Warning));

// HttpClient factory (for llama-server API calls)
services.AddHttpClient();

// Core services
services.AddSingleton<HardwareDetectionService>();
services.AddSingleton<LlamaServerService>();
services.AddSingleton<BenchmarkService>();
services.AddSingleton<RecommendationService>();
services.AddSingleton<OptimizerService>();
services.AddSingleton<TurboQuantService>();
services.AddSingleton<SessionPersistenceService>();

// UI
services.AddSingleton<MainMenu>();

var provider = services.BuildServiceProvider();

// ── Graceful cancellation ────────────────────────────────────────────────────
using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;            // don't kill the process immediately
    cts.Cancel();
    AnsiConsole.MarkupLine("\n[yellow]Stopping… waiting for current run to finish.[/]");
};

// ── Run ──────────────────────────────────────────────────────────────────────
var menu = provider.GetRequiredService<MainMenu>();
try
{
    await menu.RunAsync(cts.Token);
}
catch (OperationCanceledException)
{
    // Clean exit via Ctrl+C
}
finally
{
    // Ensure llama-server is stopped on exit
    await using var server = provider.GetRequiredService<LlamaServerService>();
}

AnsiConsole.MarkupLine("[grey]Goodbye.[/]");
