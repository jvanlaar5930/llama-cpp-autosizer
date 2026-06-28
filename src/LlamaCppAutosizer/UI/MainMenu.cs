using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

public class MainMenu(
    HardwareDetectionService hwService,
    OptimizerService optimizer,
    TurboQuantService turboQuant,
    SessionPersistenceService persistence)
{
    // -------------------------------------------------------------------------
    // Configuration (loaded once, persisted to disk)
    // -------------------------------------------------------------------------

    private string _serverExecutable = "llama-server";
    private string _modelFolder = "";
    private string _modelPath = "";
    private OptimizationProfile _profile = OptimizationProfile.Chat();
    private LlamaSettings _manualSettings = new();
    private HardwareInfo? _hardware;

    private const string ConfigFile = "autosizer-config.json";

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public async Task RunAsync(CancellationToken ct = default)
    {
        LoadConfig();
        ShowBanner();

        while (!ct.IsCancellationRequested)
        {
            AnsiConsole.WriteLine();
            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[bold yellow]llama.cpp Auto-Sizer[/] — Main Menu")
                    .PageSize(15)
                    .HighlightStyle(new Style(Color.Yellow))
                    .AddChoices(
                        "1. Select Model Folder  (auto-discover .gguf)",
                        "2. Choose Profile  (Chat / Agentic)",
                        "3. Set llama-server Path",
                        "4. Detect Hardware",
                        "5. Run Auto-Optimization",
                        "6. Edit Settings Manually",
                        "7. TurboQuant Options",
                        "8. View / Load Sessions",
                        "9. Historical Benchmark Comparison",
                        "0. Exit"
                    ));

            try
            {
                switch (choice[0])
                {
                    case '1': await SelectModelAsync(ct); break;
                    case '2': ChooseProfile(); break;
                    case '3': SetServerPath(); break;
                    case '4': await DetectHardwareAsync(ct); break;
                    case '5': await RunOptimizationAsync(ct); break;
                    case '6': EditSettingsManually(); break;
                    case '7': await TurboQuantMenuAsync(ct); break;
                    case '8': await ViewSessionsAsync(ct); break;
                    case '9': await HistoricalComparisonAsync(ct); break;
                    case '0': return;
                }
                SaveConfig();
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                AnsiConsole.WriteLine("Press any key to continue...");
                Console.ReadKey(intercept: true);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Menu handlers
    // -------------------------------------------------------------------------

    private async Task SelectModelAsync(CancellationToken ct)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold]Select Model[/]").RuleStyle("cyan"));

        // ── 1. Pick / confirm folder ─────────────────────────────────────────
        string defaultFolder = _modelFolder.Length > 0 && Directory.Exists(_modelFolder)
            ? _modelFolder
            : (!string.IsNullOrEmpty(_modelPath)
                ? Path.GetDirectoryName(_modelPath) ?? Directory.GetCurrentDirectory()
                : Directory.GetCurrentDirectory());

        string folder = AnsiConsole.Prompt(
            new TextPrompt<string>("[grey]Folder to scan for[/] [cyan].gguf[/] [grey]files:[/]")
                .DefaultValue(defaultFolder)
                .Validate(p => Directory.Exists(p)
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Folder not found")));

        _modelFolder = folder;

        bool recursive = AnsiConsole.Confirm("Search sub-folders?", false);

        // ── 2. Discover models ───────────────────────────────────────────────
        List<(string Path, long SizeMb)> models = [];
        await AnsiConsole.Status()
            .StartAsync("Scanning for .gguf files...", async _ =>
            {
                var option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                models = Directory
                    .EnumerateFiles(folder, "*.gguf", option)
                    .Select(p => (Path: p, SizeMb: new FileInfo(p).Length / (1024 * 1024)))
                    .OrderBy(m => Path.GetFileName(m.Path))
                    .ToList();
                await Task.CompletedTask;
            });

        if (models.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No .gguf files found.[/] Enter a path manually:");
            string manual = AnsiConsole.Prompt(
                new TextPrompt<string>("Full path to .gguf file:")
                    .DefaultValue(_modelPath)
                    .Validate(p => File.Exists(p)
                        ? ValidationResult.Success()
                        : ValidationResult.Error("File not found")));
            SetModel(manual);
            return;
        }

        // ── 3. Display discovered models ─────────────────────────────────────
        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderModelList(models);
        AnsiConsole.WriteLine();

        // Build choice list: file name + size as the label
        var choices = models
            .Select((m, i) =>
                $"{i + 1,3}. {Path.GetFileName(m.Path)}  [{m.SizeMb:N0} MB]")
            .Append("── Enter path manually ──")
            .ToList();

        // Pre-select currently active model if in this folder
        int defaultIdx = models.FindIndex(m =>
            m.Path.Equals(_modelPath, StringComparison.OrdinalIgnoreCase));
        string defaultChoice = defaultIdx >= 0 ? choices[defaultIdx] : choices[0];

        string selected = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Choose a model:")
                .PageSize(Math.Min(20, models.Count + 2))
                .HighlightStyle(new Style(Color.Cyan1))
                .AddChoices(choices)
                .Mode(SelectionMode.Independent));

        if (selected.StartsWith("──"))
        {
            string manual = AnsiConsole.Prompt(
                new TextPrompt<string>("Full path to .gguf file:")
                    .Validate(p => File.Exists(p)
                        ? ValidationResult.Success()
                        : ValidationResult.Error("File not found")));
            SetModel(manual);
        }
        else
        {
            // Parse index from "  3. filename.gguf  [4,321 MB]"
            int dot = selected.IndexOf('.');
            if (dot > 0 && int.TryParse(selected[..dot].Trim(), out int idx) && idx >= 1 && idx <= models.Count)
                SetModel(models[idx - 1].Path);
        }
    }

    private void SetModel(string path)
    {
        _modelPath = path;
        _manualSettings = new LlamaSettings();
        AnsiConsole.MarkupLine($"[green]Model:[/] {Path.GetFileName(_modelPath)}  " +
            $"[grey]({new FileInfo(_modelPath).Length / (1024 * 1024):N0} MB)[/]");
    }

    private void ChooseProfile()
    {
        AnsiConsole.WriteLine();
        var choice = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Select optimization [bold]profile[/]:")
                .AddChoices("Chat — low latency, responsive generation",
                             "Agentic — high throughput, reliable tool use"));

        _profile = choice.StartsWith("Chat")
            ? OptimizationProfile.Chat()
            : OptimizationProfile.Agentic();

        AnsiConsole.MarkupLine($"[green]Profile:[/] {_profile.Name} — {_profile.Description}");

        // Let the user adjust scoring weights if desired
        if (AnsiConsole.Confirm("Customize scoring weights?", false))
            CustomizeWeights();
    }

    private void CustomizeWeights()
    {
        AnsiConsole.MarkupLine("[grey]Weights must sum to 1.0. Leave blank to keep defaults.[/]");

        double tg = PromptWeight("TG speed weight", _profile.Weights.TgSpeed);
        double pp = PromptWeight("PP speed weight", _profile.Weights.PpSpeed);
        double ttft = PromptWeight("TTFT (lower-is-better) weight", _profile.Weights.TimeToFirstToken);
        double quality = PromptWeight("Quality weight", _profile.Weights.Quality);
        double tool = PromptWeight("Tool-use success weight", _profile.Weights.ToolSuccess);

        double sum = tg + pp + ttft + quality + tool;
        if (Math.Abs(sum - 1.0) > 0.01)
        {
            AnsiConsole.MarkupLine($"[yellow]Weights sum to {sum:F2} — normalizing.[/]");
            tg /= sum; pp /= sum; ttft /= sum; quality /= sum; tool /= sum;
        }

        _profile = _profile with { Weights = new ScoringWeights(tg, pp, ttft, quality, tool) };
    }

    private static double PromptWeight(string label, double current)
        => AnsiConsole.Prompt(
            new TextPrompt<double>($"[grey]{label}[/] (0.0–1.0):")
                .DefaultValue(current)
                .Validate(v => v >= 0 && v <= 1
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Must be 0.0–1.0")));

    private void SetServerPath()
    {
        _serverExecutable = AnsiConsole.Prompt(
            new TextPrompt<string>("Path to [cyan]llama-server[/] executable:")
                .DefaultValue(_serverExecutable));
        AnsiConsole.MarkupLine($"[green]Server:[/] {_serverExecutable}");
    }

    private async Task DetectHardwareAsync(CancellationToken ct)
    {
        _hardware = await AnsiConsole.Status()
            .StartAsync("Detecting hardware...", async ctx =>
            {
                ctx.Spinner(Spinner.Known.Dots);
                return await hwService.DetectAsync();
            });

        BenchmarkDisplay.RenderHardwareInfo(_hardware);
    }

    private async Task RunOptimizationAsync(CancellationToken ct)
    {
        if (!ValidateReadyToRun()) return;

        _hardware ??= await AnsiConsole.Status()
            .StartAsync("Detecting hardware...", _ => hwService.DetectAsync());

        BenchmarkDisplay.RenderHardwareInfo(_hardware);
        AnsiConsole.WriteLine();

        // Build initial settings from hardware detection
        var initialSettings = OptimizerService.BuildInitialSettings(_modelPath, _profile, _hardware);

        // Show and allow override
        AnsiConsole.Write(new Rule("[bold]Initial Settings (estimated)[/]").RuleStyle("yellow"));
        SettingsEditor.RenderSettings(initialSettings);
        AnsiConsole.WriteLine();

        if (AnsiConsole.Confirm("Override any settings before starting?", false))
            initialSettings = SettingsEditor.Edit(initialSettings, "Starting Settings");

        AnsiConsole.WriteLine();
        int maxIter = AnsiConsole.Prompt(
            new TextPrompt<int>("Max optimization iterations:")
                .DefaultValue(15)
                .Validate(v => v >= 2 && v <= 50
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Must be 2–50")));

        var opts = new OptimizationOptions(MaxIterations: maxIter);
        var session = new OptimizationSession
        {
            ModelPath = _modelPath,
            Profile = _profile.Type,
            Hardware = _hardware,
        };

        AnsiConsole.Clear();
        AnsiConsole.Write(new Rule("[bold cyan]Running Optimization[/]").RuleStyle("cyan"));
        AnsiConsole.MarkupLine($"[grey]Model: {session.ModelName}  Profile: {_profile.Name}  Max iterations: {maxIter}[/]");
        AnsiConsole.MarkupLine("[grey]Press Ctrl+C to stop early and keep the best result found so far.[/]");
        AnsiConsole.WriteLine();

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        var iterations = optimizer.OptimizeAsync(
            _serverExecutable, _modelPath, initialSettings, _profile, opts, cts.Token);

        // Collect to session while displaying live
        var allIter = new List<OptimizationIteration>();
        await AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(new ProgressColumn[]
            {
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new SpinnerColumn(),
            })
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask($"[cyan]Optimizing {session.ModelName}[/]", maxValue: maxIter);

                await foreach (var iter in iterations.WithCancellation(cts.Token))
                {
                    allIter.Add(iter);
                    session.AddIteration(iter);
                    task.Increment(1);
                    task.Description = $"[cyan]iter {iter.Number}/{maxIter}[/] score=[green]{iter.Result.CompositeScore:F3}[/] " +
                        $"TG=[cyan]{iter.Result.GenerationRate:F1}t/s[/]";
                }
            });

        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderFinalResults(session, _serverExecutable);

        if (AnsiConsole.Confirm("\nExport results to file?", true))
        {
            string path = AnsiConsole.Ask<string>(
                "Export path:",
                $"results_{session.ModelName}_{session.Profile}.json");
            await persistence.ExportResultsAsync(session, path);
            AnsiConsole.MarkupLine($"[green]Exported to {path}[/]");
        }

        _manualSettings = session.BestSettings ?? _manualSettings;
    }

    private void EditSettingsManually()
    {
        _manualSettings = SettingsEditor.Edit(_manualSettings, "Manual Settings Editor");
        AnsiConsole.MarkupLine("[green]Settings updated.[/]");
    }

    private async Task TurboQuantMenuAsync(CancellationToken ct)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold magenta]TurboQuant[/]").RuleStyle("magenta"));
        AnsiConsole.MarkupLine("[grey]https://github.com/TheTom/llama-cpp-turboquant[/]");
        AnsiConsole.WriteLine();

        bool available = turboQuant.IsAvailable();
        if (!available)
        {
            AnsiConsole.MarkupLine("[yellow]turboquant not found in PATH.[/]");
            string? customPath = AnsiConsole.Prompt(
                new TextPrompt<string>("Path to turboquant executable (empty to skip):")
                    .AllowEmpty());
            if (string.IsNullOrEmpty(customPath) || !turboQuant.IsAvailable(customPath))
            {
                AnsiConsole.MarkupLine("[red]turboquant not available. Install from the GitHub repo.[/]");
                AnsiConsole.WriteLine("Press any key...");
                Console.ReadKey(intercept: true);
                return;
            }
        }

        var choice = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("TurboQuant Options:")
                .AddChoices(
                    "Quantize a model",
                    "Quantize current model and compare",
                    "Back"));

        if (choice == "Back") return;

        string inputModel = choice.Contains("current") && !string.IsNullOrEmpty(_modelPath)
            ? _modelPath
            : AnsiConsole.Ask<string>("Path to model to quantize:", _modelPath);

        if (!File.Exists(inputModel))
        {
            AnsiConsole.MarkupLine("[red]File not found.[/]");
            return;
        }

        var quantType = AnsiConsole.Prompt(
            new SelectionPrompt<QuantizationType>()
                .Title("Select quantization type:")
                .AddChoices(Enum.GetValues<QuantizationType>()));

        string outputDir = AnsiConsole.Ask<string>(
            "Output directory:",
            Path.GetDirectoryName(inputModel) ?? ".");

        var opts = new TurboQuantOptions
        {
            InputModelPath = inputModel,
            OutputDirectory = outputDir,
            QuantType = quantType,
            UseImportance = AnsiConsole.Confirm("Use importance-matrix calibration?", true),
            CalibrationSamples = AnsiConsole.Ask<int>("Calibration samples:", 512),
        };

        AnsiConsole.WriteLine();
        var result = await AnsiConsole.Progress()
            .Columns(new ProgressColumn[] { new TaskDescriptionColumn(), new SpinnerColumn() })
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[magenta]Quantizing...[/]");
                return await turboQuant.QuantizeAsync(opts,
                    new Progress<string>(msg => task.Description = $"[magenta]{msg.Truncate(60)}[/]"),
                    ct);
            });

        AnsiConsole.WriteLine();
        if (result.Success)
        {
            AnsiConsole.MarkupLine($"[green]Quantization complete![/]");
            AnsiConsole.MarkupLine($"Output: [cyan]{result.OutputPath}[/]");
            AnsiConsole.MarkupLine($"Size: [cyan]{result.OriginalSizeMb:N0} MB → {result.QuantizedSizeMb:N0} MB[/] " +
                $"([green]{(1 - result.CompressionRatio) * 100:F1}% smaller[/])");
            AnsiConsole.MarkupLine($"Duration: [grey]{result.Duration:mm\\:ss}[/]");

            if (choice.Contains("compare") && AnsiConsole.Confirm("Benchmark quantized model now?", true))
            {
                _modelPath = result.OutputPath;
                AnsiConsole.MarkupLine($"[green]Model path updated to quantized model.[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine($"[red]Quantization failed:[/] {result.ErrorMessage}");
        }

        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
    }

    private async Task ViewSessionsAsync(CancellationToken ct)
    {
        var sessions = persistence.ListSessions();

        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderSessionList(sessions);

        if (sessions.Count == 0)
        {
            AnsiConsole.WriteLine("Press any key...");
            Console.ReadKey(intercept: true);
            return;
        }

        if (!AnsiConsole.Confirm("Load a session?", false)) return;

        int idx = AnsiConsole.Prompt(
            new TextPrompt<int>("Session number to load:")
                .Validate(v => v >= 1 && v <= sessions.Count
                    ? ValidationResult.Success()
                    : ValidationResult.Error($"Must be 1–{sessions.Count}")));

        var session = await persistence.LoadAsync(sessions[idx - 1]);
        if (session is null)
        {
            AnsiConsole.MarkupLine("[red]Failed to load session.[/]");
            return;
        }

        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderFinalResults(session, _serverExecutable);

        if (session.BestSettings is not null &&
            AnsiConsole.Confirm("Apply best settings from this session?", true))
        {
            _manualSettings = session.BestSettings;
            _modelPath = session.ModelPath;
            AnsiConsole.MarkupLine("[green]Settings applied.[/]");
        }

        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
    }

    private async Task HistoricalComparisonAsync(CancellationToken ct)
    {
        AnsiConsole.WriteLine();

        var sessions = await AnsiConsole.Status()
            .StartAsync("Loading all sessions...", _ => persistence.LoadAllAsync());

        if (sessions.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No completed sessions found in the[/] [cyan]sessions/[/] [yellow]folder.[/]");
            AnsiConsole.WriteLine("Run some optimizations first, then come back here.");
            AnsiConsole.WriteLine("Press any key...");
            Console.ReadKey(intercept: true);
            return;
        }

        // Optionally filter to a specific model
        string? filterModel = null;
        var modelNames = sessions.Select(s => s.ModelName).Distinct().OrderBy(n => n).ToList();
        if (modelNames.Count > 1)
        {
            string choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("Filter by model? (or show all):")
                    .AddChoices(["Show all models", .. modelNames]));

            if (choice != "Show all models")
                filterModel = choice;
        }

        var displayed = filterModel is null
            ? sessions
            : sessions.Where(s => s.ModelName == filterModel).ToList();

        BenchmarkDisplay.RenderHistoricalComparison(displayed, filterModel);
        AnsiConsole.WriteLine();

        // Export options
        var exportChoice = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Export?")
                .AddChoices(
                    "No — just view",
                    "Export as CSV",
                    "Export as JSON"));

        if (exportChoice.Contains("CSV"))
        {
            string path = AnsiConsole.Ask<string>(
                "CSV output path:", "benchmark-comparison.csv");
            await persistence.ExportComparisonCsvAsync(displayed, path);
            AnsiConsole.MarkupLine($"[green]Exported to {path}[/]");
        }
        else if (exportChoice.Contains("JSON"))
        {
            // Re-use the existing per-session export in a loop
            string dir = AnsiConsole.Ask<string>(
                "Output directory:", "results");
            Directory.CreateDirectory(dir);
            foreach (var s in displayed)
            {
                string outPath = Path.Combine(dir, $"{s.ModelName}_{s.Profile}_{s.StartedAt:yyyyMMdd_HHmm}.json");
                await persistence.ExportResultsAsync(s, outPath);
            }
            AnsiConsole.MarkupLine($"[green]Exported {displayed.Count} files to {dir}/[/]");
        }

        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private bool ValidateReadyToRun()
    {
        var errors = new List<string>();
        if (string.IsNullOrEmpty(_modelPath) || !File.Exists(_modelPath))
            errors.Add("No valid model selected (menu option 1)");
        if (string.IsNullOrEmpty(_serverExecutable))
            errors.Add("llama-server path not set (menu option 3)");

        if (errors.Count == 0) return true;

        AnsiConsole.MarkupLine("[red]Cannot start — please fix:[/]");
        foreach (var e in errors)
            AnsiConsole.MarkupLine($"  [yellow]• {e}[/]");
        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
        return false;
    }

    private static void ShowBanner()
    {
        AnsiConsole.Clear();
        AnsiConsole.Write(
            new FigletText("LLM AutoSizer")
                .LeftJustified()
                .Color(Color.Cyan1));
        AnsiConsole.MarkupLine("[grey]Auto-tune llama.cpp settings for optimal local LLM performance[/]");
        AnsiConsole.MarkupLine("[grey]TurboQuant: https://github.com/TheTom/llama-cpp-turboquant[/]");
        AnsiConsole.WriteLine();
    }

    // -------------------------------------------------------------------------
    // Persistence of app config (paths, last profile)
    // -------------------------------------------------------------------------

    private void LoadConfig()
    {
        try
        {
            if (!File.Exists(ConfigFile)) return;
            var json = File.ReadAllText(ConfigFile);
            var doc = System.Text.Json.JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("serverExecutable", out var se)) _serverExecutable = se.GetString() ?? _serverExecutable;
            if (root.TryGetProperty("modelFolder", out var mf)) _modelFolder = mf.GetString() ?? "";
            if (root.TryGetProperty("modelPath", out var mp)) _modelPath = mp.GetString() ?? "";
            if (root.TryGetProperty("profile", out var pf))
            {
                _profile = pf.GetString() == "Agentic"
                    ? OptimizationProfile.Agentic()
                    : OptimizationProfile.Chat();
            }
        }
        catch { /* ignore corrupt config */ }
    }

    private void SaveConfig()
    {
        try
        {
            var json = System.Text.Json.JsonSerializer.Serialize(new
            {
                serverExecutable = _serverExecutable,
                modelFolder = _modelFolder,
                modelPath = _modelPath,
                profile = _profile.Name,
            }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(ConfigFile, json);
        }
        catch { /* non-critical */ }
    }
}

// Simple string extension to avoid pulling in extra deps
file static class StringExtensions
{
    public static string Truncate(this string s, int maxLen)
        => s.Length <= maxLen ? s : s[..maxLen] + "…";
}
