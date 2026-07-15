using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

public class MainMenu(
    HardwareDetectionService hwService,
    OptimizerService optimizer,
    LlamaServerService llamaServer,
    RecommendationService recommender,
    SessionPersistenceService persistence,
    ProfileLibraryService profileLibrary,
    ProfileMenu profileMenu,
    AppSettingsService appSettings,
    AppSettingsMenu settingsMenu)
{
    // -------------------------------------------------------------------------
    // Configuration (loaded once, persisted to disk)
    // -------------------------------------------------------------------------

    private string _modelPath = "";
    private OptimizationProfile _profile = OptimizationProfile.Chat();
    private LlamaSettings _manualSettings = new();
    private HardwareInfo? _hardware;

    private const string ConfigFile = "autosizer-config.json";
    private static readonly string[] SpinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public async Task RunAsync(CancellationToken ct = default)
    {
        LoadConfig();
        ShowBanner();

        while (!ct.IsCancellationRequested)
        {
            ShowBanner();

            string? choice = MenuHelper.Select(
                "[bold yellow]llama.cpp Auto-Sizer[/] — Main Menu",
                [
                    "1. Choose Profile  (Chat / Agentic)",
                    "2. Detect Hardware",
                    "3. Run Auto-Optimization",
                    "4. Edit llama.cpp Settings Manually",
                    "5. Named Profiles  (run / manage)",
                    "6. View / Load Sessions",
                    "7. Settings  (model folder, server path, storage, TurboQuant)",
                    "H. Historical Benchmark Comparison",
                    "0. Exit",
                ],
                ct: ct);

            // Escape on the main menu = exit
            if (choice is null || choice[0] == '0') return;

            try
            {
                switch (choice[0])
                {
                    case '1': ChooseProfile(); break;
                    case '2': await DetectHardwareAsync(ct); break;
                    case '3': await RunOptimizationAsync(ct); break;
                    case '4': EditSettingsManually(); break;
                    case '5': await profileMenu.RunAsync(appSettings.EffectiveServerExecutable, ct); break;
                    case '6': await ViewSessionsAsync(ct); break;
                    case '7':
                        string? newModelPath = await settingsMenu.RunAsync(_manualSettings, _modelPath, ct);
                        if (newModelPath is not null) _modelPath = newModelPath;
                        break;
                    case 'H': case 'h': await HistoricalComparisonAsync(ct); break;
                }
                SaveConfig();
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                AnsiConsole.WriteLine("Press any key to continue...");
                Console.ReadKey(intercept: true);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Menu handlers
    // -------------------------------------------------------------------------

    // Scans the model folder configured in Settings and lets the user pick a .gguf file.
    // Returns false if the user backs out without selecting anything (e.g. no folder
    // configured and no manual path given). Called from RunOptimizationAsync when the
    // user doesn't start from an existing named profile.
    private async Task<bool> SelectModelFromConfiguredFolderAsync(CancellationToken ct)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold]Select Model[/]").RuleStyle("cyan"));

        string folder = appSettings.Current.ModelFolder;
        if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
        {
            AnsiConsole.MarkupLine(
                "[yellow]No valid model folder configured.[/] Set one in [cyan]Settings[/] (menu option 7), " +
                "or enter a folder now.");
            string manualFolder = AnsiConsole.Prompt(
                new TextPrompt<string>(
                    "[grey]Folder to scan for[/] [cyan].gguf[/] [grey]files (empty to enter a file path directly):[/]")
                    .AllowEmpty());

            if (string.IsNullOrWhiteSpace(manualFolder))
            {
                string manualPath = AnsiConsole.Prompt(
                    new TextPrompt<string>("Full path to .gguf file:")
                        .Validate(p => File.Exists(p)
                            ? ValidationResult.Success()
                            : ValidationResult.Error("File not found")));
                SetModel(manualPath);
                return true;
            }

            if (!Directory.Exists(manualFolder))
            {
                AnsiConsole.MarkupLine("[red]Folder not found.[/]");
                AnsiConsole.WriteLine("Press any key...");
                Console.ReadKey(intercept: true);
                return false;
            }

            folder = manualFolder;
            if (AnsiConsole.Confirm("Save this as the default model folder in Settings?", true))
            {
                appSettings.Current.ModelFolder = folder;
                appSettings.Save();
            }
        }

        bool recursive = AnsiConsole.Confirm($"Scan [cyan]{Markup.Escape(folder)}[/] — include sub-folders?", true);

        // ── Discover models ─────────────────────────────────────────────────
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
            return true;
        }

        // ── Display discovered models ────────────────────────────────────────
        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderModelList(models);
        AnsiConsole.WriteLine();

        // Build choice list — escape markup-special chars; use [[ ]] for literal brackets
        var choices = models
            .Select((m, i) =>
                $"{i + 1,3}. {Markup.Escape(Path.GetFileName(m.Path))}  [[{m.SizeMb:N0} MB]]")
            .Append("── Enter path manually ──")
            .ToList();

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
            return true;
        }

        // Parse index from "  3. filename.gguf  [4,321 MB]"
        int dot = selected.IndexOf('.');
        if (dot > 0 && int.TryParse(selected[..dot].Trim(), out int idx) && idx >= 1 && idx <= models.Count)
        {
            SetModel(models[idx - 1].Path);
            return true;
        }
        return false;
    }

    private void SetModel(string path)
    {
        _modelPath = path;
        _manualSettings = new LlamaSettings();
        AnsiConsole.MarkupLine($"[green]Model:[/] {Markup.Escape(Path.GetFileName(_modelPath))}  " +
            $"[grey]({new FileInfo(_modelPath).Length / (1024 * 1024):N0} MB)[/]");
        AnsiConsole.WriteLine();
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

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Press any key to return to menu...[/]");
        Console.ReadKey(intercept: true);
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

    private async Task DetectHardwareAsync(CancellationToken ct)
    {
        _hardware = await AnsiConsole.Status()
            .StartAsync("Detecting hardware...", async ctx =>
            {
                ctx.Spinner(Spinner.Known.Dots);
                return await hwService.DetectAsync();
            });

        BenchmarkDisplay.RenderHardwareInfo(_hardware);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Press any key to return to menu...[/]");
        Console.ReadKey(intercept: true);
    }

    private async Task RunOptimizationAsync(CancellationToken ct)
    {
        if (!ValidateReadyToRun()) return;

        // ── Step 1: start from an existing named profile (model + settings both come from
        // it), or fall back to scanning the model folder configured in Settings ──────────
        var savedProfiles = await profileLibrary.ListProfilesAsync();
        SavedProfile? sourceProfile = null;

        if (savedProfiles.Count > 0 &&
            AnsiConsole.Confirm("Start from an existing named profile?", false))
        {
            var profileChoices = savedProfiles
                .Select(p =>
                {
                    string score = p.BenchmarkScore.HasValue ? $"  score={p.BenchmarkScore:F3}" : "";
                    string tg    = p.BenchmarkTgRate.HasValue ? $"  TG={p.BenchmarkTgRate:F0}t/s" : "";
                    return $"[cyan]{Markup.Escape(p.Name)}[/]{score}{tg}  [grey]({Markup.Escape(p.ModelName)})[/]";
                })
                .ToList();

            string? picked = MenuHelper.Select("Select starting profile:", profileChoices);
            if (picked is not null)
            {
                int idx = profileChoices.IndexOf(picked);
                sourceProfile = savedProfiles[idx];
            }
        }

        if (sourceProfile is not null)
        {
            if (!File.Exists(sourceProfile.ModelPath))
            {
                AnsiConsole.MarkupLine(
                    $"[red]Model file for this profile no longer exists:[/] {Markup.Escape(sourceProfile.ModelPath)}");
                AnsiConsole.WriteLine("Press any key...");
                Console.ReadKey(intercept: true);
                return;
            }
            _modelPath = sourceProfile.ModelPath;
            AnsiConsole.MarkupLine($"[grey]Model (from profile):[/] {Markup.Escape(Path.GetFileName(_modelPath))}");
        }
        else if (!await SelectModelFromConfiguredFolderAsync(ct))
        {
            return;
        }

        _hardware ??= await AnsiConsole.Status()
            .StartAsync("Detecting hardware...", _ => hwService.DetectAsync());

        BenchmarkDisplay.RenderHardwareInfo(_hardware);
        AnsiConsole.WriteLine();

        LlamaSettings initialSettings;
        if (sourceProfile is not null)
        {
            initialSettings = sourceProfile.Settings.Clone();
            AnsiConsole.MarkupLine($"[grey]Starting from profile:[/] [cyan]{Markup.Escape(sourceProfile.Name)}[/]");
            AnsiConsole.WriteLine();
        }
        else
        {
            initialSettings = OptimizerService.BuildInitialSettings(_modelPath, _profile, _hardware);

            // Hardware presets never set a KV cache type (they leave it at the f16 default).
            // Carry forward whatever cache type was last chosen via "Edit llama.cpp Settings
            // Manually" or the TurboQuant cache-type screen — otherwise those choices are
            // silently discarded the moment you start a fresh (non-profile) optimization run.
            if (_manualSettings.CacheTypeK is not null) initialSettings.CacheTypeK = _manualSettings.CacheTypeK;
            if (_manualSettings.CacheTypeV is not null) initialSettings.CacheTypeV = _manualSettings.CacheTypeV;
        }

        // Show and allow override
        AnsiConsole.Write(new Rule("[bold]Initial Settings[/]").RuleStyle("yellow"));
        SettingsEditor.RenderSettings(initialSettings, LlamaSettings.IsMoeModel(_modelPath));
        AnsiConsole.WriteLine();

        if (AnsiConsole.Confirm("Override any settings before starting?", false))
            initialSettings = SettingsEditor.Edit(initialSettings, "Starting Settings", _modelPath, appSettings.TurboQuantAvailable);

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Panel(
            "[yellow]WARNING:[/] Optimization runs the model at full load repeatedly.\n" +
            "Your system [bold]may become slow or temporarily unresponsive[/] during this process,\n" +
            "especially with large models or limited VRAM. Save any open work before proceeding.\n" +
            "Press [bold]Ctrl+C[/] at any time to stop early and keep the best result found so far.")
        {
            Border = BoxBorder.Rounded,
            BorderStyle = new Style(Color.Yellow),
            Padding = new Padding(1, 0),
        });
        AnsiConsole.WriteLine();

        if (!AnsiConsole.Confirm("Proceed with optimization?", true))
            return;

        AnsiConsole.WriteLine();
        int maxIter = AnsiConsole.Prompt(
            new TextPrompt<int>("Max optimization iterations:")
                .DefaultValue(20)
                .Validate(v => v >= 2 && v <= 50
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Must be 2–50")));

        double targetTgSpeed = AnsiConsole.Prompt(
            new TextPrompt<double>(
                "[grey]Target generation speed (t/s) — once reached, further tuning favors quality over more speed:[/]")
                .DefaultValue(30.0)
                .Validate(v => v > 0
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Must be greater than 0")));

        string userGuidance = AnsiConsole.Prompt(
            new TextPrompt<string>(
                "[grey]Any guidance for the optimizer, e.g. \"prioritize low VRAM usage\" or[/]\n" +
                "[grey]\"favor large context over raw speed\" (empty to skip):[/]")
                .AllowEmpty());

        bool includeStressTest = AnsiConsole.Confirm(
            "[grey]Include a long-form repetition-loop stress test in each benchmark? " +
            "(slower iterations, more likely to catch degenerate/looping output)[/]", false);

        var opts = new OptimizationOptions(MaxIterations: maxIter, IncludeRepetitionStressTest: includeStressTest);

        // Single session shared with the optimizer — CompletionReason and iteration history
        // are written directly to this object, so the final display is always correct.
        var session = new OptimizationSession
        {
            ModelPath = _modelPath,
            Profile = _profile.Type,
            Hardware = _hardware,   // optimizer will refresh this with a live DetectAsync
            TargetTgSpeed = targetTgSpeed,
            UserGuidance = string.IsNullOrWhiteSpace(userGuidance) ? null : userGuidance.Trim(),
        };

        AnsiConsole.Clear();
        AnsiConsole.Write(new Rule("[bold cyan]Running Optimization[/]").RuleStyle("cyan"));
        AnsiConsole.MarkupLine($"[grey]Model: {Markup.Escape(session.ModelName)}  Profile: {Markup.Escape(_profile.Name)}  Max iterations: {maxIter}  Target: {targetTgSpeed:F0}t/s[/]");
        if (session.UserGuidance is not null)
            AnsiConsole.MarkupLine($"[grey]Guidance:[/] [italic]{Markup.Escape(session.UserGuidance)}[/]");
        AnsiConsole.MarkupLine("[yellow]⚠ System may become slow or unresponsive during benchmarking — this is normal.[/]");
        AnsiConsole.MarkupLine("[grey]Press Ctrl+C to stop early and keep the best result found so far.[/]");
        AnsiConsole.WriteLine();

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        // Server log: in full mode, lines print as normal scrolling console output (so the
        // user can scroll the terminal back through full history). In truncated mode, only
        // the last few lines are shown, redrawn in place inside the footer. Either way the
        // footer (status/spinner + progress bar) is redrawn in place below via relative ANSI
        // cursor movement — the same technique MenuHelper uses for its own redraws. Press L
        // to toggle between the two.
        var logQueue = new System.Collections.Concurrent.ConcurrentQueue<string>();
        llamaServer.SetLogHandler(logQueue.Enqueue);
        // Recommender activity (which model was asked — Claude / local LLM / heuristic —
        // and what each determined) lands in the same rolling log, prefixed for scanning.
        recommender.SetActivityLog(line => logQueue.Enqueue($"[advisor] {line}"));

        // Updated from OptimizerService/BenchmarkService as they move through phases (hardware
        // detection, server start, which benchmark prompt/case is running, LLM recommendation
        // calls, etc.) so the waiting spinner below always says what's actually happening
        // instead of a bare "Working…".
        string currentPhase = "";

        // Optimizer writes iterations into session directly; we only read them here for the UI.
        var iterations = optimizer.OptimizeAsync(
            appSettings.EffectiveServerExecutable, _modelPath, initialSettings, _profile, session, opts,
            onPhase: p => currentPhase = p, cts.Token);

        AnsiConsole.Write(new Rule("[grey dim]llama-server & advisor log (scroll up for history) — L: toggle full/truncated[/]").RuleStyle("grey dim"));

        int footerLineCount = 0;
        int spinnerFrame = 0;
        int iterDone = 0;
        bool showFullLog = true;
        var allLogs = new List<string>();      // every log line ever received, never trimmed
        int printedCount = 0;                   // how many of allLogs have hit the scrolling console
        const int MaxTruncatedLines = 10;

        // Background key watcher: only reacts to L (toggle log mode); everything else is
        // ignored — Ctrl+C (handled separately via the cancellation token) remains the only
        // way to stop early.
        using var keyWatcherCts = new CancellationTokenSource();
        _ = Task.Run(() =>
        {
            while (!keyWatcherCts.IsCancellationRequested)
            {
                if (Console.KeyAvailable)
                {
                    if (Console.ReadKey(intercept: true).Key == ConsoleKey.L)
                        showFullLog = !showFullLog;
                }
                else
                {
                    Thread.Sleep(20);
                }
            }
        });

        void DrainLogs()
        {
            while (logQueue.TryDequeue(out var line))
            {
                if (!string.IsNullOrWhiteSpace(line))
                    allLogs.Add(line);
            }

            // Full mode: flush everything not yet printed — including lines that arrived
            // while truncated mode was active, so switching back to full catches up in full.
            if (showFullLog)
            {
                for (int i = printedCount; i < allLogs.Count; i++)
                {
                    // Advisor traffic stands out from llama-server noise
                    string color = allLogs[i].StartsWith("[advisor]") ? "cyan" : "grey";
                    AnsiConsole.MarkupLine($"[{color}]{Markup.Escape(allLogs[i])}[/]");
                }
                printedCount = allLogs.Count;
            }
        }

        List<string> WithTruncatedLog(IReadOnlyList<string> statusLines)
        {
            if (showFullLog || allLogs.Count == 0) return [.. statusLines];

            int width = Math.Max(40, Console.WindowWidth);
            int innerWidth = Math.Max(10, width - 4);

            string header = $" llama-server & advisor log (last {MaxTruncatedLines}) ";
            int dashCount = Math.Max(0, width - header.Length - 3);

            var lines = new List<string> { $"[white]╭─{header}{new string('─', dashCount)}╮[/]" };
            foreach (var raw in allLogs.Skip(Math.Max(0, allLogs.Count - MaxTruncatedLines)))
            {
                string clipped = raw.Length > innerWidth ? raw[..innerWidth] : raw;
                string padded = clipped.PadRight(innerWidth);
                string color = raw.StartsWith("[advisor]") ? "cyan" : "grey";
                lines.Add($"[white]│[/] [{color}]{Markup.Escape(padded)}[/] [white]│[/]");
            }
            lines.Add($"[white]╰{new string('─', width - 2)}╯[/]");

            lines.AddRange(statusLines);
            return lines;
        }

        void RedrawFooter(IReadOnlyList<string> statusLines)
        {
            if (footerLineCount > 0)
                Console.Write($"\x1b[{footerLineCount}A\r\x1b[0J");
            DrainLogs();
            var lines = WithTruncatedLog(statusLines);
            foreach (var line in lines)
                AnsiConsole.MarkupLine(line);
            footerLineCount = lines.Count;
        }

        // Footer redraw erases N lines by cursor-up, assuming 1 logical line == 1 physical
        // row. A status/reasoning string longer than the terminal width soft-wraps into
        // extra rows the erase math doesn't know about, leaving stacked duplicate garbage
        // on every redraw. Clip raw (pre-markup) text so that invariant always holds.
        string ClipToWidth(string text, int reserve)
        {
            int max = Math.Max(10, Console.WindowWidth - reserve);
            return text.Length > max ? text[..(max - 1)] + "…" : text;
        }

        string BuildProgressLine()
        {
            double pct = maxIter > 0 ? (double)iterDone / (maxIter + 1) * 100.0 : 0;
            int barWidth = 30;
            int filled = (int)(barWidth * pct / 100.0);
            string bar = $"[cyan]{new string('█', filled)}[/]{new string('░', barWidth - filled)}";
            return $"{bar}  {pct:F0}%  ({iterDone}/{maxIter + 1})";
        }

        // Last completed iteration's status — kept on screen (under the spinner) while the
        // next one is being prepared, instead of being replaced by a bare spinner line.
        string lastIterStatus = "";
        string lastIterSub = "";

        try
        {
            await using var enumerator = iterations.GetAsyncEnumerator(cts.Token);
            var moveNextTask = enumerator.MoveNextAsync().AsTask();

            while (true)
            {
                // While waiting for the next iteration (server start + benchmark can take a
                // while, especially the first/baseline one), keep the footer alive: spin,
                // flush any server log lines as they scroll in, and keep showing what the
                // last completed iteration found so that info doesn't just vanish.
                while (!moveNextTask.IsCompleted)
                {
                    string waitingLabel = currentPhase.Length > 0
                        ? Markup.Escape(ClipToWidth(currentPhase, 6))
                        : iterDone == 0 ? "Preparing llama-server…" : "Working…";
                    var waitingLines = new List<string>();
                    if (lastIterStatus.Length > 0) waitingLines.Add(lastIterStatus);
                    if (lastIterSub.Length > 0) waitingLines.Add($"  [grey]{lastIterSub}[/]");
                    waitingLines.Add($"[yellow]{SpinnerFrames[spinnerFrame++ % SpinnerFrames.Length]} {waitingLabel}[/]");
                    waitingLines.Add(BuildProgressLine());
                    RedrawFooter(waitingLines);
                    await Task.WhenAny(moveNextTask, Task.Delay(150, cts.Token));
                }

                if (!await moveNextTask) break;
                var iter = enumerator.Current;

                iterDone++;
                string bestMark = iter.IsBestSoFar ? "  [bold green]★ best[/]" : "";
                string toolRate = iter.Result.ToolCallEffectiveTgRate > 0
                    ? $"  Tool=[cyan]{iter.Result.ToolCallEffectiveTgRate:F1}t/s[/]" : "";
                string loopRate = iter.Result.AgentLoopEffectiveTgRate > 0
                    ? $"  Loop=[cyan]{iter.Result.AgentLoopEffectiveTgRate:F1}t/s[/]" : "";
                lastIterStatus = $"[grey]iter {iter.Number}/{maxIter}[/]  " +
                             $"score=[green]{iter.Result.CompositeScore:F3}[/]  " +
                             $"TG=[cyan]{iter.Result.GenerationRate:F1}t/s[/]  " +
                             $"PP=[cyan]{iter.Result.PromptProcessingRate:F0}t/s[/]" +
                             toolRate + loopRate +
                             bestMark;
                lastIterSub = iter.StatusMessage is not null
                    ? Markup.Escape(ClipToWidth(iter.StatusMessage, 4))
                    : "";

                var statusLines = new List<string> { lastIterStatus };
                if (!string.IsNullOrEmpty(lastIterSub)) statusLines.Add($"  [grey]{lastIterSub}[/]");
                statusLines.Add(BuildProgressLine());
                RedrawFooter(statusLines);

                moveNextTask = enumerator.MoveNextAsync().AsTask();
            }
        }
        finally
        {
            keyWatcherCts.Cancel();
        }

        DrainLogs();

        llamaServer.SetLogHandler(null);
        recommender.SetActivityLog(null);

        AnsiConsole.WriteLine();
        BenchmarkDisplay.RenderFinalResults(session, appSettings.EffectiveServerExecutable);

        if (AnsiConsole.Confirm("\nExport results to file?", true))
        {
            string path = AnsiConsole.Ask<string>(
                "Export path:",
                $"results_{session.ModelName}_{session.Profile}.json");
            await persistence.ExportResultsAsync(session, path);
            AnsiConsole.MarkupLine($"[green]Exported to {Markup.Escape(path)}[/]");
        }

        // Let the user pick which starred iteration to use (prompts only if multiple exist)
        var chosenIter = BenchmarkDisplay.SelectBestIteration(session);
        _manualSettings = chosenIter?.Settings ?? session.BestSettings ?? _manualSettings;

        if (chosenIter is not null &&
            AnsiConsole.Confirm("\nSave best settings as a named profile?", true))
        {
            await SaveProfileFromSessionAsync(session, chosenIter);
        }
    }

    private void EditSettingsManually()
    {
        _manualSettings = SettingsEditor.Edit(_manualSettings, "Manual Settings Editor", _modelPath, appSettings.TurboQuantAvailable);
        AnsiConsole.MarkupLine("[green]Settings updated.[/]");

        if (string.IsNullOrEmpty(_modelPath) || !File.Exists(_modelPath))
        {
            AnsiConsole.MarkupLine("[grey](Select a model first to save as a named profile.)[/]");
        }
        else if (AnsiConsole.Confirm("Save these settings as a named profile?", false))
        {
            string name = AnsiConsole.Prompt(
                new TextPrompt<string>("Profile name:")
                    .Validate(n => !string.IsNullOrWhiteSpace(n)
                        ? ValidationResult.Success()
                        : ValidationResult.Error("Name cannot be empty")));

            name = name.Trim();

            var existing = profileLibrary.FindByNameAsync(name).GetAwaiter().GetResult();
            bool doSave = true;
            if (existing is not null)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] A profile named [cyan]{Markup.Escape(name)}[/] already exists.");
                if (!AnsiConsole.Confirm("Overwrite it?", false))
                    doSave = false;
                else
                    profileLibrary.Delete(existing);
            }

            if (doSave)
            {
                var profile = SavedProfile.FromSettings(name, _modelPath, _manualSettings);
                profileLibrary.SaveAsync(profile).GetAwaiter().GetResult();
                AnsiConsole.MarkupLine($"[green]Profile saved:[/] {Markup.Escape(profile.Name)}");
            }
        }

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Press any key to return to menu...[/]");
        Console.ReadKey(intercept: true);
    }

    private async Task SaveProfileFromSessionAsync(
        OptimizationSession session,
        OptimizationIteration? chosenIteration = null)
    {
        string defaultName = $"{session.ModelName} — {session.Profile}";
        string name = AnsiConsole.Prompt(
            new TextPrompt<string>("Profile name:")
                .DefaultValue(defaultName)
                .Validate(n => !string.IsNullOrWhiteSpace(n)
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Name cannot be empty")));

        name = name.Trim();

        // Warn if a profile with this name already exists
        var existing = await profileLibrary.FindByNameAsync(name);
        if (existing is not null)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] A profile named [cyan]{Markup.Escape(name)}[/] already exists.");
            if (existing.BenchmarkScore.HasValue)
                AnsiConsole.MarkupLine($"  [grey]Existing score: {existing.BenchmarkScore:F3}  TG={existing.BenchmarkTgRate:F0}t/s[/]");
            if (!AnsiConsole.Confirm("Overwrite it?", false))
            {
                AnsiConsole.MarkupLine("[grey]Save cancelled.[/]");
                return;
            }
            profileLibrary.Delete(existing);
        }

        string? notes = AnsiConsole.Prompt(
            new TextPrompt<string>("Optional notes (empty to skip):")
                .AllowEmpty());

        var profile = SavedProfile.FromSession(session, name, chosenIteration);
        if (!string.IsNullOrWhiteSpace(notes)) profile.Notes = notes;

        await profileLibrary.SaveAsync(profile);
        AnsiConsole.MarkupLine($"[green]Profile saved:[/] {Markup.Escape(profile.Name)}");
        AnsiConsole.MarkupLine("[grey]Access it any time from menu option 5.[/]");
    }

    // TurboQuant configuration (cache types, server executable, quantization) now lives in
    // Settings — see AppSettingsMenu.RunAsync / TurboQuantSubmenuAsync.

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
        BenchmarkDisplay.RenderFinalResults(session, appSettings.EffectiveServerExecutable);

        if (session.BestSettings is not null &&
            AnsiConsole.Confirm("Apply best settings from this session?", true))
        {
            var chosen = BenchmarkDisplay.SelectBestIteration(session);
            _manualSettings = chosen?.Settings ?? session.BestSettings;
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
                    .AddChoices(["Show all models", .. modelNames.Select(Markup.Escape)]));

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
            AnsiConsole.MarkupLine($"[green]Exported to {Markup.Escape(path)}[/]");
        }
        else if (exportChoice.Contains("JSON"))
        {
            string dir = AnsiConsole.Ask<string>(
                "Output directory:", "results");
            Directory.CreateDirectory(dir);
            foreach (var s in displayed)
            {
                string outPath = Path.Combine(dir, $"{s.ModelName}_{s.Profile}_{s.StartedAt:yyyyMMdd_HHmm}.json");
                await persistence.ExportResultsAsync(s, outPath);
            }
            AnsiConsole.MarkupLine($"[green]Exported {displayed.Count} files to {Markup.Escape(dir)}/[/]");
        }

        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private bool ValidateReadyToRun()
    {
        if (!string.IsNullOrEmpty(appSettings.EffectiveServerExecutable)) return true;

        AnsiConsole.MarkupLine("[red]Cannot start — please fix:[/]");
        AnsiConsole.MarkupLine("  [yellow]• llama-server path not set (Settings menu, option 7)[/]");
        AnsiConsole.WriteLine("Press any key...");
        Console.ReadKey(intercept: true);
        return false;
    }

    private void ShowBanner()
    {
        AnsiConsole.Clear();
        AnsiConsole.Write(
            new FigletText("LLM AutoSizer")
                .LeftJustified()
                .Color(Color.Cyan1));
        AnsiConsole.MarkupLine("[grey]Auto-tune llama.cpp settings for optimal local LLM performance[/]");
        AnsiConsole.MarkupLine("[grey]TurboQuant: https://github.com/TheTom/llama-cpp-turboquant[/]");

        // Show current session state if anything is configured
        bool hasModel = !string.IsNullOrEmpty(_modelPath) && File.Exists(_modelPath);
        bool hasServer = !string.IsNullOrEmpty(appSettings.EffectiveServerExecutable);

        if (hasModel || hasServer || _manualSettings.GpuLayers != new LlamaSettings().GpuLayers)
        {
            AnsiConsole.WriteLine();
            RenderStatusPanel(hasModel, hasServer);
        }

        AnsiConsole.WriteLine();
    }

    private void RenderStatusPanel(bool hasModel, bool hasServer)
    {
        var grid = new Grid().AddColumn().AddColumn();

        // ── Model / server row ────────────────────────────────────────────────
        string modelCell = hasModel
            ? $"[bold deepskyblue1]{Markup.Escape(Path.GetFileNameWithoutExtension(_modelPath))}[/]"
            : "[grey]no model selected[/]";

        string serverCell = hasServer
            ? $"[deepskyblue1]{Markup.Escape(Path.GetFileName(appSettings.EffectiveServerExecutable))}[/]"
            : "[grey]not set[/]";

        // ── Settings summary row ──────────────────────────────────────────────
        var s = _manualSettings;
        string ngl    = s.GpuLayers == -1 ? "[magenta1]all[/]" : $"[magenta1]{s.GpuLayers}[/]";
        string ctx    = $"[deepskyblue1]{s.ContextSize:N0}[/]";
        string fa     = s.FlashAttention ? "[chartreuse1]on[/]"  : "[grey]off[/]";
        string cacheK = s.CacheTypeK is null ? "[grey]f16[/]" : $"[magenta1]{s.CacheTypeK}[/]";
        string cacheV = s.CacheTypeV is null ? "[grey]f16[/]" : $"[magenta1]{s.CacheTypeV}[/]";
        string prof   = $"[yellow]{_profile.Name}[/]";

        var inner = new Table()
            .Border(TableBorder.None)
            .HideHeaders()
            .AddColumn(new TableColumn("").Width(14))
            .AddColumn("");

        inner.AddRow("[grey]model[/]",    modelCell);
        inner.AddRow("[grey]server[/]",   serverCell);
        inner.AddRow("[grey]profile[/]",  prof);
        inner.AddRow("[grey]ngl[/]",      ngl);
        inner.AddRow("[grey]ctx[/]",      ctx);
        inner.AddRow("[grey]flash-attn[/]", fa);
        inner.AddRow("[grey]kv-k / kv-v[/]", $"{cacheK} [grey]/[/] {cacheV}");

        if (!string.IsNullOrWhiteSpace(s.ExtraArgs))
            inner.AddRow("[grey]extra args[/]", $"[dim]{Markup.Escape(s.ExtraArgs)}[/]");

        AnsiConsole.Write(new Panel(inner)
        {
            Header = new PanelHeader("[bold magenta1] ▸ current config [/]"),
            Border = BoxBorder.Heavy,
            BorderStyle = new Style(Color.DeepSkyBlue1),
            Padding = new Padding(1, 0),
        });
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

            // One-time migration: this file used to also hold serverExecutable/modelFolder,
            // which now live in AppSettingsService's central config (see Settings menu).
            bool migrated = false;
            if (string.IsNullOrWhiteSpace(appSettings.Current.ModelFolder) &&
                root.TryGetProperty("modelFolder", out var mf) && mf.GetString() is { Length: > 0 } mfVal)
            {
                appSettings.Current.ModelFolder = mfVal;
                migrated = true;
            }
            if (appSettings.Current.ServerExecutable == "llama-server" &&
                root.TryGetProperty("serverExecutable", out var se) &&
                se.GetString() is { Length: > 0 } seVal && seVal != "llama-server")
            {
                appSettings.Current.ServerExecutable = seVal;
                migrated = true;
            }
            if (migrated) appSettings.Save();

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
