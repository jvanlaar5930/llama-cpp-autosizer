using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using LlamaCppAutosizer.UI;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace LlamaCppAutosizer.UI;

public class ProfileMenu(
    ProfileLibraryService library,
    LlamaServerService server,
    AppSettingsService appSettings)
{
    // -------------------------------------------------------------------------
    // Entry point — called from MainMenu
    // -------------------------------------------------------------------------

    public async Task RunAsync(string serverExecutable, CancellationToken ct = default)
    {
        while (!ct.IsCancellationRequested)
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[bold cyan]Named Profiles[/]").RuleStyle("cyan"));
            AnsiConsole.WriteLine();

            var profiles = await library.ListProfilesAsync();

            if (profiles.Count == 0)
            {
                AnsiConsole.MarkupLine("[grey]No saved profiles yet.[/]");
                AnsiConsole.MarkupLine("[grey]Run an optimization (option 3) or edit settings manually (option 4)[/]");
                AnsiConsole.MarkupLine("[grey]and choose 'Save as named profile' when prompted.[/]");
                AnsiConsole.WriteLine();
                AnsiConsole.WriteLine("Press any key to go back...");
                Console.ReadKey(intercept: true);
                return;
            }

            RenderProfileTable(profiles);
            AnsiConsole.WriteLine();

            string? choice = MenuHelper.Select(
                "[grey]Choose action[/]  [dim](Esc = back)[/]",
                [
                    "Run a profile",
                    "Edit profile settings",
                    "Rename a profile",
                    "Add notes to a profile",
                    "Delete a profile",
                ]);

            if (choice is null) return; // Escape

            switch (choice)
            {
                case "Run a profile":
                    await RunProfileAsync(profiles, serverExecutable, ct);
                    break;
                case "Edit profile settings":
                    await EditProfileAsync(profiles);
                    break;
                case "Rename a profile":
                    await RenameProfileAsync(profiles);
                    break;
                case "Add notes to a profile":
                    await AddNotesAsync(profiles);
                    break;
                case "Delete a profile":
                    await DeleteProfileAsync(profiles);
                    break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Run
    // -------------------------------------------------------------------------

    private async Task RunProfileAsync(
        List<SavedProfile> profiles, string serverExecutable, CancellationToken ct)
    {
        var profile = PickProfile(profiles, "Select profile to run:");
        if (profile is null) return;

        if (!ProfileLibraryService.IsModelAvailable(profile))
        {
            AnsiConsole.MarkupLine(
                $"[red]Model not found:[/] {Markup.Escape(profile.ModelPath)}");
            AnsiConsole.MarkupLine("[grey]Update the model path by editing the profile.[/]");
            AnsiConsole.WriteLine("Press any key...");
            Console.ReadKey(intercept: true);
            return;
        }

        AnsiConsole.Clear();
        AnsiConsole.Write(new Rule($"[bold green]Running:[/] [cyan]{Markup.Escape(profile.Name)}[/]").RuleStyle("green"));
        AnsiConsole.WriteLine();

        // Buffer logs during the startup spinner so they don't corrupt ANSI rendering.
        // On failure: flush to console. On success: move to the live display queue.
        var startupLogs = new System.Collections.Concurrent.ConcurrentQueue<string>();
        server.SetLogHandler(startupLogs.Enqueue);

        bool started = false;
        string? startError = null;
        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync($"Starting llama-server for [cyan]{Markup.Escape(profile.Name)}[/]...", async ctx =>
            {
                try
                {
                    await server.StartAsync(serverExecutable, profile.ModelPath, profile.Settings, 8080, ct);
                    started = true;
                }
                catch (Exception ex)
                {
                    startError = ex.Message;
                }
            });

        if (!started)
        {
            AnsiConsole.MarkupLine($"[red]Failed to start server:[/] {Markup.Escape(startError ?? "unknown error")}");
            AnsiConsole.WriteLine();
            if (!startupLogs.IsEmpty)
            {
                AnsiConsole.MarkupLine("[grey]llama-server output:[/]");
                while (startupLogs.TryDequeue(out var line))
                    Console.WriteLine(line);
                AnsiConsole.WriteLine();
            }
            AnsiConsole.WriteLine("Press any key...");
            Console.ReadKey(intercept: true);
            return;
        }

        await library.RecordRunAsync(profile);

        // --- Live display: settings + real-time stats + rolling log ---

        var logQueue = new System.Collections.Concurrent.ConcurrentQueue<string>();
        // Move any buffered startup logs into the display queue
        while (startupLogs.TryDequeue(out var l)) logQueue.Enqueue(l);
        server.SetLogHandler(logQueue.Enqueue);

        var stopCts = new CancellationTokenSource();
        bool showFullLog = false;
        bool showCommand = false;
        _ = Task.Run(() =>
        {
            while (!stopCts.IsCancellationRequested)
            {
                if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(intercept: true);
                    if (key.Key == ConsoleKey.L)
                    {
                        showFullLog = !showFullLog;
                        showCommand = false;
                    }
                    else if (key.Key == ConsoleKey.P)
                    {
                        showCommand = !showCommand;
                        showFullLog = false;
                    }
                    else if (key.Key == ConsoleKey.Escape)
                        stopCts.Cancel();
                }
                else
                {
                    Thread.Sleep(20);
                }
            }
        });

        int? pid = server.ServerProcessId;
        var prevCpuTime = TimeSpan.Zero;
        var prevCpuSample = DateTime.UtcNow;
        (int GpuPct, long UsedMb, long TotalMb)? gpuStats = null;
        int gpuPollTick = 0;
        var logLines = new Queue<string>();
        const int MaxLogLines = 500;

        LlamaMetrics? prevMetrics = null;
        var prevMetricsSample = DateTime.UtcNow;
        double? liveTokPerSec = null;

        await AnsiConsole.Live(BuildRunningPanel(profile, serverExecutable, null, null, [], false, false, null))
            .AutoClear(false)
            .Overflow(VerticalOverflow.Crop)
            .StartAsync(async ctx =>
            {
                while (!stopCts.IsCancellationRequested)
                {
                    // Drain new log lines into rolling buffer
                    while (logQueue.TryDequeue(out var line))
                    {
                        logLines.Enqueue(line);
                        while (logLines.Count > MaxLogLines) logLines.Dequeue();
                    }

                    // Sample llama-server process stats
                    (double CpuPct, double RamGb)? procStats = null;
                    if (pid.HasValue)
                    {
                        try
                        {
                            var proc = System.Diagnostics.Process.GetProcessById(pid.Value);
                            var now = DateTime.UtcNow;
                            var cpuTime = proc.TotalProcessorTime;
                            if (prevCpuTime != TimeSpan.Zero)
                            {
                                var elapsedMs = (now - prevCpuSample).TotalMilliseconds;
                                var cpuMs = (cpuTime - prevCpuTime).TotalMilliseconds;
                                double cpuPct = elapsedMs > 0
                                    ? Math.Min(100.0, cpuMs / (elapsedMs * Environment.ProcessorCount) * 100.0)
                                    : 0;
                                procStats = (cpuPct, proc.WorkingSet64 / (1024.0 * 1024 * 1024));
                            }
                            prevCpuTime = cpuTime;
                            prevCpuSample = now;
                        }
                        catch { /* process exited */ }
                    }

                    // Poll GPU every 3 seconds
                    if (gpuPollTick++ % 3 == 0)
                        gpuStats = await GetGpuStatsAsync();

                    // Live generation throughput from llama-server's /metrics, derived from
                    // the delta of cumulative predicted-token count between polls.
                    var metrics = await server.GetMetricsAsync(stopCts.Token);
                    if (metrics is not null)
                    {
                        var now = DateTime.UtcNow;
                        if (prevMetrics is not null && metrics.Value.RequestsProcessing > 0)
                        {
                            double tokenDelta = metrics.Value.TokensPredictedTotal - prevMetrics.Value.TokensPredictedTotal;
                            double secondsDelta = (now - prevMetricsSample).TotalSeconds;
                            liveTokPerSec = secondsDelta > 0 && tokenDelta >= 0 ? tokenDelta / secondsDelta : liveTokPerSec;
                        }
                        else if (metrics.Value.RequestsProcessing == 0)
                        {
                            liveTokPerSec = null; // idle — nothing generating right now
                        }
                        prevMetrics = metrics;
                        prevMetricsSample = now;
                    }

                    ctx.UpdateTarget(BuildRunningPanel(profile, serverExecutable, procStats, gpuStats, [.. logLines], showFullLog, showCommand, liveTokPerSec));

                    try { await Task.Delay(1000, stopCts.Token); }
                    catch (TaskCanceledException) { break; }
                }
            });

        server.SetLogHandler(null);
        await AnsiConsole.Status()
            .StartAsync("Stopping server...", async _ => await server.StopAsync());

        AnsiConsole.MarkupLine("[grey]Server stopped.[/]");
        await Task.Delay(600, CancellationToken.None);
    }

    // Wraps the active view (summary / full-log / command) with a sticky shortcut bar
    // that remains visible when switching between views.
    private static IRenderable BuildRunningPanel(
        SavedProfile profile,
        string serverExecutable,
        (double CpuPct, double RamGb)? proc,
        (int GpuPct, long UsedMb, long TotalMb)? gpu,
        string[] logs,
        bool showFullLog,
        bool showCommand,
        double? liveTokPerSec)
    {
        string lLabel = showFullLog ? "[bold green][[L]][/] [green]logs ◀[/]" : "[grey dim][[L]][/] [grey]logs[/]";
        string pLabel = showCommand ? "[bold green][[P]][/] [green]command ◀[/]" : "[grey dim][[P]][/] [grey]command[/]";
        var shortcuts = new Markup($"  {lLabel}   {pLabel}   [grey dim][[Esc]][/] [grey]stop server[/]");

        IRenderable body = showCommand
            ? BuildCommandPanel(profile, serverExecutable, liveTokPerSec)
            : showFullLog
                ? BuildFullLogPanel(logs, liveTokPerSec)
                : BuildSummaryPanel(profile, proc, gpu, liveTokPerSec);

        return new Rows(shortcuts, body);
    }

    private static IRenderable BuildSummaryPanel(
        SavedProfile profile,
        (double CpuPct, double RamGb)? proc,
        (int GpuPct, long UsedMb, long TotalMb)? gpu,
        double? liveTokPerSec)
    {
        var s = profile.Settings;

        // LEFT column: live stats
        var left = new Table().Border(TableBorder.None).HideHeaders()
            .AddColumn(new TableColumn("").NoWrap())
            .AddColumn("");

        if (profile.BenchmarkTgRate.HasValue)
        {
            left.AddRow("[grey]Bench TG (peak)[/]",   $"[green]~{profile.BenchmarkTgRate:F0} t/s[/]");
            left.AddRow("[grey]Bench PP (peak)[/]",   $"[green]~{profile.BenchmarkPpRate:F0} t/s[/]");
            left.AddRow("[grey]Bench TTFT[/]",        $"[green]~{profile.BenchmarkTtftMs:F0} ms[/]");
            left.AddRow("", "");
        }

        if (proc.HasValue)
        {
            left.AddRow("[grey]CPU  (server)[/]", FormatPct(proc.Value.CpuPct));
            left.AddRow("[grey]RAM  (server)[/]", $"[cyan]{proc.Value.RamGb:F2} GB[/]");
        }
        else
        {
            left.AddRow("[grey]CPU  (server)[/]", "[dim]sampling...[/]");
            left.AddRow("[grey]RAM  (server)[/]", "[dim]sampling...[/]");
        }
        if (gpu.HasValue)
        {
            left.AddRow("[grey]GPU[/]",  FormatPct(gpu.Value.GpuPct));
            left.AddRow("[grey]VRAM[/]", $"[cyan]{gpu.Value.UsedMb / 1024.0:F1} / {gpu.Value.TotalMb / 1024.0:F1} GB[/]");
        }
        left.AddRow("", "");
        left.AddRow("[grey]Live TG[/]", liveTokPerSec.HasValue
            ? $"[bold green]{liveTokPerSec.Value:F1} t/s[/]"
            : "[dim]idle[/]");

        // RIGHT column: connection + all settings
        var right = new Table().Border(TableBorder.None).HideHeaders()
            .AddColumn(new TableColumn("").NoWrap())
            .AddColumn("");

        right.AddRow("[grey]api-name[/]",  $"[cyan]{Markup.Escape(profile.ModelName)}[/]");
        right.AddRow("[grey]endpoint[/]",  "[bold]http://127.0.0.1:8080[/][grey]/v1  /health  /metrics[/]");
        right.AddRow("", "");
        right.AddRow("[grey]ctx-size[/]",          $"[cyan]{s.ContextSize:N0}[/] [grey]tokens[/]");
        right.AddRow("[grey]n-gpu-layers[/]",       $"[cyan]{(s.GpuLayers == -1 ? "all" : s.GpuLayers)}[/]");
        right.AddRow("[grey]batch / ubatch[/]",     $"[cyan]{s.BatchSize}[/] / [cyan]{s.UBatchSize}[/]");
        right.AddRow("[grey]parallel[/]",           $"[cyan]{s.ParallelSlots}[/]");
        right.AddRow("[grey]threads / t-batch[/]",  $"[cyan]{(s.Threads < 0 ? "auto" : s.Threads)}[/] / [cyan]{(s.ThreadsBatch < 0 ? "auto" : s.ThreadsBatch)}[/]");
        right.AddRow("[grey]flash-attn[/]",         s.FlashAttention ? "[green]yes[/]" : "no");
        right.AddRow("[grey]mmap[/]",               s.Mmap ? "yes" : "[yellow]no[/]");
        right.AddRow("[grey]mlock[/]",              s.Mlock ? "[yellow]yes[/]" : "no");
        right.AddRow("[grey]cache-type-k[/]",       $"[cyan]{s.CacheTypeK ?? "f16"}[/]");
        right.AddRow("[grey]cache-type-v[/]",       $"[cyan]{s.CacheTypeV ?? "f16"}[/]");
        if (s.MoeExpertUsed.HasValue)
            right.AddRow("[grey]experts (MoE)[/]", $"[cyan]{s.MoeExpertUsed.Value}[/]");
        if (s.ThinkingEnabled.HasValue)
        {
            string tv = s.ThinkingEnabled switch { true => "[green]enabled[/]", false => "[grey]disabled[/]", _ => "model default" };
            right.AddRow("[grey]thinking[/]", tv);
        }
        if (!string.IsNullOrWhiteSpace(s.ExtraArgs))
            right.AddRow("[grey]extra-args[/]", $"[yellow]{Markup.Escape(s.ExtraArgs)}[/]");
        if (!string.IsNullOrWhiteSpace(profile.Notes))
        {
            right.AddRow("", "");
            right.AddRow("[grey]notes[/]", $"[italic]{Markup.Escape(profile.Notes)}[/]");
        }

        var twoCol = new Table()
            .Border(TableBorder.None)
            .HideHeaders()
            .AddColumn(new TableColumn("").Padding(1, 0))
            .AddColumn(new TableColumn("").Padding(1, 0));
        twoCol.AddRow(left, right);

        return new Panel(twoCol)
        {
            Header = new PanelHeader($"[bold green] {Markup.Escape(profile.Name)} [/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(0, 0),
        };
    }

    private static IRenderable BuildCommandPanel(
        SavedProfile profile, string serverExecutable, double? liveTokPerSec)
    {
        string cmd = $"{serverExecutable} {string.Join(" ", profile.Settings.ToServerArgs(profile.ModelPath))}";

        var rows = new List<IRenderable>
        {
            new Markup("[grey]Run in any terminal to start this configuration:[/]"),
            new Rule().RuleStyle("grey dim"),
            new Panel(new Text(cmd, new Style(Color.LightGreen)))
            {
                Border = BoxBorder.Rounded,
                Padding = new Padding(1, 0),
            },
            new Rule().RuleStyle("grey dim"),
            new Markup($"[grey]API endpoint:[/]   [bold]http://127.0.0.1:8080/v1[/]"),
            new Markup($"[grey]Model name:[/]     [cyan]{Markup.Escape(profile.ModelName)}[/]"),
            new Markup($"[grey]Live TG:[/]        {(liveTokPerSec.HasValue ? $"[bold green]{liveTokPerSec.Value:F1} t/s[/]" : "[dim]idle[/]")}"),
        };

        return new Panel(new Rows(rows))
        {
            Header = new PanelHeader($"[bold green] {Markup.Escape(profile.Name)} — Command [/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(1, 0),
        };
    }

    private static IRenderable BuildFullLogPanel(string[] logs, double? liveTokPerSec)
    {
        // Stay a few lines under the window height to avoid cursor-drift on redraws.
        // Shortcut bar (1 line) + panel borders (2) + live-gen line + rule (2) + margin (4)
        int height = Math.Max(3, Console.WindowHeight - 10);
        var tail = logs.Skip(Math.Max(0, logs.Length - height)).ToList();

        var rows = new List<IRenderable>
        {
            new Markup(liveTokPerSec.HasValue
                ? $"[grey]Live generation:[/] [bold green]{liveTokPerSec.Value:F1} t/s[/]"
                : "[grey]Live generation:[/] [dim]idle[/]"),
            new Rule().RuleStyle("grey dim"),
        };
        foreach (var line in tail)
            rows.Add(new Text(line));
        if (tail.Count == 0)
            rows.Add(new Text("[waiting for output...]", new Style(Color.Grey)));

        return new Panel(new Rows(rows))
        {
            Header = new PanelHeader("[bold green] Full Log [/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(0, 0),
        };
    }

    private static string FormatPct(double pct)
    {
        var col = pct > 90 ? "red" : pct > 70 ? "yellow" : "green";
        return $"[{col}]{pct:F1}%[/]";
    }

    private static async Task<(int GpuPct, long UsedMb, long TotalMb)?> GetGpuStatsAsync()
    {
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = System.Diagnostics.Process.Start(psi);
            if (p is null) return null;
            var line = await p.StandardOutput.ReadLineAsync();
            await p.WaitForExitAsync();
            if (line is null) return null;
            var parts = line.Split(',');
            if (parts.Length >= 3
                && int.TryParse(parts[0].Trim(), out int gpuPct)
                && long.TryParse(parts[1].Trim(), out long used)
                && long.TryParse(parts[2].Trim(), out long total))
                return (gpuPct, used, total);
        }
        catch { }
        return null;
    }

    // -------------------------------------------------------------------------
    // Edit / manage
    // -------------------------------------------------------------------------

    private async Task EditProfileAsync(List<SavedProfile> profiles)
    {
        var profile = PickProfile(profiles, "Select profile to edit:");
        if (profile is null) return;

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"Editing settings for [cyan]{Markup.Escape(profile.Name)}[/]:");

        var updated = SettingsEditor.Edit(profile.Settings, $"Edit — {profile.Name}",
            turboQuantAvailable: !string.IsNullOrWhiteSpace(appSettings.Current.TurboQuantServerExecutable));

        // Copy changed values back (Settings is init-only, so replace inline)
        var fresh = new SavedProfile
        {
            Name = profile.Name,
            ModelPath = profile.ModelPath,
            Settings = updated,
            OptimizationProfile = profile.OptimizationProfile,
            BenchmarkScore = profile.BenchmarkScore,
            BenchmarkPpRate = profile.BenchmarkPpRate,
            BenchmarkTgRate = profile.BenchmarkTgRate,
            BenchmarkTtftMs = profile.BenchmarkTtftMs,
            Notes = profile.Notes + (profile.Notes?.Length > 0 ? " " : "") + "[manually edited]",
        };

        // Preserve same file by keeping the ID — delete old, save fresh with same Id
        library.Delete(profile);

        var finalProfile = new SavedProfile
        {
            Id = profile.Id,
            Name = fresh.Name,
            ModelPath = fresh.ModelPath,
            Settings = fresh.Settings,
            OptimizationProfile = fresh.OptimizationProfile,
            BenchmarkScore = fresh.BenchmarkScore,
            BenchmarkPpRate = fresh.BenchmarkPpRate,
            BenchmarkTgRate = fresh.BenchmarkTgRate,
            BenchmarkTtftMs = fresh.BenchmarkTtftMs,
            Notes = fresh.Notes,
            CreatedAt = profile.CreatedAt,
            LastRunAt = profile.LastRunAt,
        };

        await library.SaveAsync(finalProfile);
        AnsiConsole.MarkupLine($"[green]Profile updated:[/] {Markup.Escape(profile.Name)}");
        await Task.Delay(800, CancellationToken.None);
    }

    private async Task RenameProfileAsync(List<SavedProfile> profiles)
    {
        var profile = PickProfile(profiles, "Select profile to rename:");
        if (profile is null) return;

        string newName = AnsiConsole.Prompt(
            new TextPrompt<string>($"New name for [cyan]{Markup.Escape(profile.Name)}[/]:")
                .DefaultValue(profile.Name)
                .Validate(n => !string.IsNullOrWhiteSpace(n)
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Name cannot be empty")));

        await library.RenameAsync(profile, newName.Trim());
        AnsiConsole.MarkupLine($"[green]Renamed to:[/] {Markup.Escape(newName)}");
        await Task.Delay(800, CancellationToken.None);
    }

    private async Task AddNotesAsync(List<SavedProfile> profiles)
    {
        var profile = PickProfile(profiles, "Select profile to annotate:");
        if (profile is null) return;

        string notes = AnsiConsole.Prompt(
            new TextPrompt<string>($"Notes for [cyan]{Markup.Escape(profile.Name)}[/]:")
                .DefaultValue(profile.Notes ?? "")
                .AllowEmpty());

        profile.Notes = string.IsNullOrWhiteSpace(notes) ? null : notes;
        await library.SaveAsync(profile);
        AnsiConsole.MarkupLine("[green]Notes saved.[/]");
        await Task.Delay(600, CancellationToken.None);
    }

    private async Task DeleteProfileAsync(List<SavedProfile> profiles)
    {
        var profile = PickProfile(profiles, "Select profile to delete:");
        if (profile is null) return;

        if (!AnsiConsole.Confirm(
            $"[red]Delete[/] [cyan]{Markup.Escape(profile.Name)}[/]? This cannot be undone.", false))
            return;

        library.Delete(profile);
        AnsiConsole.MarkupLine($"[grey]Deleted {Markup.Escape(profile.Name)}.[/]");
        await Task.Delay(600, CancellationToken.None);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static SavedProfile? PickProfile(List<SavedProfile> profiles, string title)
    {
        if (profiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No profiles available.[/]");
            return null;
        }

        var labels = profiles
            .Select(p =>
            {
                string avail = ProfileLibraryService.IsModelAvailable(p) ? "" : " [red][missing][/]";
                string meta = p.BenchmarkTgRate.HasValue
                    ? $"[grey]  TG={p.BenchmarkTgRate:F0}t/s[/]"
                    : "";
                string profileType = p.OptimizationProfile.HasValue
                    ? $"[grey]({p.OptimizationProfile})[/]  " : "";
                string lastRun = p.LastRunAt.HasValue
                    ? $"[grey]  last run {p.LastRunAt:MM-dd}[/]"
                    : "";
                return $"[cyan]{Markup.Escape(p.Name)}[/]{avail}  {profileType}{meta}{lastRun}";
            })
            .ToList();

        string? selected = MenuHelper.Select(title, labels);

        if (selected is null) return null; // Escape pressed

        int idx = labels.IndexOf(selected);
        return idx >= 0 ? profiles[idx] : null;
    }

    private static void RenderProfileTable(List<SavedProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .Title("[bold]Saved Profiles[/]");

        table.AddColumn("[grey]Name[/]");
        table.AddColumn("[grey]Model[/]");
        table.AddColumn("[grey]Type[/]");
        table.AddColumn(new TableColumn("[grey]TG t/s[/]").RightAligned());
        table.AddColumn(new TableColumn("[grey]PP t/s[/]").RightAligned());
        table.AddColumn(new TableColumn("[grey]Score[/]").RightAligned());
        table.AddColumn("[grey]Key Settings[/]");
        table.AddColumn("[grey]Last Run[/]");

        foreach (var p in profiles)
        {
            bool avail = ProfileLibraryService.IsModelAvailable(p);
            string nameCell = avail
                ? $"[bold]{Markup.Escape(p.Name)}[/]"
                : $"[grey]{Markup.Escape(p.Name)}[/] [red][missing][/]";

            table.AddRow(
                nameCell,
                Markup.Escape(p.ModelName),
                p.OptimizationProfile?.ToString() ?? "[grey]—[/]",
                p.BenchmarkTgRate.HasValue ? $"{p.BenchmarkTgRate:F0}" : "[grey]—[/]",
                p.BenchmarkPpRate.HasValue ? $"{p.BenchmarkPpRate:F0}" : "[grey]—[/]",
                p.BenchmarkScore.HasValue ? $"{p.BenchmarkScore:F3}" : "[grey]—[/]",
                $"[grey]{Markup.Escape(p.Settings.Summary())}[/]",
                p.LastRunAt.HasValue ? p.LastRunAt.Value.ToString("MM-dd HH:mm") : "[grey]never[/]"
            );
        }

        AnsiConsole.Write(table);
    }
}
