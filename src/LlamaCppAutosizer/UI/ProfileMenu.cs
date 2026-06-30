using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using LlamaCppAutosizer.UI;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace LlamaCppAutosizer.UI;

public class ProfileMenu(
    ProfileLibraryService library,
    LlamaServerService server)
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
                AnsiConsole.MarkupLine("[grey]Run an optimization (option 5) or edit settings manually (option 6)[/]");
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
        _ = Task.Run(() =>
        {
            while (!stopCts.IsCancellationRequested)
            {
                if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(intercept: true);
                    if (key.Key == ConsoleKey.L)
                        showFullLog = !showFullLog;
                    else
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

        await AnsiConsole.Live(BuildRunningPanel(profile, null, null, [], false, null))
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

                    ctx.UpdateTarget(BuildRunningPanel(profile, procStats, gpuStats, [.. logLines], showFullLog, liveTokPerSec));

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

    private static IRenderable BuildRunningPanel(
        SavedProfile profile,
        (double CpuPct, double RamGb)? proc,
        (int GpuPct, long UsedMb, long TotalMb)? gpu,
        string[] logs,
        bool showFullLog,
        double? liveTokPerSec)
    {
        if (showFullLog)
            return BuildFullLogPanel(logs, liveTokPerSec);


        var s = profile.Settings;
        bool isMoe = s.MoeExpertUsed.HasValue;
        bool isThinking = s.ThinkingEnabled.HasValue;

        var info = new Table().Border(TableBorder.None).HideHeaders()
            .AddColumn(new TableColumn("").NoWrap())
            .AddColumn("");

        info.AddRow("[grey]Model[/]",    $"[cyan]{Markup.Escape(profile.ModelName)}[/]");
        info.AddRow("[grey]Profile[/]",  $"[cyan]{Markup.Escape(profile.Name)}[/]");
        info.AddRow("[grey]Endpoints[/]", "[bold]http://127.0.0.1:8080/v1[/]  [grey]/health  /metrics[/]");
        info.AddRow("", "");

        // All settings (mirrors llama-server command exactly)
        info.AddRow("[grey]ctx-size[/]",      $"[cyan]{s.ContextSize:N0}[/] tokens");
        info.AddRow("[grey]n-gpu-layers[/]",   $"[cyan]{(s.GpuLayers == -1 ? "all" : s.GpuLayers)}[/]");
        info.AddRow("[grey]batch-size[/]",     $"[cyan]{s.BatchSize}[/]");
        info.AddRow("[grey]ubatch-size[/]",    $"[cyan]{s.UBatchSize}[/]");
        info.AddRow("[grey]parallel[/]",       $"[cyan]{s.ParallelSlots}[/]");
        info.AddRow("[grey]threads[/]",        $"[cyan]{(s.Threads < 0 ? "auto" : s.Threads)}[/]");
        info.AddRow("[grey]threads-batch[/]",  $"[cyan]{(s.ThreadsBatch < 0 ? "auto" : s.ThreadsBatch)}[/]");
        info.AddRow("[grey]flash-attn[/]",     s.FlashAttention ? "[green]yes[/]" : "no");
        info.AddRow("[grey]mmap[/]",           s.Mmap ? "yes" : "[yellow]no (--no-mmap)[/]");
        info.AddRow("[grey]mlock[/]",          s.Mlock ? "[yellow]yes[/]" : "no");
        info.AddRow("[grey]cache-type-k[/]",   $"[cyan]{s.CacheTypeK ?? "f16"}[/]");
        info.AddRow("[grey]cache-type-v[/]",   $"[cyan]{s.CacheTypeV ?? "f16"}[/]");
        if (isMoe)
            info.AddRow("[grey]experts (MoE)[/]", $"[cyan]{s.MoeExpertUsed!.Value}[/]");
        if (isThinking)
        {
            string tv = s.ThinkingEnabled switch { true => "[green]enabled[/]", false => "[grey]disabled[/]", _ => "model default" };
            info.AddRow("[grey]thinking[/]", tv);
        }
        if (!string.IsNullOrWhiteSpace(s.ExtraArgs))
            info.AddRow("[grey]extra-args[/]", $"[yellow]{Markup.Escape(s.ExtraArgs)}[/]");

        if (profile.BenchmarkTgRate.HasValue)
        {
            info.AddRow("", "");
            info.AddRow("[grey]Benchmark TG[/]",   $"[green]~{profile.BenchmarkTgRate:F0} t/s[/]");
            info.AddRow("[grey]Benchmark PP[/]",   $"[green]~{profile.BenchmarkPpRate:F0} t/s[/]");
            info.AddRow("[grey]Benchmark TTFT[/]", $"[green]~{profile.BenchmarkTtftMs:F0} ms[/]");
        }

        // Real-time stats
        info.AddRow("", "");
        if (proc.HasValue)
        {
            info.AddRow("[grey]CPU  (server)[/]", FormatPct(proc.Value.CpuPct));
            info.AddRow("[grey]RAM  (server)[/]", $"[cyan]{proc.Value.RamGb:F2} GB[/]");
        }
        else
        {
            info.AddRow("[grey]CPU  (server)[/]", "[dim]sampling...[/]");
            info.AddRow("[grey]RAM  (server)[/]", "[dim]sampling...[/]");
        }
        if (gpu.HasValue)
        {
            info.AddRow("[grey]GPU[/]", FormatPct(gpu.Value.GpuPct));
            info.AddRow("[grey]VRAM[/]", $"[cyan]{gpu.Value.UsedMb / 1024.0:F1} / {gpu.Value.TotalMb / 1024.0:F1} GB[/]");
        }
        info.AddRow("[grey]Live TG[/]", liveTokPerSec.HasValue
            ? $"[green]{liveTokPerSec.Value:F1} t/s[/]"
            : "[dim]idle[/]");

        if (!string.IsNullOrWhiteSpace(profile.Notes))
        {
            info.AddRow("", "");
            info.AddRow("[grey]Notes[/]", $"[italic]{Markup.Escape(profile.Notes)}[/]");
        }

        List<IRenderable> parts = [new Padder(info, new Padding(1, 0))];

        // Keep the panel within the console window — Spectre's Live display doesn't
        // overwrite cleanly when a frame is taller than the terminal, which produces
        // a runaway stack of duplicated, truncated panels instead of one in-place update.
        const int MaxTailLogLines = 8;
        const int ChromeRows = 4; // top+bottom border, plus safety margin for cursor drift
        int available = Math.Max(0, Console.WindowHeight - info.Rows.Count - ChromeRows);
        int tailLogLines = logs.Length > 0 ? Math.Clamp(available - 1, 0, MaxTailLogLines) : 0;

        if (tailLogLines > 0)
        {
            parts.Add(new Rule("[grey dim]logs[/]").RuleStyle("grey dim"));
            foreach (var line in logs.Skip(Math.Max(0, logs.Length - tailLogLines)))
                parts.Add(new Text(line));
        }

        var panel = new Panel(new Rows(parts))
        {
            Header = new PanelHeader("[bold green] Server Running [/]  [grey]L = full logs   any other key = stop[/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(0, 0),
        };

        return panel;
    }

    private static IRenderable BuildFullLogPanel(string[] logs, double? liveTokPerSec)
    {
        // Chrome: top+bottom border (2), live-gen line + rule (2), plus safety margin
        // for cursor drift — Live's relative redraw breaks if a frame forces the
        // terminal to scroll, so we deliberately stay a few lines under the window.
        int height = Math.Max(3, Console.WindowHeight - 9);
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

        var panel = new Panel(new Rows(rows))
        {
            Header = new PanelHeader("[bold green] Server Running — Full Log [/]  [grey]L = back to summary   any other key = stop[/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(0, 0),
        };

        return panel;
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

        var updated = SettingsEditor.Edit(profile.Settings, $"Edit — {profile.Name}");

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
