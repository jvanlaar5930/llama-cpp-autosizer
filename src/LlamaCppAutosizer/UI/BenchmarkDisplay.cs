using LlamaCppAutosizer.Models;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

public static class BenchmarkDisplay
{
    /// <summary>
    /// Full per-iteration history — what was tried, why (the recommender's reasoning,
    /// whether it came from the LLM or a heuristic), and the resulting score/speeds.
    /// Used both right after an optimization run and when re-loading a past session.
    /// </summary>
    public static void RenderIterationHistory(OptimizationSession session)
    {
        if (session.Iterations.Count == 0) return;

        var table = BuildIterationTable();
        foreach (var iter in session.Iterations)
            AddIterationRow(table, iter);

        AnsiConsole.Write(new Rule("[bold]Optimization History[/]").RuleStyle("grey"));
        AnsiConsole.Write(table);
    }

    private static Table BuildIterationTable()
    {
        var t = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .Title("[bold]Iteration History[/]");

        t.AddColumn(new TableColumn("[grey]#[/]").RightAligned());
        t.AddColumn("[grey]Change[/]");
        t.AddColumn(new TableColumn("[grey]PP t/s[/]").RightAligned());
        t.AddColumn(new TableColumn("[grey]TG t/s[/]").RightAligned());
        t.AddColumn(new TableColumn("[grey]TTFT ms[/]").RightAligned());
        t.AddColumn(new TableColumn("[grey]Score[/]").RightAligned());
        t.AddColumn("[grey]Source[/]");
        t.AddColumn("[grey]Reasoning / Notes[/]");
        return t;
    }

    private static void AddIterationRow(Table table, OptimizationIteration iter)
    {
        var r = iter.Result;
        string num = iter.IsBestSoFar
            ? $"[bold green]{iter.Number}★[/]"
            : iter.Number.ToString();

        string change = iter.AppliedChange is null
            ? "[grey]baseline[/]"
            : $"[yellow]{Markup.Escape(iter.AppliedChange.Parameter)}[/] → [cyan]{Markup.Escape(iter.AppliedChange.NewValue?.ToString() ?? "")}[/]";

        string source = iter.AppliedChange?.Source switch
        {
            "llm" => "[magenta]LLM[/]",
            "llm-push" => "[magenta]LLM★[/]",
            "user" => "[green]user[/]",
            _ => "[grey]heuristic[/]",
        };

        bool wasSkipped = iter.AppliedChange is not null && r.CompositeScore == 0 && r.GenerationRate == 0;
        string score = wasSkipped ? "[red]—[/]"
            : iter.IsBestSoFar ? $"[bold green]{r.CompositeScore:F3}[/]"
            : $"[grey]{r.CompositeScore:F3}[/]";

        // Skipped iterations: show why the server failed to start (from StatusMessage), not
        // the change's original reasoning — the failure is the more useful fact here.
        // Otherwise prefer the recommender's own reasoning, falling back to the full status
        // line (which also carries any auto-adjustment note) when there's no structured reasoning.
        string notes = wasSkipped
            ? Markup.Escape(iter.StatusMessage ?? "")
            : !string.IsNullOrWhiteSpace(iter.AppliedChange?.Reasoning)
                ? Markup.Escape(iter.AppliedChange.Reasoning)
                : Markup.Escape(iter.StatusMessage ?? "");

        table.AddRow(
            num,
            change,
            wasSkipped ? "[grey]—[/]" : $"{r.PromptProcessingRate,7:F1}",
            wasSkipped ? "[grey]—[/]" : $"{r.GenerationRate,6:F1}",
            wasSkipped ? "[grey]—[/]" : $"{r.TimeToFirstTokenMs,6:F0}",
            score,
            source,
            notes);
    }

    // -------------------------------------------------------------------------
    // Static result displays
    // -------------------------------------------------------------------------

    public static void RenderFinalResults(OptimizationSession session, string serverExecutable)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold green]Optimization Complete[/]").RuleStyle("green"));
        AnsiConsole.WriteLine();

        if (session.BestResult is null)
        {
            AnsiConsole.MarkupLine("[red]No results collected.[/]");
            return;
        }

        var r = session.BestResult;
        var s = session.BestSettings!;

        // Metrics panel
        AnsiConsole.Write(
            new Panel(
                new Table()
                    .Border(TableBorder.None)
                    .HideHeaders()
                    .AddColumn("").AddColumn("")
                    .AddRow("[grey]Model[/]", $"[cyan]{session.ModelName}[/]")
                    .AddRow("[grey]Profile[/]", $"[cyan]{session.Profile}[/]")
                    .AddRow("[grey]Iterations[/]", $"[cyan]{session.Iterations.Count}[/]")
                    .AddRow("[grey]Best score[/]", $"[bold green]{r.CompositeScore:F3}[/]")
                    .AddRow("[grey]Context[/]", $"[cyan]{s.ContextSize:N0} tokens[/]")
                    .AddRow("[grey]GPU layers[/]", $"[cyan]{(s.GpuLayers == -1 ? "all" : s.GpuLayers)}[/]")
                    .AddRow("[grey]PP speed[/]", $"[cyan]{r.PromptProcessingRate:F1} t/s[/]")
                    .AddRow("[grey]TG speed[/]", $"[cyan]{r.GenerationRate:F1} t/s[/]")
                    .AddRow("[grey]TTFT[/]", $"[cyan]{r.TimeToFirstTokenMs:F0} ms[/]")
                    .AddRow("[grey]Completion[/]", $"[grey]{Markup.Escape(session.CompletionReason ?? "")}[/]"))
            {
                Header = new PanelHeader("[bold]Results Summary[/]"),
                Border = BoxBorder.Rounded,
                Padding = new Padding(1, 0),
            });

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold yellow]Best Settings[/]").RuleStyle("yellow"));
        SettingsEditor.RenderSettings(s);

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold blue]Ready-to-use Command[/]").RuleStyle("blue"));
        var cmd = $"{serverExecutable} {string.Join(" ", s.ToServerArgs(session.ModelPath))}";
        AnsiConsole.Write(new Panel(new Text(cmd, new Style(Color.LightGreen)))
        {
            Border = BoxBorder.Rounded,
            Padding = new Padding(1, 0),
        });

        // Show improvement over baseline
        if (session.Iterations.Count >= 2)
        {
            var baseline = session.Iterations[0].Result;
            double tgImprove = (r.GenerationRate - baseline.GenerationRate) / Math.Max(1, baseline.GenerationRate) * 100;
            double ppImprove = (r.PromptProcessingRate - baseline.PromptProcessingRate) / Math.Max(1, baseline.PromptProcessingRate) * 100;

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]vs baseline:[/] PP [green]+{ppImprove:F1}%[/]  TG [green]+{tgImprove:F1}%[/]");
        }

        AnsiConsole.WriteLine();
        RenderIterationHistory(session);
    }

    public static void RenderSessionList(List<string> sessionFiles)
    {
        if (sessionFiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No saved sessions found.[/]");
            return;
        }

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Saved Sessions[/]")
            .AddColumn("#")
            .AddColumn("File")
            .AddColumn("Modified");

        for (int i = 0; i < sessionFiles.Count; i++)
        {
            var f = sessionFiles[i];
            table.AddRow(
                (i + 1).ToString(),
                Path.GetFileName(f),
                File.GetLastWriteTime(f).ToString("yyyy-MM-dd HH:mm"));
        }

        AnsiConsole.Write(table);
    }

    public static void RenderHardwareInfo(HardwareInfo hw)
    {
        AnsiConsole.Write(new Rule("[bold]Hardware[/]").RuleStyle("blue"));

        var table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        table.AddColumn("[grey]Component[/]");
        table.AddColumn("[grey]Details[/]");

        if (hw.Gpus.Count == 0)
        {
            table.AddRow("GPU", "[yellow]None detected[/]");
        }
        else
        {
            foreach (var gpu in hw.Gpus)
            {
                table.AddRow($"GPU {gpu.Index}", $"[cyan]{Markup.Escape(gpu.Name)}[/]");
                table.AddRow("  VRAM", $"[cyan]{gpu.VramTotalMb:N0} MB[/] ({gpu.VramFreeMb:N0} MB free)");
            }
        }

        table.AddRow("CPU", $"[cyan]{Markup.Escape(hw.CpuName)}[/]");
        table.AddRow("Cores / Threads", $"[cyan]{hw.CpuCores} / {hw.CpuThreads}[/]");
        table.AddRow("RAM", $"[cyan]{hw.RamTotalMb:N0} MB[/] ({hw.RamFreeMb:N0} MB free)");

        AnsiConsole.Write(table);
    }

    // -------------------------------------------------------------------------
    // Historical comparison
    // -------------------------------------------------------------------------

    /// <summary>
    /// Renders a rich side-by-side comparison of the best result from every session.
    /// Rows are sorted by composite score descending so the best configuration
    /// is always at the top.
    /// </summary>
    public static void RenderHistoricalComparison(
        IReadOnlyList<OptimizationSession> sessions,
        string? highlightModel = null)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold cyan]Historical Benchmark Comparison[/]").RuleStyle("cyan"));
        AnsiConsole.WriteLine();

        if (sessions.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No completed sessions found.[/]");
            return;
        }

        // Group by profile so Chat and Agentic don't compete against each other
        var byProfile = sessions
            .Where(s => s.BestResult is not null)
            .GroupBy(s => s.Profile)
            .OrderBy(g => g.Key.ToString());

        foreach (var group in byProfile)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule($"[bold]{group.Key} Profile[/]").RuleStyle("yellow"));

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            table.AddColumn(new TableColumn("[grey]Rank[/]").RightAligned());
            table.AddColumn("[grey]Model[/]");
            table.AddColumn(new TableColumn("[grey]Score[/]").RightAligned());
            table.AddColumn(new TableColumn("[grey]PP t/s[/]").RightAligned());
            table.AddColumn(new TableColumn("[grey]TG t/s[/]").RightAligned());
            table.AddColumn(new TableColumn("[grey]TTFT ms[/]").RightAligned());
            table.AddColumn(new TableColumn("[grey]Iters[/]").RightAligned());
            table.AddColumn("[grey]Key Settings[/]");
            table.AddColumn("[grey]Date[/]");

            int rank = 1;
            foreach (var s in group.OrderByDescending(s => s.BestResult!.CompositeScore))
            {
                var r = s.BestResult!;
                var st = s.BestSettings!;

                bool isHighlight = highlightModel is not null &&
                    s.ModelName.Contains(highlightModel, StringComparison.OrdinalIgnoreCase);
                bool isFirst = rank == 1;

                string modelCell = isFirst
                    ? $"[bold green]{Markup.Escape(s.ModelName)}[/]"
                    : isHighlight
                        ? $"[yellow]{Markup.Escape(s.ModelName)}[/]"
                        : Markup.Escape(s.ModelName);

                string scoreCell = isFirst
                    ? $"[bold green]{r.CompositeScore:F3}[/]"
                    : $"[grey]{r.CompositeScore:F3}[/]";

                // Delta vs best for each metric
                var bestInGroup = group.Max(x => x.BestResult!.CompositeScore);
                double delta = r.CompositeScore - bestInGroup;
                string rankCell = rank == 1 ? "[bold green]#1[/]" : $"[grey]#{rank}[/]";

                string settingsCell =
                    $"[grey]ctx={st.ContextSize} ngl={st.GpuLayers} fa={st.FlashAttention} " +
                    $"kv={st.CacheTypeK ?? "f16"}[/]";

                table.AddRow(
                    rankCell,
                    modelCell,
                    scoreCell,
                    $"{r.PromptProcessingRate,7:F1}",
                    $"{r.GenerationRate,6:F1}",
                    $"{r.TimeToFirstTokenMs,6:F0}",
                    s.Iterations.Count.ToString(),
                    settingsCell,
                    s.StartedAt.ToString("MM-dd HH:mm"));

                rank++;
            }

            AnsiConsole.Write(table);
        }

        // Summary stats
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Total sessions: {sessions.Count}  " +
            $"Models: {sessions.Select(s => s.ModelName).Distinct().Count()}  " +
            $"Date range: {sessions.Min(s => s.StartedAt):yyyy-MM-dd} → {sessions.Max(s => s.StartedAt):yyyy-MM-dd}[/]");
    }

    /// <summary>Renders the model discovery results as a formatted pick-list table.</summary>
    public static void RenderModelList(IReadOnlyList<(string Path, long SizeMb)> models)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .Title("[bold]Discovered Models[/]");

        table.AddColumn(new TableColumn("[grey]#[/]").RightAligned());
        table.AddColumn("[grey]File[/]");
        table.AddColumn(new TableColumn("[grey]Size[/]").RightAligned());
        table.AddColumn("[grey]Directory[/]");

        for (int i = 0; i < models.Count; i++)
        {
            var (path, sizeMb) = models[i];
            string sizeStr = sizeMb >= 1024
                ? $"[cyan]{sizeMb / 1024.0:F1} GB[/]"
                : $"[grey]{sizeMb} MB[/]";

            table.AddRow(
                (i + 1).ToString(),
                Markup.Escape(Path.GetFileName(path)),
                sizeStr,
                Markup.Escape(Path.GetDirectoryName(path) ?? ""));
        }

        AnsiConsole.Write(table);
    }
}
