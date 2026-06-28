using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using LlamaCppAutosizer.UI;
using Spectre.Console;

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

        // Start server
        bool started = false;
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
                    AnsiConsole.MarkupLine($"[red]Failed to start server:[/] {Markup.Escape(ex.Message)}");
                }
            });

        if (!started)
        {
            AnsiConsole.WriteLine("Press any key...");
            Console.ReadKey(intercept: true);
            return;
        }

        await library.RecordRunAsync(profile);

        // Show the running panel
        RenderRunningPanel(profile);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Press any key to stop the server.[/]");
        Console.ReadKey(intercept: true);

        await AnsiConsole.Status()
            .StartAsync("Stopping server...", async _ => await server.StopAsync());

        AnsiConsole.MarkupLine("[grey]Server stopped.[/]");
        await Task.Delay(600, CancellationToken.None);
    }

    private static void RenderRunningPanel(SavedProfile profile)
    {
        var inner = new Table().Border(TableBorder.None).HideHeaders()
            .AddColumn("").AddColumn("");

        inner.AddRow("[grey]Model[/]",    $"[cyan]{Markup.Escape(profile.ModelName)}[/]");
        inner.AddRow("[grey]Profile[/]",  $"[cyan]{Markup.Escape(profile.Name)}[/]");
        inner.AddRow("[grey]Endpoint[/]", "[bold]http://127.0.0.1:8080/v1[/]");
        inner.AddRow("[grey]Health[/]",   "[bold]http://127.0.0.1:8080/health[/]");
        inner.AddRow("", "");
        inner.AddRow("[grey]Context[/]",  $"[cyan]{profile.Settings.ContextSize:N0} tokens[/]");
        inner.AddRow("[grey]GPU layers[/]", $"[cyan]{(profile.Settings.GpuLayers == -1 ? "all" : profile.Settings.GpuLayers)}[/]");
        inner.AddRow("[grey]Flash attn[/]", profile.Settings.FlashAttention ? "[green]on[/]" : "[grey]off[/]");
        inner.AddRow("[grey]KV cache K[/]", $"[cyan]{profile.Settings.CacheTypeK ?? "f16"}[/]");
        inner.AddRow("[grey]KV cache V[/]", $"[cyan]{profile.Settings.CacheTypeV ?? "f16"}[/]");

        if (profile.BenchmarkTgRate.HasValue)
        {
            inner.AddRow("", "");
            inner.AddRow("[grey]Expected TG[/]", $"[green]~{profile.BenchmarkTgRate:F0} t/s[/]");
            inner.AddRow("[grey]Expected PP[/]", $"[green]~{profile.BenchmarkPpRate:F0} t/s[/]");
            inner.AddRow("[grey]Expected TTFT[/]", $"[green]~{profile.BenchmarkTtftMs:F0} ms[/]");
        }

        if (!string.IsNullOrWhiteSpace(profile.Notes))
            inner.AddRow("[grey]Notes[/]", $"[italic]{Markup.Escape(profile.Notes)}[/]");

        AnsiConsole.Write(new Panel(inner)
        {
            Header = new PanelHeader("[bold green] Server Running [/]"),
            Border = BoxBorder.Double,
            Padding = new Padding(1, 0),
        });
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
