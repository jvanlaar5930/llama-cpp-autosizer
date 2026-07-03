using LlamaCppAutosizer.Services;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

// File-path-related app configuration: model folder, llama-server executable, and where
// named profiles / optimization history (sessions) are stored. Distinct from
// "Edit llama.cpp Settings Manually", which edits per-run model/server tuning parameters.
public class AppSettingsMenu(AppSettingsService appSettings)
{
    public void Run()
    {
        while (true)
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[bold cyan]Settings[/]").RuleStyle("cyan"));
            AnsiConsole.WriteLine();
            RenderCurrent();
            AnsiConsole.WriteLine();

            string? choice = MenuHelper.Select(
                "[grey]Choose a setting to edit[/]  [dim](Esc = back)[/]",
                [
                    "Model Folder  (scanned for .gguf files before optimization)",
                    "llama-server Executable Path",
                    "Profiles Output Directory",
                    "Sessions / Optimization History Directory",
                    "Reset directories to default shared location",
                ]);

            if (choice is null) return; // Esc

            if (choice.StartsWith("Model Folder")) EditModelFolder();
            else if (choice.StartsWith("llama-server")) EditServerExecutable();
            else if (choice.StartsWith("Profiles")) EditProfilesDirectory();
            else if (choice.StartsWith("Sessions")) EditSessionsDirectory();
            else if (choice.StartsWith("Reset")) ResetDirectories();
        }
    }

    private void RenderCurrent()
    {
        var s = appSettings.Current;
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .HideHeaders()
            .AddColumn(new TableColumn("").Width(28))
            .AddColumn("");

        table.AddRow("[grey]Model folder[/]", string.IsNullOrWhiteSpace(s.ModelFolder)
            ? "[yellow]not set[/]" : $"[cyan]{Markup.Escape(s.ModelFolder)}[/]");
        table.AddRow("[grey]llama-server path[/]", $"[cyan]{Markup.Escape(s.ServerExecutable)}[/]");
        table.AddRow("[grey]Profiles directory[/]", $"[cyan]{Markup.Escape(s.ProfilesDirectory)}[/]");
        table.AddRow("[grey]Sessions directory[/]", $"[cyan]{Markup.Escape(s.SessionsDirectory)}[/]");

        AnsiConsole.Write(table);
    }

    private void EditModelFolder()
    {
        string current = appSettings.Current.ModelFolder;
        string folder = AnsiConsole.Prompt(
            new TextPrompt<string>("[grey]Folder to scan for[/] [cyan].gguf[/] [grey]files:[/]")
                .DefaultValue(string.IsNullOrWhiteSpace(current) ? Directory.GetCurrentDirectory() : current)
                .Validate(p => Directory.Exists(p)
                    ? ValidationResult.Success()
                    : ValidationResult.Error("Folder not found")));

        appSettings.Current.ModelFolder = folder;
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]Model folder set:[/] {Markup.Escape(folder)}");
        Pause();
    }

    private void EditServerExecutable()
    {
        string exe = AnsiConsole.Prompt(
            new TextPrompt<string>("Path to [cyan]llama-server[/] executable:")
                .DefaultValue(appSettings.Current.ServerExecutable));

        appSettings.Current.ServerExecutable = exe;
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]Server path set:[/] {Markup.Escape(exe)}");
        Pause();
    }

    private void EditProfilesDirectory()
    {
        string dir = AnsiConsole.Prompt(
            new TextPrompt<string>("[grey]Directory to store named profiles in:[/]")
                .DefaultValue(appSettings.Current.ProfilesDirectory));

        if (!TryCreateDirectory(dir)) { Pause(); return; }

        appSettings.Current.ProfilesDirectory = dir;
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]Profiles directory set:[/] {Markup.Escape(dir)}");
        AnsiConsole.MarkupLine("[grey]Existing profiles in the old location are not moved automatically.[/]");
        Pause();
    }

    private void EditSessionsDirectory()
    {
        string dir = AnsiConsole.Prompt(
            new TextPrompt<string>("[grey]Directory to store optimization history (sessions) in:[/]")
                .DefaultValue(appSettings.Current.SessionsDirectory));

        if (!TryCreateDirectory(dir)) { Pause(); return; }

        appSettings.Current.SessionsDirectory = dir;
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]Sessions directory set:[/] {Markup.Escape(dir)}");
        AnsiConsole.MarkupLine("[grey]Existing sessions in the old location are not moved automatically.[/]");
        Pause();
    }

    private void ResetDirectories()
    {
        if (!AnsiConsole.Confirm("Reset profiles/sessions directories to the default shared location?", false))
            return;

        appSettings.Current.ProfilesDirectory = Path.Combine(AppSettingsService.DefaultRootDirectory, "profiles");
        appSettings.Current.SessionsDirectory = Path.Combine(AppSettingsService.DefaultRootDirectory, "sessions");
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]Reset to:[/] {Markup.Escape(AppSettingsService.DefaultRootDirectory)}");
        Pause();
    }

    private static bool TryCreateDirectory(string dir)
    {
        try
        {
            Directory.CreateDirectory(dir);
            return true;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Could not create/use that directory:[/] {Markup.Escape(ex.Message)}");
            return false;
        }
    }

    private static void Pause()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Press any key to continue...[/]");
        Console.ReadKey(intercept: true);
    }
}
