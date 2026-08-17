using LlamaCppAutosizer.Models;
using LlamaCppAutosizer.Services;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

// File-path-related app configuration: model folder, llama-server executable, where named
// profiles / optimization history (sessions) are stored, and TurboQuant setup. Distinct from
// "Edit llama.cpp Settings Manually", which edits per-run model/server tuning parameters.
public class AppSettingsMenu(AppSettingsService appSettings, TurboQuantService turboQuant, CloudAdvisorService cloudAdvisor)
{
    /// <summary>
    /// Runs the Settings menu. <paramref name="manualSettings"/> is the caller's current
    /// LlamaSettings (mutated in place by "Configure TurboQuant cache types" — it's a
    /// reference type, so changes are visible to the caller without any return plumbing).
    /// Returns a new model path if TurboQuant quantization switched the active model, or
    /// null if nothing changed.
    /// </summary>
    public async Task<string?> RunAsync(LlamaSettings manualSettings, string currentModelPath, CancellationToken ct = default)
    {
        string? updatedModelPath = null;

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
                    "TurboQuant Options  (KV cache types, quantize a model)",
                    "Cloud Advisor  (Claude CLI as the optimizer's recommender)",
                    "Reset directories to default shared location",
                ]);

            if (choice is null) return updatedModelPath; // Esc

            if (choice.StartsWith("Model Folder")) EditModelFolder();
            else if (choice.StartsWith("llama-server")) EditServerExecutable();
            else if (choice.StartsWith("Profiles")) EditProfilesDirectory();
            else if (choice.StartsWith("Sessions")) EditSessionsDirectory();
            else if (choice.StartsWith("Cloud Advisor")) await EditCloudAdvisorAsync(ct);
            else if (choice.StartsWith("TurboQuant"))
            {
                string? result = await TurboQuantSubmenuAsync(manualSettings, currentModelPath, ct);
                if (result is not null)
                {
                    currentModelPath = result;
                    updatedModelPath = result;
                }
            }
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
        table.AddRow("[grey]TurboQuant server[/]", string.IsNullOrWhiteSpace(s.TurboQuantServerExecutable)
            ? "[yellow]not set up — turbo4/turbo3/etc. cache types hidden until configured (TurboQuant Options)[/]"
            : $"[cyan]{Markup.Escape(s.TurboQuantServerExecutable)}[/] [grey](overrides llama-server path above)[/]");
        table.AddRow("[grey]Cloud advisor[/]", string.IsNullOrWhiteSpace(s.CloudAdvisorCommand)
            ? "[yellow]disabled — the local model recommends its own tuning steps[/]"
            : $"[cyan]{Markup.Escape(s.CloudAdvisorCommand)}[/] [grey](model: {Markup.Escape(cloudAdvisor.ModelDisplay)})[/]");

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

    private async Task EditCloudAdvisorAsync(CancellationToken ct)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold cyan]Cloud Advisor[/]").RuleStyle("cyan"));
        AnsiConsole.MarkupLine("[grey]Uses the Claude Code CLI as the optimizer's recommender instead of the local model[/]");
        AnsiConsole.MarkupLine("[grey]under test. A frontier model reasons over the run history far better than a heavily[/]");
        AnsiConsole.MarkupLine("[grey]quantized local model advising on its own tuning. Falls back to the local LLM, then[/]");
        AnsiConsole.MarkupLine("[grey]heuristics, whenever the CLI is unavailable. Uses your existing Claude login.[/]");
        AnsiConsole.WriteLine();

        string current = appSettings.Current.CloudAdvisorCommand ?? "";
        string command = AnsiConsole.Prompt(
            new TextPrompt<string>("Claude CLI command or full path (empty to disable):")
                .DefaultValue(string.IsNullOrWhiteSpace(current) ? "claude" : current)
                .AllowEmpty());

        if (string.IsNullOrWhiteSpace(command))
        {
            appSettings.Current.CloudAdvisorCommand = null;
            appSettings.Save();
            AnsiConsole.MarkupLine("[grey]Cloud advisor disabled — recommendations come from the local model again.[/]");
            Pause();
            return;
        }

        string? modelChoice = MenuHelper.Select(
            "Model for recommendations  [dim](Esc = CLI default)[/]",
            [
                "claude-opus-4-8  (recommended — strongest reasoning)",
                "claude-sonnet-5  (faster, cheaper)",
                "claude-haiku-4-5  (fastest)",
                "CLI default model",
                "Custom…",
            ]);

        string? model = modelChoice switch
        {
            null => null,
            var c when c.StartsWith("CLI default") => null,
            var c when c.StartsWith("Custom") =>
                AnsiConsole.Prompt(new TextPrompt<string>("Model name/alias to pass via --model:").AllowEmpty()) is { Length: > 0 } m ? m : null,
            var c => c.Split(' ')[0],
        };

        appSettings.Current.CloudAdvisorCommand = command.Trim();
        appSettings.Current.CloudAdvisorModel = model;
        appSettings.Save();

        AnsiConsole.MarkupLine($"[green]Cloud advisor set:[/] {Markup.Escape(command)} [grey](model: {Markup.Escape(model ?? "CLI default")})[/]");
        bool available = await AnsiConsole.Status()
            .StartAsync("Checking CLI availability…", _ => cloudAdvisor.IsAvailableAsync(ct));
        AnsiConsole.MarkupLine(available
            ? "[green]CLI responded — cloud recommendations will be used on the next run.[/]"
            : "[yellow]CLI did not respond to --version — the optimizer will silently fall back to the local model. " +
              "Check that Claude Code is installed and the command/path is correct.[/]");
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

    // -------------------------------------------------------------------------
    // TurboQuant
    // -------------------------------------------------------------------------

    private async Task<string?> TurboQuantSubmenuAsync(
        LlamaSettings manualSettings, string currentModelPath, CancellationToken ct)
    {
        string? updatedModelPath = null;

        while (true)
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule("[bold magenta]TurboQuant[/]").RuleStyle("magenta"));
            AnsiConsole.MarkupLine("[grey]https://github.com/TheTom/llama-cpp-turboquant[/]");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine(appSettings.TurboQuantAvailable
                ? $"[grey]TurboQuant server:[/] [cyan]{Markup.Escape(appSettings.Current.TurboQuantServerExecutable!)}[/]"
                : "[yellow]TurboQuant server: not set up[/]");
            AnsiConsole.WriteLine();

            var choices = new List<string>
            {
                "Set TurboQuant llama-server.exe path  (overrides main llama-server path)",
            };
            if (appSettings.TurboQuantAvailable)
            {
                choices.Add("Configure TurboQuant cache types  (turbo4 / turbo3 / …)");
                choices.Add("Quantize a model");
                choices.Add("Quantize current model and compare");
            }

            string? choice = MenuHelper.Select("[bold magenta]TurboQuant Options[/]  [dim](Esc = back)[/]", choices);
            if (choice is null) return updatedModelPath; // Esc

            if (choice.StartsWith("Set TurboQuant"))
            {
                EditTurboQuantServerExecutable();
            }
            else if (choice.StartsWith("Configure"))
            {
                ConfigureTurboQuantCacheTypes(manualSettings);
            }
            else
            {
                string? result = await QuantizeAsync(choice, currentModelPath, ct);
                if (result is not null)
                {
                    currentModelPath = result;
                    updatedModelPath = result;
                }
            }
        }
    }

    private void EditTurboQuantServerExecutable()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold magenta]TurboQuant llama-server Path[/]").RuleStyle("magenta"));
        AnsiConsole.MarkupLine("[grey]TurboQuant ships its own llama-server build — point this at that executable, or the folder containing it.[/]");
        AnsiConsole.MarkupLine("[grey]Once set, it overrides the main llama-server Executable Path for every run.[/]");
        AnsiConsole.WriteLine();

        string current = appSettings.Current.TurboQuantServerExecutable ?? "";
        string userInput = AnsiConsole.Prompt(
            new TextPrompt<string>("Folder or path to the TurboQuant llama-server (empty to clear override):")
                .DefaultValue(current)
                .AllowEmpty());

        if (string.IsNullOrWhiteSpace(userInput))
        {
            appSettings.Current.TurboQuantServerExecutable = null;
            appSettings.Save();
            AnsiConsole.MarkupLine("[grey]Override cleared — the main llama-server path will be used, and turbo cache types are hidden again.[/]");
            Pause();
            return;
        }

        string? resolved = turboQuant.ResolveFromUserInput(userInput);
        if (resolved is null || !turboQuant.IsAvailable(resolved))
        {
            AnsiConsole.MarkupLine($"[red]llama-server not found in:[/] {Markup.Escape(userInput)}");
            AnsiConsole.MarkupLine("[grey]TurboQuant ships its own llama-server.exe. Point to the folder that contains it, or the exe directly.[/]");
            Pause();
            return;
        }

        appSettings.Current.TurboQuantServerExecutable = resolved;
        appSettings.Save();
        AnsiConsole.MarkupLine($"[green]TurboQuant server set:[/] {Markup.Escape(resolved)}");
        AnsiConsole.MarkupLine("[grey]This now overrides the main llama-server path for all runs.[/]");
        Pause();
    }

    private void ConfigureTurboQuantCacheTypes(LlamaSettings manualSettings)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[bold magenta]TurboQuant KV Cache Types[/]").RuleStyle("magenta"));
        AnsiConsole.MarkupLine("[grey]These types are only supported by the TurboQuant fork of llama-server.[/]");
        AnsiConsole.MarkupLine("[grey]Typical recommendation: --cache-type-k turbo4 --cache-type-v turbo3[/]");
        AnsiConsole.WriteLine();

        var allK = LlamaSettings.ValidCacheTypes.Concat(LlamaSettings.TurboQuantCacheTypes).ToArray();
        var allV = allK;

        string currentK = manualSettings.CacheTypeK ?? "f16";
        string currentV = manualSettings.CacheTypeV ?? "f16";

        AnsiConsole.MarkupLine($"[grey]Current cache-type-k:[/] [cyan]{currentK}[/]");
        string? newK = MenuHelper.Select(
            "Select [bold]cache-type-k[/]  [dim](Esc = keep current)[/]",
            allK);
        if (newK is not null) manualSettings.CacheTypeK = newK == "f16" ? null : newK;

        AnsiConsole.MarkupLine($"[grey]Current cache-type-v:[/] [cyan]{currentV}[/]");
        string? newV = MenuHelper.Select(
            "Select [bold]cache-type-v[/]  [dim](Esc = keep current)[/]",
            allV);
        if (newV is not null) manualSettings.CacheTypeV = newV == "f16" ? null : newV;

        string finalK = manualSettings.CacheTypeK ?? "f16";
        string finalV = manualSettings.CacheTypeV ?? "f16";

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[green]Set:[/] --cache-type-k [cyan]{finalK}[/]  --cache-type-v [cyan]{finalV}[/]");
        AnsiConsole.MarkupLine("[grey]Will be used as the starting cache type on your next optimization run, unless you start from a saved profile.[/]");
        Pause();
    }

    private async Task<string?> QuantizeAsync(string choice, string currentModelPath, CancellationToken ct)
    {
        string inputModel = choice.Contains("current") && !string.IsNullOrEmpty(currentModelPath)
            ? currentModelPath
            : AnsiConsole.Ask<string>("Path to model to quantize:", currentModelPath);

        if (!File.Exists(inputModel))
        {
            AnsiConsole.MarkupLine("[red]File not found.[/]");
            Pause();
            return null;
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
                    new Progress<string>(msg => task.Description = $"[magenta]{(msg.Length <= 60 ? msg : msg[..60] + "…")}[/]"),
                    ct);
            });

        AnsiConsole.WriteLine();
        string? newModelPath = null;
        if (result.Success)
        {
            AnsiConsole.MarkupLine($"[green]Quantization complete![/]");
            AnsiConsole.MarkupLine($"Output: [cyan]{Markup.Escape(result.OutputPath)}[/]");
            AnsiConsole.MarkupLine($"Size: [cyan]{result.OriginalSizeMb:N0} MB → {result.QuantizedSizeMb:N0} MB[/] " +
                $"([green]{(1 - result.CompressionRatio) * 100:F1}% smaller[/])");
            AnsiConsole.MarkupLine($"Duration: [grey]{result.Duration:mm\\:ss}[/]");

            if (choice.Contains("compare") && AnsiConsole.Confirm("Use this quantized model as your active model now?", true))
            {
                newModelPath = result.OutputPath;
                AnsiConsole.MarkupLine("[green]Model path updated to quantized model.[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine($"[red]Quantization failed:[/] {Markup.Escape(result.ErrorMessage ?? "unknown error")}");
        }

        Pause();
        return newModelPath;
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
