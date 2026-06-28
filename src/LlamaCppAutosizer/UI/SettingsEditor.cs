using LlamaCppAutosizer.Models;
using Spectre.Console;

namespace LlamaCppAutosizer.UI;

public static class SettingsEditor
{
    /// <summary>
    /// Interactively lets the user review and override any LlamaSettings field.
    /// Returns the (possibly modified) settings.
    /// </summary>
    public static LlamaSettings Edit(LlamaSettings settings, string title = "Edit Settings")
    {
        var s = settings.Clone();

        while (true)
        {
            AnsiConsole.Clear();
            AnsiConsole.Write(new Rule($"[bold yellow]{title}[/]").RuleStyle("yellow"));
            AnsiConsole.WriteLine();

            RenderSettings(s);
            AnsiConsole.WriteLine();

            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[grey]Select a setting to edit, or confirm:[/]")
                    .PageSize(20)
                    .AddChoices(
                        "-- Done (use these settings) --",
                        "Context Size",
                        "GPU Layers (-1 = all)",
                        "Batch Size",
                        "Micro-Batch Size (ubatch)",
                        "CPU Threads (-1 = auto)",
                        "CPU Threads for Batch (-1 = auto)",
                        "Flash Attention",
                        "Memory-map model (mmap)",
                        "Lock model in RAM (mlock)",
                        "KV Cache Type K",
                        "KV Cache Type V",
                        "Parallel Slots",
                        "Rope Scaling",
                        "Extra CLI Args (raw)",
                        "-- Reset to defaults --"
                    ));

            if (choice.StartsWith("-- Done")) break;
            if (choice.StartsWith("-- Reset"))
            {
                s = new LlamaSettings();
                continue;
            }

            switch (choice)
            {
                case "Context Size":
                    s.ContextSize = PromptInt("Context size (tokens, power of 2 recommended)", s.ContextSize, 512, 524288);
                    break;
                case "GPU Layers (-1 = all)":
                    s.GpuLayers = PromptInt("Number of GPU layers (-1 = offload all)", s.GpuLayers, -1, 200);
                    break;
                case "Batch Size":
                    s.BatchSize = PromptInt("Batch size (prompt processing chunk size)", s.BatchSize, 32, 4096);
                    break;
                case "Micro-Batch Size (ubatch)":
                    s.UBatchSize = PromptInt("Micro-batch size (ubatch)", s.UBatchSize, 32, 4096);
                    break;
                case "CPU Threads (-1 = auto)":
                    s.Threads = PromptInt("CPU threads (-1 = auto)", s.Threads, -1, 256);
                    break;
                case "CPU Threads for Batch (-1 = auto)":
                    s.ThreadsBatch = PromptInt("CPU threads for batch (-1 = auto)", s.ThreadsBatch, -1, 256);
                    break;
                case "Flash Attention":
                    s.FlashAttention = AnsiConsole.Confirm("Enable flash attention?", s.FlashAttention);
                    break;
                case "Memory-map model (mmap)":
                    s.Mmap = AnsiConsole.Confirm("Enable mmap?", s.Mmap);
                    break;
                case "Lock model in RAM (mlock)":
                    s.Mlock = AnsiConsole.Confirm("Lock model in RAM (mlock)?", s.Mlock);
                    break;
                case "KV Cache Type K":
                    s.CacheTypeK = PromptEnum("KV Cache type K",
                        s.CacheTypeK ?? "f16", LlamaSettings.ValidCacheTypes);
                    if (s.CacheTypeK == "f16") s.CacheTypeK = null;
                    break;
                case "KV Cache Type V":
                    s.CacheTypeV = PromptEnum("KV Cache type V",
                        s.CacheTypeV ?? "f16", LlamaSettings.ValidCacheTypes);
                    if (s.CacheTypeV == "f16") s.CacheTypeV = null;
                    break;
                case "Parallel Slots":
                    s.ParallelSlots = PromptInt("Parallel slots (concurrent requests)", s.ParallelSlots, 1, 16);
                    break;
                case "Rope Scaling":
                    s.RopeScaling = PromptOptionalString("RoPE scaling type (none/linear/yarn, empty to clear)",
                        s.RopeScaling);
                    break;
                case "Extra CLI Args (raw)":
                    s.ExtraArgs = PromptOptionalString(
                        "Extra raw CLI args to pass to llama-server (space-separated, empty to clear)",
                        s.ExtraArgs);
                    break;
            }
        }

        return s;
    }

    /// <summary>Displays a read-only summary of the settings in a formatted table.</summary>
    public static void RenderSettings(LlamaSettings s)
    {
        var table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        table.AddColumn("[grey]Parameter[/]");
        table.AddColumn("[grey]Value[/]");
        table.AddColumn("[grey]Notes[/]");

        table.AddRow("ctx-size", $"[cyan]{s.ContextSize:N0}[/]", "tokens");
        table.AddRow("n-gpu-layers", $"[cyan]{(s.GpuLayers == -1 ? "all" : s.GpuLayers)}[/]", "GPU layer offload");
        table.AddRow("batch-size", $"[cyan]{s.BatchSize}[/]", "PP chunk size");
        table.AddRow("ubatch-size", $"[cyan]{s.UBatchSize}[/]", "micro-batch");
        table.AddRow("threads", $"[cyan]{(s.Threads == -1 ? "auto" : s.Threads)}[/]", "CPU threads");
        table.AddRow("threads-batch", $"[cyan]{(s.ThreadsBatch == -1 ? "auto" : s.ThreadsBatch)}[/]", "batch threads");
        table.AddRow("flash-attn", s.FlashAttention ? "[green]yes[/]" : "[grey]no[/]", "Flash Attention 2");
        table.AddRow("mmap", s.Mmap ? "[green]yes[/]" : "[grey]no[/]", "memory map model");
        table.AddRow("mlock", s.Mlock ? "[green]yes[/]" : "[grey]no[/]", "lock in RAM");
        table.AddRow("cache-type-k", $"[cyan]{s.CacheTypeK ?? "f16"}[/]", "KV cache K type");
        table.AddRow("cache-type-v", $"[cyan]{s.CacheTypeV ?? "f16"}[/]", "KV cache V type");
        table.AddRow("parallel", $"[cyan]{s.ParallelSlots}[/]", "concurrent slots");

        if (s.RopeScaling is not null)
            table.AddRow("rope-scaling", $"[cyan]{s.RopeScaling}[/]", "");
        if (!string.IsNullOrWhiteSpace(s.ExtraArgs))
            table.AddRow("extra-args", $"[yellow]{s.ExtraArgs}[/]", "user-provided");

        AnsiConsole.Write(table);
    }

    // -------------------------------------------------------------------------
    // Prompt helpers
    // -------------------------------------------------------------------------

    private static int PromptInt(string label, int current, int min, int max)
    {
        return AnsiConsole.Prompt(
            new TextPrompt<int>($"[grey]{label}[/] [[{min}–{max}]]:")
                .DefaultValue(current)
                .Validate(v => v >= min && v <= max
                    ? ValidationResult.Success()
                    : ValidationResult.Error($"Must be between {min} and {max}")));
    }

    private static string PromptEnum(string label, string current, string[] options)
    {
        return AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title($"[grey]{label}[/] (current: [cyan]{current}[/])")
                .AddChoices(options));
    }

    private static string? PromptOptionalString(string label, string? current)
    {
        var result = AnsiConsole.Prompt(
            new TextPrompt<string>($"[grey]{label}[/]:")
                .DefaultValue(current ?? "")
                .AllowEmpty());
        return string.IsNullOrWhiteSpace(result) ? null : result;
    }
}
