using System.Diagnostics;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class TurboQuantService(ILogger<TurboQuantService> logger)
{
    private string? _cachedExecutable;

    // -------------------------------------------------------------------------
    // Availability check
    // -------------------------------------------------------------------------

    public string? FindExecutable(string? hint = null)
    {
        if (_cachedExecutable is not null) return _cachedExecutable;

        var candidates = new List<string>();
        if (hint is not null) candidates.Add(hint);
        candidates.AddRange([
            "turboquant",
            "turboquant.exe",
            "llama-turboquant",
            "llama-turboquant.exe",
            "python -m turboquant",   // pip-installed
        ]);

        foreach (var candidate in candidates)
        {
            if (IsExecutableAvailable(candidate))
            {
                _cachedExecutable = candidate;
                return candidate;
            }
        }
        return null;
    }

    public bool IsAvailable(string? executableHint = null)
        => FindExecutable(executableHint) is not null;

    // -------------------------------------------------------------------------
    // Quantization
    // -------------------------------------------------------------------------

    public async Task<TurboQuantResult> QuantizeAsync(
        TurboQuantOptions options,
        IProgress<string>? progress = null,
        CancellationToken ct = default)
    {
        var exe = FindExecutable(options.TurboQuantExecutable)
            ?? throw new InvalidOperationException(
                "turboquant executable not found. Install from https://github.com/TheTom/llama-cpp-turboquant");

        var outputDir = string.IsNullOrEmpty(options.OutputDirectory)
            ? Path.GetDirectoryName(options.InputModelPath)!
            : options.OutputDirectory;
        Directory.CreateDirectory(outputDir);

        var originalSize = new FileInfo(options.InputModelPath).Length / (1024 * 1024);
        var args = BuildArgs(options);
        var sw = Stopwatch.StartNew();

        logger.LogInformation("Running turboquant: {Exe} {Args}", exe, args);
        progress?.Report($"Starting quantization ({options.QuantType})...");

        try
        {
            var output = await RunWithProgressAsync(exe, args, progress, ct);
            sw.Stop();

            if (!File.Exists(options.OutputModelPath))
            {
                return new TurboQuantResult
                {
                    Success = false,
                    ErrorMessage = $"Output file not found at {options.OutputModelPath}. Output:\n{output[..Math.Min(500, output.Length)]}",
                    Duration = sw.Elapsed,
                };
            }

            long quantizedSize = new FileInfo(options.OutputModelPath).Length / (1024 * 1024);

            return new TurboQuantResult
            {
                Success = true,
                OutputPath = options.OutputModelPath,
                OriginalSizeMb = originalSize,
                QuantizedSizeMb = quantizedSize,
                Duration = sw.Elapsed,
            };
        }
        catch (OperationCanceledException)
        {
            return new TurboQuantResult { Success = false, ErrorMessage = "Cancelled", Duration = sw.Elapsed };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "turboquant failed");
            return new TurboQuantResult { Success = false, ErrorMessage = ex.Message, Duration = sw.Elapsed };
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static string BuildArgs(TurboQuantOptions options)
    {
        var quantStr = options.QuantType.ToString().ToUpperInvariant().Replace('_', '-');
        // turboquant CLI signature (based on TheTom/llama-cpp-turboquant README):
        // turboquant --input <model.gguf> --output <out.gguf> --type <QUANT_TYPE>
        //            [--importance] [--calibration-data <path>] [--samples <N>]
        var args = new List<string>
        {
            "--input", $"\"{options.InputModelPath}\"",
            "--output", $"\"{options.OutputModelPath}\"",
            "--type", quantStr,
        };

        if (options.UseImportance) args.Add("--importance");
        if (options.CalibrationDataset is not null)
        {
            args.Add("--calibration-data");
            args.Add($"\"{options.CalibrationDataset}\"");
        }
        args.Add("--samples");
        args.Add(options.CalibrationSamples.ToString());

        return string.Join(" ", args);
    }

    private static async Task<string> RunWithProgressAsync(
        string exe, string args, IProgress<string>? progress, CancellationToken ct)
    {
        var psi = new ProcessStartInfo(exe, args)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start turboquant");

        var sb = new System.Text.StringBuilder();

        proc.OutputDataReceived += (_, e) =>
        {
            if (e.Data is null) return;
            sb.AppendLine(e.Data);
            progress?.Report(e.Data);
        };
        proc.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null) return;
            sb.AppendLine(e.Data);
            progress?.Report(e.Data);
        };
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        await proc.WaitForExitAsync(ct);
        return sb.ToString();
    }

    private static bool IsExecutableAvailable(string exe)
    {
        try
        {
            // Handle "python -m X" style
            string actualExe = exe.StartsWith("python") ? "python" : exe;
            string actualArgs = exe.StartsWith("python") ? $"-m {exe[(exe.IndexOf(' ') + 1)..]} --help" : "--help";

            var psi = new ProcessStartInfo(actualExe, actualArgs)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi);
            proc?.Kill();
            return true;
        }
        catch
        {
            return false;
        }
    }
}
