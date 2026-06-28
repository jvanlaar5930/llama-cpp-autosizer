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

        // If the user provided a path, resolve it to something Windows can actually start.
        // Without an extension Process.Start throws "Access is denied" on Windows.
        if (hint is not null)
        {
            var resolved = ResolveWindowsExecutable(hint);
            candidates.Add(resolved ?? hint);
        }

        // No dedicated turboquant binary — it ships as a custom llama-server build.
        // If a hint was given and resolved above, nothing else to try here.
        // We still check if the user has a turboquant-branded llama-server on PATH.
        candidates.AddRange([
            "llama-server",
            "llama-server.exe",
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
    // Resolve a user-provided string that may be a folder OR a direct exe path
    // -------------------------------------------------------------------------

    // llama-cpp-turboquant ships a custom llama-server build, not a separate binary.
    // When the user points to a folder, we look for that server executable inside it.
    private static readonly string[] KnownExeNames =
    [
        "llama-server.exe",
        "llama-server",
    ];

    /// <summary>
    /// Accepts either:
    ///   • A direct path to the executable (or a path with no extension on Windows)
    ///   • A folder — scans for any known turboquant executable names inside it
    /// Returns the resolved executable string, or null if nothing usable was found.
    /// </summary>
    public string? ResolveFromUserInput(string input)
    {
        input = Environment.ExpandEnvironmentVariables(input.Trim());

        // Direct file path — hand off to the normal resolver
        if (File.Exists(input) || HasExecutableExtension(input))
            return ResolveWindowsExecutable(input);

        // User gave a folder — scan for known executable names
        if (Directory.Exists(input))
        {
            foreach (var name in KnownExeNames)
            {
                var candidate = Path.Combine(input, name);
                if (File.Exists(candidate))
                    return candidate;
            }
            // Nothing found inside the folder
            return null;
        }

        // Neither file nor folder — maybe a PATH command, try resolving extensions
        return ResolveWindowsExecutable(input);
    }

    // -------------------------------------------------------------------------
    // Resolve a user-supplied path to a runnable form on Windows
    // -------------------------------------------------------------------------

    // Windows won't execute files with no extension (throws "Access is denied").
    // Try appending .exe / .bat / .cmd, detect .py files, and fall back to cmd /c.
    private static string? ResolveWindowsExecutable(string path)
    {
        path = Environment.ExpandEnvironmentVariables(path.Trim());

        // Python script — delegate to interpreter
        if (path.EndsWith(".py", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
            return $"python \"{path}\"";

        // File exists with a known executable extension — use as-is
        if (File.Exists(path) && HasExecutableExtension(path))
            return path;

        // No recognised extension — try appending Windows executable extensions
        if (!HasExecutableExtension(path))
        {
            foreach (var ext in new[] { ".exe", ".bat", ".cmd" })
            {
                if (File.Exists(path + ext))
                    return path + ext;
            }
        }

        // File exists but has an unrecognised extension — let cmd.exe try to run it
        if (File.Exists(path))
            return $"cmd /c \"{path}\"";

        // Not a local file path — treat as a PATH command name (e.g. "turboquant")
        return path;
    }

    private static bool HasExecutableExtension(string path)
    {
        var ext = Path.GetExtension(path);
        return ext.Equals(".exe", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".bat", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".cmd", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".py",  StringComparison.OrdinalIgnoreCase);
    }

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
        // "cmd /c ..." and "python ..." style commands need the whole string parsed
        var (fileName, arguments) = SplitCommand(exe, args);

        var psi = new ProcessStartInfo(fileName, arguments)
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

    // Split "cmd /c \"path\"" or "python \"path\"" into (fileName, rest + userArgs)
    private static (string fileName, string arguments) SplitCommand(string exe, string args)
    {
        if (exe.StartsWith("cmd /c ", StringComparison.OrdinalIgnoreCase))
            return ("cmd", $"/c {exe[7..]} {args}");

        if (exe.StartsWith("python ", StringComparison.OrdinalIgnoreCase))
            return ("python", $"{exe[7..]} {args}");

        return (exe, args);
    }

    private static bool IsExecutableAvailable(string exe)
    {
        try
        {
            var (fileName, arguments) = SplitCommand(exe, "--help");

            // For "python -m X" style already encoded in exe
            if (exe.StartsWith("python -m ", StringComparison.OrdinalIgnoreCase))
            {
                fileName = "python";
                arguments = $"-m {exe[10..]} --help";
            }

            var psi = new ProcessStartInfo(fileName, arguments)
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
