using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

/// <summary>
/// Optional stronger recommender: sends the optimizer's recommendation prompts to a frontier
/// model via the Claude Code CLI (`claude -p`) instead of the local model under test. The
/// local model is self-referential — an aggressively quantized model advising on its own
/// tuning produces weak or unparseable suggestions exactly when tuning matters most — and
/// the CLI path also works while the local server is down (e.g. after a failed start).
/// Uses the user's existing Claude subscription/login; no API key is stored by this app.
/// Not configured (empty command) = disabled: callers fall back to the local LLM.
/// </summary>
public class CloudAdvisorService(
    AppSettingsService appSettings,
    ILogger<CloudAdvisorService> logger)
{
    // A recommendation is a small JSON object; frontier models answer well within this.
    private static readonly TimeSpan CompletionTimeout = TimeSpan.FromSeconds(180);
    private static readonly TimeSpan ProbeTimeout = TimeSpan.FromSeconds(20);

    // Availability probe result, cached per configured command so a missing CLI is
    // detected once per session instead of stalling every iteration.
    private string? _probedCommand;
    private bool _probeSucceeded;

    public bool IsConfigured => !string.IsNullOrWhiteSpace(appSettings.Current.CloudAdvisorCommand);

    public string ModelDisplay => string.IsNullOrWhiteSpace(appSettings.Current.CloudAdvisorModel)
        ? "CLI default"
        : appSettings.Current.CloudAdvisorModel!;

    /// <summary>
    /// True if the configured CLI command responds to <c>--version</c>. Cached per command.
    /// </summary>
    public async Task<bool> IsAvailableAsync(CancellationToken ct = default)
    {
        if (!IsConfigured) return false;
        string command = appSettings.Current.CloudAdvisorCommand!.Trim();
        if (command == _probedCommand) return _probeSucceeded;

        _probedCommand = command;
        _probeSucceeded = false;
        try
        {
            var (exitCode, stdout, _) = await RunAsync(command, ["--version"], stdin: null, ProbeTimeout, ct);
            _probeSucceeded = exitCode == 0;
            if (_probeSucceeded)
                logger.LogInformation("Cloud advisor available: {Command} ({Version})", command, stdout.Trim());
            else
                logger.LogInformation("Cloud advisor probe failed (exit {Code}) for: {Command}", exitCode, command);
        }
        catch (Exception ex)
        {
            logger.LogInformation("Cloud advisor not available ({Command}): {Msg}", command, ex.Message);
        }
        return _probeSucceeded;
    }

    /// <summary>
    /// Sends <paramref name="prompt"/> to the configured Claude CLI in print mode and returns
    /// the raw text response, or null when unconfigured/unavailable/failed — callers treat
    /// null as "fall back to the local LLM".
    /// </summary>
    public async Task<string?> CompleteAsync(string prompt, CancellationToken ct = default)
    {
        if (!await IsAvailableAsync(ct)) return null;

        string command = appSettings.Current.CloudAdvisorCommand!.Trim();
        var args = new List<string> { "-p" };
        string? model = appSettings.Current.CloudAdvisorModel;
        if (!string.IsNullOrWhiteSpace(model)) { args.Add("--model"); args.Add(model.Trim()); }

        try
        {
            // Prompt goes via stdin — recommendation prompts carry the full optimization
            // history and can exceed command-line length limits.
            var (exitCode, stdout, stderr) = await RunAsync(command, args, prompt, CompletionTimeout, ct);
            if (exitCode != 0 || string.IsNullOrWhiteSpace(stdout))
            {
                logger.LogInformation("Cloud advisor call failed (exit {Code}): {Err}",
                    exitCode, stderr.Length > 200 ? stderr[..200] : stderr);
                return null;
            }
            return stdout.Trim();
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw; // user cancellation propagates
        }
        catch (Exception ex)
        {
            logger.LogInformation("Cloud advisor call failed: {Msg}", ex.Message);
            return null;
        }
    }

    private static async Task<(int exitCode, string stdout, string stderr)> RunAsync(
        string command, IEnumerable<string> args, string? stdin, TimeSpan timeout, CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            UseShellExecute = false,
            RedirectStandardInput = stdin is not null,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        // On Windows, npm-installed CLIs are .cmd shims that CreateProcess can't launch
        // directly by bare name — route bare commands through cmd.exe. Explicit paths
        // (containing a separator or extension) are launched directly on both platforms.
        bool bareCommand = !command.Contains(Path.DirectorySeparatorChar)
                        && !command.Contains(Path.AltDirectorySeparatorChar)
                        && !Path.HasExtension(command);
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && bareCommand)
        {
            psi.FileName = "cmd.exe";
            psi.ArgumentList.Add("/c");
            psi.ArgumentList.Add(command);
        }
        else
        {
            psi.FileName = command;
        }
        foreach (var a in args) psi.ArgumentList.Add(a);

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException($"Failed to start: {command}");

        if (stdin is not null)
        {
            await process.StandardInput.WriteAsync(stdin.AsMemory(), ct);
            process.StandardInput.Close();
        }

        var stdoutTask = process.StandardOutput.ReadToEndAsync(ct);
        var stderrTask = process.StandardError.ReadToEndAsync(ct);

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(timeout);
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            if (ct.IsCancellationRequested) throw;
            throw new TimeoutException($"{command} did not respond within {timeout.TotalSeconds:F0}s");
        }

        return (process.ExitCode, await stdoutTask, await stderrTask);
    }
}
