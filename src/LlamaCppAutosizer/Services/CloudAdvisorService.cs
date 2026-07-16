using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
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
    IHttpClientFactory httpClientFactory,
    ILogger<CloudAdvisorService> logger)
{
    // A recommendation is a small JSON object; frontier models answer well within this.
    private static readonly TimeSpan CompletionTimeout = TimeSpan.FromSeconds(180);
    private static readonly TimeSpan ProbeTimeout = TimeSpan.FromSeconds(20);

    // Availability probe result, cached per configured command so a missing CLI is
    // detected once per session instead of stalling every iteration.
    private string? _probedCommand;
    private bool _probeSucceeded;

    // User-visible activity trail (wired by RecommendationService.SetActivityLog into the
    // TUI's live log panel) — call mechanics only; the recommendation decisions are logged
    // by RecommendationService.
    private Action<string>? _activityLog;
    public void SetActivityLog(Action<string>? log)
    {
        _activityLog = log;
        // A non-null handler marks the start of an optimization run — reset the running
        // usage totals so the per-run summary in each log line starts from zero.
        if (log is not null)
        {
            _runCalls = 0; _runInputTokens = 0; _runOutputTokens = 0; _runCostUsd = 0;
            _planUsageUnavailable = false;
        }
    }
    private void LogActivity(string message)
    {
        logger.LogInformation("{Message}", message);
        _activityLog?.Invoke(message);
    }

    // Cumulative usage across the current optimization run, shown after every call so
    // users always see what the cloud advisor is consuming.
    private int _runCalls;
    private long _runInputTokens;
    private long _runOutputTokens;
    private double _runCostUsd;

    // Plan-limit usage (the "/usage" numbers): sticky-disabled after the first failure so
    // a missing credential file or a changed endpoint logs nothing per-iteration. Reset at
    // run start alongside the run totals.
    private bool _planUsageUnavailable;

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
            LogActivity(_probeSucceeded
                ? $"Claude CLI available: {command} ({stdout.Trim()})"
                : $"Claude CLI probe failed (exit {exitCode}) for '{command}' — falling back to the local model");
        }
        catch (Exception ex)
        {
            LogActivity($"Claude CLI not available ('{command}': {ex.Message}) — falling back to the local model");
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
        // JSON output envelope carries the answer plus token usage and cost per call.
        var args = new List<string> { "-p", "--output-format", "json" };
        string? model = appSettings.Current.CloudAdvisorModel;
        if (!string.IsNullOrWhiteSpace(model)) { args.Add("--model"); args.Add(model.Trim()); }

        try
        {
            LogActivity($"Asking Claude ({ModelDisplay})…");
            var sw = Stopwatch.StartNew();
            // Prompt goes via stdin — recommendation prompts carry the full optimization
            // history and can exceed command-line length limits.
            var (exitCode, stdout, stderr) = await RunAsync(command, args, prompt, CompletionTimeout, ct);
            if (exitCode != 0 || string.IsNullOrWhiteSpace(stdout))
            {
                string err = stderr.Trim();
                LogActivity($"Claude CLI call failed after {sw.Elapsed.TotalSeconds:F0}s " +
                            $"(exit {exitCode}{(err.Length > 0 ? $": {(err.Length > 120 ? err[..120] : err)}" : "")})");
                return null;
            }

            var parsed = TryParseCliJson(stdout);
            if (parsed is null)
            {
                // Older CLI or unexpected shape — treat stdout as the plain answer.
                LogActivity($"Claude replied in {sw.Elapsed.TotalSeconds:F0}s (no usage info in response)");
                return stdout.Trim();
            }
            if (parsed.IsError)
            {
                LogActivity($"Claude CLI reported an error after {sw.Elapsed.TotalSeconds:F0}s: " +
                            $"{(parsed.Text.Length > 120 ? parsed.Text[..120] : parsed.Text)}");
                return null;
            }

            _runCalls++;
            _runInputTokens += parsed.InputTokens;
            _runOutputTokens += parsed.OutputTokens;
            _runCostUsd += parsed.CostUsd;

            string cached = parsed.CacheReadTokens > 0 ? $" ({parsed.CacheReadTokens:N0} cached)" : "";
            string cost = parsed.CostUsd > 0 ? $", ${parsed.CostUsd:F4}" : "";
            string runCost = _runCostUsd > 0 ? $", ${_runCostUsd:F4}" : "";
            LogActivity($"Claude replied in {sw.Elapsed.TotalSeconds:F0}s — " +
                        $"tokens in {parsed.InputTokens:N0}{cached} / out {parsed.OutputTokens:N0}{cost}  " +
                        $"[run total: {_runCalls} call{(_runCalls == 1 ? "" : "s")}, " +
                        $"{_runInputTokens:N0} in / {_runOutputTokens:N0} out{runCost}]");

            string? planUsage = await TryGetPlanUsageAsync(ct);
            if (planUsage is not null)
                LogActivity($"Plan usage: {planUsage}");

            return parsed.Text.Trim();
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw; // user cancellation propagates
        }
        catch (Exception ex)
        {
            LogActivity($"Claude CLI call failed: {ex.Message}");
            return null;
        }
    }

    // -------------------------------------------------------------------------
    // Plan-limit usage (the numbers Claude Code's /usage screen shows)
    // -------------------------------------------------------------------------
    // The CLI exposes these only in its interactive TUI, so we query the same OAuth
    // endpoint it uses, authenticated with the token Claude Code already stores locally.
    // The token is read-only here and never logged or persisted anywhere else. Undocumented
    // endpoint: parsing is defensive, and any failure sticky-disables the feature for the
    // rest of the run so nothing breaks if the shape changes.

    private static readonly (string Key, string Label)[] UsageWindows =
    [
        ("five_hour", "session"),
        ("seven_day", "weekly"),
        ("seven_day_opus", "weekly Opus"),
    ];

    private async Task<string?> TryGetPlanUsageAsync(CancellationToken ct)
    {
        if (_planUsageUnavailable) return null;
        try
        {
            string credPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".claude", ".credentials.json");
            if (!File.Exists(credPath)) { _planUsageUnavailable = true; return null; }

            using var credDoc = JsonDocument.Parse(await File.ReadAllTextAsync(credPath, ct));
            if (!credDoc.RootElement.TryGetProperty("claudeAiOauth", out var oauth) ||
                oauth.ValueKind != JsonValueKind.Object ||
                !oauth.TryGetProperty("accessToken", out var tokenEl) ||
                tokenEl.GetString() is not { Length: > 0 } token)
            {
                _planUsageUnavailable = true;
                return null;
            }

            using var http = httpClientFactory.CreateClient();
            http.Timeout = TimeSpan.FromSeconds(15);
            using var request = new HttpRequestMessage(HttpMethod.Get, "https://api.anthropic.com/api/oauth/usage");
            request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {token}");
            request.Headers.TryAddWithoutValidation("anthropic-beta", "oauth-2025-04-20");
            using var response = await http.SendAsync(request, ct);
            if (!response.IsSuccessStatusCode)
            {
                logger.LogDebug("Plan usage endpoint returned {Status}", response.StatusCode);
                _planUsageUnavailable = true;
                return null;
            }

            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
            var parts = new List<string>();
            foreach (var (key, label) in UsageWindows)
            {
                if (doc.RootElement.ValueKind != JsonValueKind.Object ||
                    !doc.RootElement.TryGetProperty(key, out var window) ||
                    window.ValueKind != JsonValueKind.Object ||
                    !window.TryGetProperty("utilization", out var util) ||
                    util.ValueKind != JsonValueKind.Number)
                    continue;

                double pct = util.GetDouble();
                if (key == "seven_day_opus" && pct <= 0) continue;   // skip a quiet Opus window

                string reset = "";
                if (window.TryGetProperty("resets_at", out var resetsAt) &&
                    resetsAt.ValueKind == JsonValueKind.String &&
                    DateTimeOffset.TryParse(resetsAt.GetString(), out var at))
                    reset = $" (resets in {FormatDelta(at - DateTimeOffset.UtcNow)})";

                parts.Add($"{label} {pct:F0}% used{reset}");
            }

            if (parts.Count == 0) { _planUsageUnavailable = true; return null; }
            return string.Join(" / ", parts);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            logger.LogDebug("Plan usage lookup failed: {Msg}", ex.Message);
            _planUsageUnavailable = true;
            return null;
        }
    }

    private static string FormatDelta(TimeSpan delta)
    {
        if (delta <= TimeSpan.Zero) return "now";
        if (delta.TotalDays >= 1) return $"{(int)delta.TotalDays}d {delta.Hours}h";
        if (delta.TotalHours >= 1) return $"{(int)delta.TotalHours}h {delta.Minutes}m";
        return $"{Math.Max(1, delta.Minutes)}m";
    }

    // Parsed `claude -p --output-format json` result envelope. InputTokens includes cache
    // reads/writes so the number reflects what the call actually consumed context-wise.
    private sealed record CliResult(
        string Text, long InputTokens, long CacheReadTokens, long OutputTokens, double CostUsd, bool IsError);

    private static CliResult? TryParseCliJson(string stdout)
    {
        try
        {
            using var doc = JsonDocument.Parse(stdout);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return null;
            if (root.TryGetProperty("result", out var r) is false || r.ValueKind != JsonValueKind.String)
                return null;

            bool isError = root.TryGetProperty("is_error", out var e) && e.ValueKind == JsonValueKind.True;
            double cost = root.TryGetProperty("total_cost_usd", out var c) && c.ValueKind == JsonValueKind.Number
                ? c.GetDouble() : 0;

            long inTok = 0, outTok = 0, cacheRead = 0;
            if (root.TryGetProperty("usage", out var u) && u.ValueKind == JsonValueKind.Object)
            {
                cacheRead = GetLong(u, "cache_read_input_tokens");
                inTok = GetLong(u, "input_tokens") + cacheRead + GetLong(u, "cache_creation_input_tokens");
                outTok = GetLong(u, "output_tokens");
            }

            return new CliResult(r.GetString() ?? "", inTok, cacheRead, outTok, cost, isError);
        }
        catch (JsonException)
        {
            return null;
        }

        static long GetLong(JsonElement obj, string name) =>
            obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : 0;
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
