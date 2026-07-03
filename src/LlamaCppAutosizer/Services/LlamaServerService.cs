using System.Diagnostics;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

// -------------------------------------------------------------------------
// API DTOs
// -------------------------------------------------------------------------

public class CompletionRequest
{
    [JsonPropertyName("prompt")] public string Prompt { get; init; } = "";
    [JsonPropertyName("n_predict")] public int NPredict { get; init; } = 256;
    [JsonPropertyName("temperature")] public float Temperature { get; init; } = 0.1f;
    [JsonPropertyName("top_p")] public float TopP { get; init; } = 0.9f;
    [JsonPropertyName("stop")] public string[]? Stop { get; init; }
    [JsonPropertyName("stream")] public bool Stream { get; init; } = false;
    [JsonPropertyName("cache_prompt")] public bool CachePrompt { get; init; } = false;
}

public class CompletionTimings
{
    [JsonPropertyName("prompt_n")] public double PromptN { get; init; }
    [JsonPropertyName("prompt_ms")] public double PromptMs { get; init; }
    [JsonPropertyName("prompt_per_second")] public double PromptPerSecond { get; init; }
    [JsonPropertyName("predicted_n")] public double PredictedN { get; init; }
    [JsonPropertyName("predicted_ms")] public double PredictedMs { get; init; }
    [JsonPropertyName("predicted_per_second")] public double PredictedPerSecond { get; init; }
    [JsonPropertyName("total_ms")] public double TotalMs { get; init; }
}

public class CompletionResponse
{
    [JsonPropertyName("content")] public string Content { get; init; } = "";
    [JsonPropertyName("stop")] public bool Stop { get; init; }
    [JsonPropertyName("tokens_predicted")] public int TokensPredicted { get; init; }
    [JsonPropertyName("tokens_evaluated")] public int TokensEvaluated { get; init; }
    [JsonPropertyName("timings")] public CompletionTimings Timings { get; init; } = new();
}

public class ChatMessage
{
    [JsonPropertyName("role")] public string Role { get; init; } = "user";
    [JsonPropertyName("content")] public string? Content { get; init; } = "";

    // Set on an assistant message being replayed back into the conversation after it
    // requested a tool call, and on the matching "tool" role reply (via ToolCallId).
    [JsonPropertyName("tool_calls")] public List<ToolCall>? ToolCalls { get; init; }
    [JsonPropertyName("tool_call_id")] public string? ToolCallId { get; init; }
}

public class ChatCompletionRequest
{
    [JsonPropertyName("model")] public string Model { get; init; } = "local";
    [JsonPropertyName("messages")] public List<ChatMessage> Messages { get; init; } = [];
    [JsonPropertyName("tools")] public object? Tools { get; init; }
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; init; } = 512;
    [JsonPropertyName("temperature")] public float Temperature { get; init; } = 0.1f;
    [JsonPropertyName("stream")] public bool Stream { get; init; } = false;
}

public class ChatCompletionResponse
{
    [JsonPropertyName("choices")] public List<ChatChoice> Choices { get; init; } = [];
    [JsonPropertyName("usage")] public UsageInfo? Usage { get; init; }

    public string? FirstContent => Choices.FirstOrDefault()?.Message?.Content;
    public bool HasToolCall => Choices.Any(c => c.Message?.ToolCalls?.Count > 0);
}

public class ChatChoice
{
    [JsonPropertyName("message")] public ChatResponseMessage? Message { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

public class ChatResponseMessage
{
    [JsonPropertyName("role")] public string Role { get; init; } = "";
    [JsonPropertyName("content")] public string? Content { get; init; }
    [JsonPropertyName("tool_calls")] public List<ToolCall>? ToolCalls { get; init; }
}

public class ToolCall
{
    [JsonPropertyName("id")] public string Id { get; init; } = "";
    [JsonPropertyName("type")] public string Type { get; init; } = "function";
    [JsonPropertyName("function")] public ToolCallFunction? Function { get; init; }
}

public class ToolCallFunction
{
    [JsonPropertyName("name")] public string Name { get; init; } = "";
    [JsonPropertyName("arguments")] public string Arguments { get; init; } = "";
}

public class UsageInfo
{
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; init; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; init; }
}

public readonly record struct LlamaMetrics(
    double TokensPredictedTotal,
    double PromptTokensTotal,
    double RequestsProcessing);

// -------------------------------------------------------------------------
// Service
// -------------------------------------------------------------------------

public class LlamaServerService(
    IHttpClientFactory httpClientFactory,
    ILogger<LlamaServerService> logger) : IAsyncDisposable
{
    private Process? _serverProcess;
    private int _port = 8080;
    private Action<string>? _logHandler;
    private readonly System.Collections.Concurrent.ConcurrentQueue<string> _logBuffer = new();
    private const int LogBufferMax = 150;
    // Sticky flag: once mmap fails on this machine/session, skip it on all future starts
    private bool _mmapFailed;
    // Sticky flag: once quantized K/V cache fails to start, skip it on all future starts
    // (e.g. some llama.cpp builds reject non-f16 cache types without --flash-attn, or for
    // certain head dims regardless of flash-attn).
    private bool _kvCacheQuantFailed;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public bool IsRunning => _serverProcess is { HasExited: false };
    public string BaseUrl => $"http://127.0.0.1:{_port}";
    public int? ServerProcessId => _serverProcess is { HasExited: false } ? _serverProcess.Id : null;

    /// <summary>
    /// The settings actually used to launch the currently-running (or last-launched) server.
    /// Differs from what was requested when an auto-revert (mmap / KV cache quant) kicked in.
    /// </summary>
    public LlamaSettings? LastEffectiveSettings { get; private set; }

    /// <summary>Human-readable note describing any auto-revert applied on the last start, or null.</summary>
    public string? LastStartAdjustmentNote { get; private set; }

    /// <summary>Starts llama-server with the given settings. Waits until the health endpoint responds.
    /// Automatically retries with --no-mmap if a memory-mapping failure is detected, and likewise
    /// reverts quantized K/V cache types to f16 if the server rejects that combination.</summary>
    public async Task StartAsync(
        string serverExecutable,
        string modelPath,
        LlamaSettings settings,
        int port = 8080,
        CancellationToken ct = default)
    {
        if (IsRunning) await StopAsync();

        _port = port;
        while (_logBuffer.TryDequeue(out string? _)) { }
        LastStartAdjustmentNote = null;

        var resolvedExe = ResolveExecutable(serverExecutable);

        // If mmap or quantized KV cache already failed this session, skip them rather than
        // waste a retry cycle.
        var effectiveSettings = settings;
        if (_mmapFailed && effectiveSettings.Mmap)
            effectiveSettings = WithNoMmap(effectiveSettings);
        if (_kvCacheQuantFailed && HasQuantizedKvCache(effectiveSettings))
            effectiveSettings = WithoutQuantizedKvCache(effectiveSettings);

        LaunchProcess(resolvedExe, effectiveSettings, modelPath, port);

        try
        {
            await WaitForHealthAsync(ct);
        }
        catch (InvalidOperationException ex) when (settings.Mmap && !_mmapFailed && IsMmapFailure(ex.Message))
        {
            // mmap failed — kill the crashed process, then retry without it
            _mmapFailed = true;
            LastStartAdjustmentNote = "mmap disabled (memory-mapping failed)";
            _logHandler?.Invoke(
                "[autosizer] Memory-mapping failed (GGML_ASSERT mmap) — retrying with --no-mmap");

            await KillProcessAsync();
            while (_logBuffer.TryDequeue(out string? _)) { }

            effectiveSettings = WithNoMmap(effectiveSettings);
            LaunchProcess(resolvedExe, effectiveSettings, modelPath, port);
            await WaitForHealthAsync(ct);   // if this also fails, let it throw normally
        }
        catch (InvalidOperationException ex) when (
            !_kvCacheQuantFailed && HasQuantizedKvCache(effectiveSettings) && IsCacheTypeFailure(ex.Message))
        {
            // Quantized K/V cache failed — kill the crashed process, then retry with f16 K/V.
            _kvCacheQuantFailed = true;
            LastStartAdjustmentNote = "quantized K/V cache reverted to f16 (rejected by server)";
            _logHandler?.Invoke(
                "[autosizer] Quantized K/V cache failed to start — retrying with f16 K/V cache");

            await KillProcessAsync();
            while (_logBuffer.TryDequeue(out string? _)) { }

            effectiveSettings = WithoutQuantizedKvCache(effectiveSettings);
            LaunchProcess(resolvedExe, effectiveSettings, modelPath, port);
            await WaitForHealthAsync(ct);   // if this also fails, let it throw normally
        }

        LastEffectiveSettings = effectiveSettings;
    }

    // Spawns the process and wires up stderr capture. Does not wait for health.
    private void LaunchProcess(string exe, LlamaSettings settings, string modelPath, int port)
    {
        var args = settings.ToServerArgs(modelPath, port);
        logger.LogInformation("Starting: {Exe} {Args}", exe, string.Join(" ", args));

        var psi = new ProcessStartInfo(exe, args)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _serverProcess = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start llama-server process.");

        _serverProcess.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null) return;
            _logBuffer.Enqueue(e.Data);
            while (_logBuffer.Count > LogBufferMax)
                _logBuffer.TryDequeue(out string? _);
            _logHandler?.Invoke(e.Data);
        };
        _serverProcess.BeginErrorReadLine();
    }

    private async Task KillProcessAsync()
    {
        if (_serverProcess is null) return;
        try
        {
            if (!_serverProcess.HasExited)
            {
                _serverProcess.Kill(entireProcessTree: true);
                await _serverProcess.WaitForExitAsync();
            }
        }
        catch { /* ignore */ }
        _serverProcess.Dispose();
        _serverProcess = null;
    }

    private static bool IsMmapFailure(string message) =>
        message.Contains("llama-mmap", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("GGML_ASSERT(addr)", StringComparison.OrdinalIgnoreCase);

    private static LlamaSettings WithNoMmap(LlamaSettings s)
    {
        var c = s.Clone();
        c.Mmap = false;
        return c;
    }

    private static bool HasQuantizedKvCache(LlamaSettings s) =>
        (s.CacheTypeK is not null and not "f16") || (s.CacheTypeV is not null and not "f16");

    private static LlamaSettings WithoutQuantizedKvCache(LlamaSettings s)
    {
        var c = s.Clone();
        c.CacheTypeK = null;
        c.CacheTypeV = null;
        return c;
    }

    // Best-effort match against the various ways llama.cpp builds report a rejected
    // K/V cache type — exact wording varies by version (e.g. requiring --flash-attn for
    // non-f16 cache, or a head-dim/block-size assertion).
    private static bool IsCacheTypeFailure(string message) =>
        message.Contains("type_k", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("type_v", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("cache_type", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("V cache", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("K cache", StringComparison.OrdinalIgnoreCase) ||
        (message.Contains("flash_attn", StringComparison.OrdinalIgnoreCase) &&
         (message.Contains("quant", StringComparison.OrdinalIgnoreCase) ||
          message.Contains("GGML_ASSERT", StringComparison.OrdinalIgnoreCase)));

    // Resolve a user-supplied path that may be a directory or lack an extension.
    // Mirrors TurboQuantService.ResolveWindowsExecutable — kept local to avoid coupling.
    private static string ResolveExecutable(string path)
    {
        path = Environment.ExpandEnvironmentVariables(path.Trim());

        // Directory — scan for llama-server inside it
        if (Directory.Exists(path))
        {
            string[] names = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? ["llama-server.exe", "llama-server"]
                : ["llama-server", "llama-server.exe"];

            foreach (var name in names)
            {
                string candidate = Path.Combine(path, name);
                if (File.Exists(candidate)) return candidate;
            }

            throw new FileNotFoundException(
                $"llama-server executable not found in folder: {path}\n" +
                $"Expected llama-server.exe or llama-server inside that directory.");
        }

        // File exists as-is
        if (File.Exists(path)) return path;

        // No extension on Windows — try .exe
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) &&
            string.IsNullOrEmpty(Path.GetExtension(path)))
        {
            if (File.Exists(path + ".exe")) return path + ".exe";
        }

        // Assume it's a PATH command and let the OS resolve it
        return path;
    }

    private async Task WaitForHealthAsync(CancellationToken ct)
    {
        using var http = httpClientFactory.CreateClient();
        http.Timeout = TimeSpan.FromSeconds(5);

        var deadline = DateTime.UtcNow.AddSeconds(120);
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested)
        {
            if (_serverProcess?.HasExited == true)
            {
                // Brief pause so the async stderr reader can flush remaining lines
                await Task.Delay(200, CancellationToken.None);
                var lines = _logBuffer.ToArray();
                string output = lines.Length > 0
                    ? "\n\nllama-server output:\n" + string.Join("\n", lines)
                    : "\n\n(no output captured — check that the executable path is correct)";
                throw new InvalidOperationException($"llama-server exited unexpectedly during startup.{output}");
            }

            try
            {
                var resp = await http.GetAsync($"{BaseUrl}/health", ct);
                if (resp.IsSuccessStatusCode)
                {
                    logger.LogInformation("llama-server is ready on port {Port}", _port);
                    return;
                }
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                // Not ready yet
            }

            await Task.Delay(500, ct);
        }

        ct.ThrowIfCancellationRequested();
        throw new TimeoutException("llama-server did not become healthy within 120 seconds.");
    }

    /// <summary>
    /// Polls llama-server's Prometheus /metrics endpoint (enabled by default via --metrics).
    /// Returns null if the server isn't reachable or metrics aren't available.
    /// </summary>
    public async Task<LlamaMetrics?> GetMetricsAsync(CancellationToken ct = default)
    {
        try
        {
            using var http = httpClientFactory.CreateClient();
            http.Timeout = TimeSpan.FromSeconds(2);
            string text = await http.GetStringAsync($"{BaseUrl}/metrics", ct);

            double tokensPredicted = ParsePrometheusValue(text, "llamacpp:tokens_predicted_total");
            double promptTokens = ParsePrometheusValue(text, "llamacpp:prompt_tokens_total");
            double requestsProcessing = ParsePrometheusValue(text, "llamacpp:requests_processing");

            return new LlamaMetrics(tokensPredicted, promptTokens, requestsProcessing);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            return null;
        }
    }

    private static double ParsePrometheusValue(string metricsText, string metricName)
    {
        foreach (var line in metricsText.AsSpan().EnumerateLines())
        {
            if (line.IsEmpty || line[0] == '#') continue;
            if (!line.StartsWith(metricName)) continue;

            int spaceIdx = line.LastIndexOf(' ');
            if (spaceIdx < 0) continue;

            if (double.TryParse(line[(spaceIdx + 1)..], System.Globalization.CultureInfo.InvariantCulture, out double value))
                return value;
        }
        return 0;
    }

    public void SetLogHandler(Action<string>? handler) => _logHandler = handler;

    public async Task StopAsync()
    {
        if (_serverProcess is null) return;
        try
        {
            if (!_serverProcess.HasExited)
            {
                _serverProcess.Kill(entireProcessTree: true);
                await _serverProcess.WaitForExitAsync();
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning("Error stopping server: {Msg}", ex.Message);
        }
        _serverProcess.Dispose();
        _serverProcess = null;
        _logHandler = null;
        // Wait for the OS to release the port and for the GPU driver to fully reclaim VRAM.
        // Large MoE models need longer than 500ms to unload completely.
        await Task.Delay(2000);
    }

    public async Task<(CompletionResponse response, double ttftMs)> CompleteAsync(
        CompletionRequest request, CancellationToken ct = default)
    {
        using var http = httpClientFactory.CreateClient();
        http.Timeout = TimeSpan.FromSeconds(300);

        var json = JsonSerializer.Serialize(request, _jsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        var sw = Stopwatch.StartNew();
        var httpResp = await http.PostAsync($"{BaseUrl}/completion", content, ct);
        httpResp.EnsureSuccessStatusCode();
        double ttft = sw.Elapsed.TotalMilliseconds;

        var response = await httpResp.Content.ReadFromJsonAsync<CompletionResponse>(_jsonOptions, ct)
            ?? throw new InvalidDataException("Null response from /completion");

        return (response, ttft);
    }

    public async Task<(ChatCompletionResponse response, double ttftMs)> ChatCompleteAsync(
        ChatCompletionRequest request, CancellationToken ct = default)
    {
        using var http = httpClientFactory.CreateClient();
        http.Timeout = TimeSpan.FromSeconds(300);

        var json = JsonSerializer.Serialize(request, _jsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        var sw = Stopwatch.StartNew();
        var httpResp = await http.PostAsync($"{BaseUrl}/v1/chat/completions", content, ct);
        httpResp.EnsureSuccessStatusCode();
        double ttft = sw.Elapsed.TotalMilliseconds;

        var response = await httpResp.Content.ReadFromJsonAsync<ChatCompletionResponse>(_jsonOptions, ct)
            ?? throw new InvalidDataException("Null response from /v1/chat/completions");

        return (response, ttft);
    }

    public async ValueTask DisposeAsync() => await StopAsync();
}
