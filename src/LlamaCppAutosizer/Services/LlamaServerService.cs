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
    [JsonPropertyName("content")] public string Content { get; init; } = "";
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

// -------------------------------------------------------------------------
// Service
// -------------------------------------------------------------------------

public class LlamaServerService(
    IHttpClientFactory httpClientFactory,
    ILogger<LlamaServerService> logger) : IAsyncDisposable
{
    private Process? _serverProcess;
    private int _port = 8080;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public bool IsRunning => _serverProcess is { HasExited: false };
    public string BaseUrl => $"http://127.0.0.1:{_port}";

    /// <summary>Starts llama-server with the given settings. Waits until the health endpoint responds.</summary>
    public async Task StartAsync(
        string serverExecutable,
        string modelPath,
        LlamaSettings settings,
        int port = 8080,
        CancellationToken ct = default)
    {
        if (IsRunning) await StopAsync();

        _port = port;
        var resolvedExe = ResolveExecutable(serverExecutable);
        var args = settings.ToServerArgs(modelPath, port);
        logger.LogInformation("Starting: {Exe} {Args}", resolvedExe, string.Join(" ", args));

        var psi = new ProcessStartInfo(resolvedExe, args)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _serverProcess = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start llama-server process.");

        // Suppress logs (they go to stderr)
        _serverProcess.ErrorDataReceived += (_, _) => { };
        _serverProcess.BeginErrorReadLine();

        await WaitForHealthAsync(ct);
    }

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
                throw new InvalidOperationException("llama-server exited unexpectedly during startup.");

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
