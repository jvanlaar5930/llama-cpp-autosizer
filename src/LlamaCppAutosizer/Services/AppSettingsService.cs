using System.Text.Json;

namespace LlamaCppAutosizer.Services;

// File-path-related app configuration: where models are scanned from, where profiles and
// optimization history are written, and the llama-server executable. Kept separate from
// LlamaSettings (which is per-model runtime tuning) and from OptimizationProfile (scoring).
public class AppSettings
{
    public string ModelFolder { get; set; } = "";
    public string ServerExecutable { get; set; } = "llama-server";
    public string ProfilesDirectory { get; set; } = "";
    public string SessionsDirectory { get; set; } = "";

    // Set once the TurboQuant menu has resolved a working TurboQuant-fork llama-server
    // executable. Its presence is the gate for exposing turbo4/turbo3/etc. KV cache types
    // anywhere else in the UI — without it, picking those types would silently do nothing
    // (or fail to start) since stock llama-server doesn't recognize them.
    public string? TurboQuantServerExecutable { get; set; }
}

public class AppSettingsService
{
    // Shared central location so profiles/history persist across working directories —
    // this app is launched from wherever the user happens to be, and relative "profiles"/
    // "sessions" folders scattered per-CWD made history hard to find.
    public static readonly string DefaultRootDirectory = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".llama-autosizer");

    private static readonly string ConfigPath = Path.Combine(DefaultRootDirectory, "settings.json");

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    public AppSettings Current { get; private set; } = new();

    public AppSettingsService()
    {
        Load();
        bool needsSave = false;
        if (string.IsNullOrWhiteSpace(Current.ProfilesDirectory))
        {
            Current.ProfilesDirectory = Path.Combine(DefaultRootDirectory, "profiles");
            needsSave = true;
        }
        if (string.IsNullOrWhiteSpace(Current.SessionsDirectory))
        {
            Current.SessionsDirectory = Path.Combine(DefaultRootDirectory, "sessions");
            needsSave = true;
        }
        if (needsSave) Save();
    }

    private void Load()
    {
        try
        {
            if (!File.Exists(ConfigPath)) return;
            var json = File.ReadAllText(ConfigPath);
            Current = JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
        }
        catch { /* ignore corrupt config, fall back to defaults */ }
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(DefaultRootDirectory);
            File.WriteAllText(ConfigPath, JsonSerializer.Serialize(Current, JsonOpts));
        }
        catch { /* non-critical */ }
    }
}
