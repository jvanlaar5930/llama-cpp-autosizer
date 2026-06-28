using System.Text.Json;
using System.Text.Json.Serialization;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class ProfileLibraryService(ILogger<ProfileLibraryService> logger)
{
    private const string ProfilesDir = "profiles";

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
    };

    // -------------------------------------------------------------------------
    // CRUD
    // -------------------------------------------------------------------------

    public async Task SaveAsync(SavedProfile profile)
    {
        Directory.CreateDirectory(ProfilesDir);
        await using var fs = new FileStream(profile.ProfileFile, FileMode.Create, FileAccess.Write);
        await JsonSerializer.SerializeAsync(fs, profile, JsonOpts);
        logger.LogDebug("Profile saved: {Name} → {File}", profile.Name, profile.ProfileFile);
    }

    public async Task<SavedProfile?> LoadAsync(string path)
    {
        try
        {
            await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            return await JsonSerializer.DeserializeAsync<SavedProfile>(fs, JsonOpts);
        }
        catch (Exception ex)
        {
            logger.LogError("Failed to load profile {Path}: {Msg}", path, ex.Message);
            return null;
        }
    }

    public async Task<List<SavedProfile>> ListProfilesAsync()
    {
        if (!Directory.Exists(ProfilesDir)) return [];

        var files = Directory.GetFiles(ProfilesDir, "*.json", SearchOption.TopDirectoryOnly);
        var profiles = new List<SavedProfile>();

        foreach (var file in files)
        {
            var p = await LoadAsync(file);
            if (p is not null) profiles.Add(p);
        }

        return [.. profiles.OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase)];
    }

    public void Delete(SavedProfile profile)
    {
        try
        {
            if (File.Exists(profile.ProfileFile))
                File.Delete(profile.ProfileFile);
        }
        catch (Exception ex)
        {
            logger.LogWarning("Failed to delete profile {Name}: {Msg}", profile.Name, ex.Message);
        }
    }

    /// <summary>
    /// Renames a profile and re-saves it. The file name (based on ID) stays the same.
    /// </summary>
    public async Task RenameAsync(SavedProfile profile, string newName)
    {
        profile.Name = newName;
        await SaveAsync(profile);
    }

    /// <summary>
    /// Updates the LastRunAt timestamp and re-saves.
    /// </summary>
    public async Task RecordRunAsync(SavedProfile profile)
    {
        profile.LastRunAt = DateTime.UtcNow;
        await SaveAsync(profile);
    }

    /// <summary>
    /// Checks whether the model file referenced by the profile still exists on disk.
    /// </summary>
    public static bool IsModelAvailable(SavedProfile profile)
        => File.Exists(profile.ModelPath);
}
