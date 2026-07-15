using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using LlamaCppAutosizer.Models;
using Microsoft.Extensions.Logging;

namespace LlamaCppAutosizer.Services;

public class HardwareDetectionService(ILogger<HardwareDetectionService> logger)
{
    public async Task<HardwareInfo> DetectAsync()
    {
        var gpus = await DetectGpusAsync();
        var (totalRam, freeRam) = GetRamInfo();
        var (cpuCores, cpuThreads, cpuName) = GetCpuInfo();

        return new HardwareInfo
        {
            Gpus = gpus,
            RamTotalMb = totalRam,
            RamFreeMb = freeRam,
            CpuCores = cpuCores,
            CpuThreads = cpuThreads,
            CpuName = cpuName,
        };
    }

    // -------------------------------------------------------------------------
    // GPU detection
    // -------------------------------------------------------------------------

    private async Task<List<GpuInfo>> DetectGpusAsync()
    {
        var gpus = new List<GpuInfo>();

        // Try NVIDIA first
        var nvGpus = await TryNvidiaSmiAsync();
        if (nvGpus.Count > 0)
        {
            gpus.AddRange(nvGpus);
            return gpus;
        }

        // Try AMD (rocm-smi)
        var amdGpus = await TryRocmSmiAsync();
        if (amdGpus.Count > 0)
        {
            gpus.AddRange(amdGpus);
            return gpus;
        }

        // Windows fallback for GPUs without a vendor CLI (Intel Arc, AMD without rocm-smi,
        // NVIDIA with nvidia-smi missing from PATH). Registry first — wmic's AdapterRAM is
        // a uint32 that silently clamps every card larger than 4 GB to 4095 MB, which then
        // flows into recommendation prompts as "a 4GB GPU".
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var regGpus = await TryWindowsRegistryGpusAsync();
            if (regGpus.Count > 0)
            {
                gpus.AddRange(regGpus);
                return gpus;
            }

            var wmicGpus = await TryWmicGpuAsync();
            gpus.AddRange(wmicGpus);
        }

        return gpus;
    }

    // Reads total VRAM from the display-adapter class registry keys:
    // HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-…}\NNNN\HardwareInformation.qwMemorySize
    // qwMemorySize is a QWORD, so — unlike Win32_VideoController.AdapterRAM — it reports the
    // true size for cards above 4 GB. Vendor-agnostic (NVIDIA / AMD / Intel).
    private async Task<List<GpuInfo>> TryWindowsRegistryGpusAsync()
    {
        if (!OperatingSystem.IsWindows()) return [];
        try
        {
            using var displayClass = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(
                @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}");
            if (displayClass is null) return [];

            // Stale driver entries can list the same adapter twice — keep the largest reading.
            var byName = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
            foreach (var subKeyName in displayClass.GetSubKeyNames())
            {
                if (subKeyName.Length == 0 || !subKeyName.All(char.IsAsciiDigit)) continue; // skip "Properties"
                using var adapter = displayClass.OpenSubKey(subKeyName);
                if (adapter?.GetValue("DriverDesc") is not string name || string.IsNullOrWhiteSpace(name))
                    continue;

                long bytes = adapter.GetValue("HardwareInformation.qwMemorySize") switch
                {
                    long qw => qw,
                    int dw => (uint)dw,
                    byte[] { Length: >= 8 } raw => BitConverter.ToInt64(raw, 0),
                    _ => 0,
                };
                if (bytes <= 0) continue;

                long mb = bytes / (1024 * 1024);
                byName[name] = Math.Max(byName.GetValueOrDefault(name), mb);
            }
            if (byName.Count == 0) return [];

            // Free VRAM isn't in the registry; measure total dedicated usage via perf
            // counters. With a single adapter that gives real headroom; with several we
            // can't attribute usage per adapter, so fall back to reporting total as free.
            long usedMb = byName.Count == 1 ? await TryDedicatedVramUsedMbAsync() : 0;

            int index = 0;
            return byName.Select(kv => new GpuInfo
            {
                Name = kv.Key,
                VramTotalMb = kv.Value,
                VramFreeMb = usedMb > 0 ? Math.Max(0, kv.Value - usedMb) : kv.Value,
                Vendor = GuessVendor(kv.Key),
                Index = index++,
            }).ToList();
        }
        catch (Exception ex)
        {
            logger.LogDebug("Registry GPU query failed: {Msg}", ex.Message);
            return [];
        }
    }

    // Sums "\GPU Adapter Memory(*)\Dedicated Usage" (bytes) across all adapters via
    // typeperf. Returns 0 when unavailable (older Windows, localized counter names) —
    // callers then report total as free, same as the old wmic behavior.
    private async Task<long> TryDedicatedVramUsedMbAsync()
    {
        try
        {
            var output = await RunCommandAsync(
                "typeperf", "\"\\GPU Adapter Memory(*)\\Dedicated Usage\" -sc 1");

            long totalBytes = 0;
            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                // Sample lines are CSV: "timestamp","123456","789",… — headers contain the
                // counter path, so only parse lines whose non-first fields are all numeric.
                if (!line.StartsWith('"') || line.Contains("Dedicated Usage")) continue;
                var fields = line.Split(',').Select(f => f.Trim().Trim('"')).Skip(1);
                foreach (var field in fields)
                {
                    if (double.TryParse(field, System.Globalization.CultureInfo.InvariantCulture, out double v))
                        totalBytes += (long)v;
                }
            }
            return totalBytes / (1024 * 1024);
        }
        catch (Exception ex)
        {
            logger.LogDebug("GPU perf-counter query failed: {Msg}", ex.Message);
            return 0;
        }
    }

    private static string GuessVendor(string name) =>
        name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase) || name.Contains("GeForce", StringComparison.OrdinalIgnoreCase) ? "NVIDIA"
        : name.Contains("AMD", StringComparison.OrdinalIgnoreCase) || name.Contains("Radeon", StringComparison.OrdinalIgnoreCase) ? "AMD"
        : name.Contains("Intel", StringComparison.OrdinalIgnoreCase) || name.Contains("Arc", StringComparison.OrdinalIgnoreCase) ? "Intel"
        : "Unknown";

    private async Task<List<GpuInfo>> TryNvidiaSmiAsync()
    {
        try
        {
            var output = await RunCommandAsync("nvidia-smi",
                "--query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits");

            if (string.IsNullOrWhiteSpace(output)) return [];

            var gpus = new List<GpuInfo>();
            int index = 0;
            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                var parts = line.Split(',');
                if (parts.Length < 3) continue;
                if (long.TryParse(parts[1].Trim(), out var total) &&
                    long.TryParse(parts[2].Trim(), out var free))
                {
                    gpus.Add(new GpuInfo
                    {
                        Name = parts[0].Trim(),
                        VramTotalMb = total,
                        VramFreeMb = free,
                        Vendor = "NVIDIA",
                        Index = index++,
                    });
                }
            }
            return gpus;
        }
        catch (Exception ex)
        {
            logger.LogDebug("nvidia-smi not available: {Msg}", ex.Message);
            return [];
        }
    }

    private async Task<List<GpuInfo>> TryRocmSmiAsync()
    {
        try
        {
            // rocm-smi --showmeminfo vram outputs lines like:
            // GPU[0] : VRAM Total Memory (B): 17163091968
            var output = await RunCommandAsync("rocm-smi", "--showmeminfo vram --json");
            if (string.IsNullOrWhiteSpace(output)) return [];

            // Simplified: just detect presence and grab values
            var totalMatches = Regex.Matches(output, @"""VRAM Total Memory \(B\)""\s*:\s*""?(\d+)");
            var usedMatches = Regex.Matches(output, @"""VRAM Total Used Memory \(B\)""\s*:\s*""?(\d+)");

            var gpus = new List<GpuInfo>();
            for (int i = 0; i < totalMatches.Count; i++)
            {
                long total = long.Parse(totalMatches[i].Groups[1].Value) / (1024 * 1024);
                long used = i < usedMatches.Count
                    ? long.Parse(usedMatches[i].Groups[1].Value) / (1024 * 1024)
                    : 0;
                gpus.Add(new GpuInfo
                {
                    Name = $"AMD GPU {i}",
                    VramTotalMb = total,
                    VramFreeMb = total - used,
                    Vendor = "AMD",
                    Index = i,
                });
            }
            return gpus;
        }
        catch (Exception ex)
        {
            logger.LogDebug("rocm-smi not available: {Msg}", ex.Message);
            return [];
        }
    }

    private async Task<List<GpuInfo>> TryWmicGpuAsync()
    {
        try
        {
            var output = await RunCommandAsync("wmic",
                "path Win32_VideoController get Name,AdapterRAM /format:csv");
            if (string.IsNullOrWhiteSpace(output)) return [];

            var gpus = new List<GpuInfo>();
            int index = 0;
            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Skip(1))
            {
                var parts = line.Split(',');
                if (parts.Length < 3) continue;
                string name = parts[2].Trim();
                if (string.IsNullOrEmpty(name)) continue;
                long.TryParse(parts[1].Trim(), out var adapterRamBytes);
                // AdapterRAM is a uint32 — any card over 4 GB reads as exactly ~4095 MB.
                // This path only runs when the registry query above already failed, so
                // there's nothing better to substitute; log it so a "4GB GPU" appearing
                // in recommendation prompts is traceable to this clamp.
                if (adapterRamBytes >= 0xFFF0_0000)
                    logger.LogDebug("wmic AdapterRAM clamped at 4GB for {Name} — real VRAM is likely larger", name);
                gpus.Add(new GpuInfo
                {
                    Name = name,
                    VramTotalMb = adapterRamBytes / (1024 * 1024),
                    VramFreeMb = adapterRamBytes / (1024 * 1024),  // wmic doesn't give free VRAM
                    Vendor = name.Contains("Intel", StringComparison.OrdinalIgnoreCase) ? "Intel"
                           : name.Contains("AMD", StringComparison.OrdinalIgnoreCase) ? "AMD"
                           : "Unknown",
                    Index = index++,
                });
            }
            return gpus;
        }
        catch (Exception ex)
        {
            logger.LogDebug("wmic GPU query failed: {Msg}", ex.Message);
            return [];
        }
    }

    // -------------------------------------------------------------------------
    // RAM detection
    // -------------------------------------------------------------------------

    private static (long totalMb, long freeMb) GetRamInfo()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return GetWindowsRam();

        // Linux / macOS fallback via /proc/meminfo or sysctl
        try
        {
            if (File.Exists("/proc/meminfo"))
            {
                var lines = File.ReadAllLines("/proc/meminfo");
                long total = ParseMemInfoKb(lines, "MemTotal:");
                long free = ParseMemInfoKb(lines, "MemAvailable:");
                return (total / 1024, free / 1024);
            }
        }
        catch { /* ignore */ }

        var gcInfo = GC.GetGCMemoryInfo();
        long approx = gcInfo.TotalAvailableMemoryBytes / (1024 * 1024);
        return (approx, approx / 2);
    }

    private static long ParseMemInfoKb(string[] lines, string key)
    {
        var line = lines.FirstOrDefault(l => l.StartsWith(key));
        if (line is null) return 0;
        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return parts.Length >= 2 && long.TryParse(parts[1], out var v) ? v : 0;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx lpBuffer);

    private static (long totalMb, long freeMb) GetWindowsRam()
    {
        var status = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        if (GlobalMemoryStatusEx(ref status))
            return ((long)(status.ullTotalPhys / 1024 / 1024), (long)(status.ullAvailPhys / 1024 / 1024));
        return (0, 0);
    }

    // -------------------------------------------------------------------------
    // CPU detection
    // -------------------------------------------------------------------------

    private static (int cores, int threads, string name) GetCpuInfo()
    {
        int threads = Environment.ProcessorCount;
        int cores = threads; // conservative default

        string name = "Unknown CPU";

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var psi = new ProcessStartInfo("wmic", "cpu get Name,NumberOfCores,NumberOfLogicalProcessors /format:csv")
                {
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var proc = Process.Start(psi)!;
                var output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit();

                foreach (var line in output.Split('\n').Skip(1))
                {
                    var parts = line.Trim().Split(',');
                    if (parts.Length >= 4)
                    {
                        name = parts[1].Trim();
                        int.TryParse(parts[2].Trim(), out cores);
                        int.TryParse(parts[3].Trim(), out threads);
                        break;
                    }
                }
            }
            catch { /* fall through */ }
        }
        else if (File.Exists("/proc/cpuinfo"))
        {
            try
            {
                var lines = File.ReadAllLines("/proc/cpuinfo");
                var modelLine = lines.FirstOrDefault(l => l.StartsWith("model name"));
                if (modelLine is not null)
                    name = modelLine.Split(':').LastOrDefault()?.Trim() ?? name;
                cores = lines.Count(l => l.StartsWith("cpu cores")) > 0
                    ? lines.Where(l => l.StartsWith("cpu cores"))
                           .Select(l => int.TryParse(l.Split(':').Last().Trim(), out int c) ? c : 1)
                           .Sum()
                    : threads / 2;
            }
            catch { /* fall through */ }
        }

        return (Math.Max(1, cores), Math.Max(1, threads), name);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static async Task<string> RunCommandAsync(string exe, string args)
    {
        var psi = new ProcessStartInfo(exe, args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using var proc = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start {exe}");
        var output = await proc.StandardOutput.ReadToEndAsync();
        await proc.WaitForExitAsync();
        return output;
    }
}
