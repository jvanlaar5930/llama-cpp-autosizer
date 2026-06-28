namespace LlamaCppAutosizer.Models;

public enum QuantizationType
{
    Q4_0, Q4_K_M, Q4_K_S,
    Q5_0, Q5_K_M, Q5_K_S,
    Q6_K,
    Q8_0,
    F16, BF16,
    IQ2_XXS, IQ2_XS, IQ2_S,
    IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M,
    IQ4_XS, IQ4_NL,
}

public class TurboQuantOptions
{
    public string InputModelPath { get; set; } = "";
    public string OutputDirectory { get; set; } = "";
    public QuantizationType QuantType { get; set; } = QuantizationType.Q4_K_M;
    public bool UseImportance { get; set; } = true;     // turbo quant importance-matrix calibration
    public string? CalibrationDataset { get; set; }     // path or null for default
    public int CalibrationSamples { get; set; } = 512;
    public string? TurboQuantExecutable { get; set; }   // path to turboquant or null for PATH lookup

    public string OutputModelPath
    {
        get
        {
            var name = Path.GetFileNameWithoutExtension(InputModelPath);
            var ext = Path.GetExtension(InputModelPath);
            var quantStr = QuantType.ToString().ToLowerInvariant().Replace('_', '-');
            return Path.Combine(OutputDirectory, $"{name}-turboquant-{quantStr}{ext}");
        }
    }
}

public class TurboQuantResult
{
    public bool Success { get; set; }
    public string OutputPath { get; set; } = "";
    public long OriginalSizeMb { get; set; }
    public long QuantizedSizeMb { get; set; }
    public double CompressionRatio => OriginalSizeMb > 0 ? (double)QuantizedSizeMb / OriginalSizeMb : 0;
    public string? ErrorMessage { get; set; }
    public TimeSpan Duration { get; set; }
}
