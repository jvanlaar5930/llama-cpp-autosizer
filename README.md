# llama-cpp-autosizer

A .NET 10 terminal application that automatically finds optimal `llama-server` parameters for any local GGUF model on your hardware. It runs iterative benchmarks, uses the model itself to recommend settings, and produces a ready-to-paste server command with the best configuration found.

---

## What It Does

Running a local LLM efficiently requires tuning a dozen inter-dependent parameters — GPU layer count, context window, KV cache quantization, flash attention, batch sizes, thread counts — and the best values are different for every model, GPU, and workload. This tool automates that process.

It spawns `llama-server`, runs a battery of timed prompts, collects real metrics (prompt-eval speed, token generation speed, time-to-first-token), then asks the model itself for the next setting to try. Each iteration restarts the server with one targeted change, benchmarks again, and scores the result. It stops when performance converges or a set iteration cap is hit.

---

## Features

- **Auto-discovery** — point to a folder and the tool scans for all `.gguf` files, showing name and size before you pick
- **Two optimization profiles** — Chat (minimize latency, fast token generation) and Agentic (maximize throughput and tool-call reliability for agent loops)
- **LLM-guided recommendations** — uses the running model to suggest the next parameter to test, with a heuristic fallback when the model can't respond
- **Manual overrides** — full interactive settings editor lets you pin any value or inject raw CLI args as hints before or during optimization
- **Live progress display** — iteration table updates in real time showing PP speed, TG speed, TTFT, score, and whether a recommendation came from the model or heuristics
- **Historical comparison** — loads all past sessions and renders a ranked side-by-side table across every model and run; exportable as CSV or JSON
- **TurboQuant integration** — quantize any model via [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) and optionally benchmark the quantized output immediately
- **Session persistence** — every run is saved to `sessions/` as JSON; reload any past session to view results or apply its best settings
- **Ready-to-use output** — final result includes the exact `llama-server` command to paste into a terminal or script

---

## Technical Stack

Built using:

- **.NET 10** / **C# 13**
- **Spectre.Console** — terminal UI, live progress tables, selection prompts, figlet banner
- **Microsoft.Extensions.DependencyInjection** — service registration and constructor injection
- **Microsoft.Extensions.Http** — `IHttpClientFactory` for llama-server API calls
- **Microsoft.Extensions.Logging** — structured logging (warnings and above surface in the UI)
- **System.Management** *(Windows)* — WMI queries for CPU core count and system RAM
- **System.Diagnostics.Process** — spawn/kill `llama-server` and `turboquant` subprocesses
- **System.Text.Json** — session serialization, config persistence, API request/response DTOs

Hardware detection covers NVIDIA GPUs via `nvidia-smi`, AMD via `rocm-smi`, and Intel/generic via `wmic`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| .NET 10 SDK | [dotnet.microsoft.com](https://dotnet.microsoft.com/download) |
| llama.cpp build with `llama-server` | Pre-built releases at [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) |
| At least one `.gguf` model file | Any GGUF-format model works |
| NVIDIA GPU *(optional)* | Required for GPU layer offload; CPU-only mode also supported |
| turboquant *(optional)* | Required only for the TurboQuant menu — [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |

---

## Building

```powershell
git clone https://github.com/jvanlaar5930/llama-cpp-autosizer
cd llama-cpp-autosizer

# Run directly
dotnet run --project src/LlamaCppAutosizer

# Or publish a self-contained Windows executable
dotnet publish src/LlamaCppAutosizer -r win-x64 -c Release -o ./publish
./publish/llama-autosizer.exe
```

---

## Usage

Launch the app and use the numbered menu:

```
1. Select Model Folder   — scan a directory for .gguf files and pick one
2. Choose Profile        — Chat or Agentic (see below)
3. Set llama-server Path — full path to llama-server.exe
4. Detect Hardware       — reads GPU VRAM, RAM, CPU cores
5. Run Auto-Optimization — start the iterative benchmark loop
6. Edit Settings Manually — override any parameter before or between runs
7. TurboQuant Options    — quantize a model and optionally benchmark it
8. View / Load Sessions  — browse and reload past optimization runs
9. Historical Comparison — ranked table across all models and sessions
0. Exit
```

Your `llama-server` path, last-used model folder, and selected profile are saved to `autosizer-config.json` so you only configure them once.

---

## Optimization Profiles

### Chat
Optimizes for interactive conversation — low time-to-first-token and fast token generation. Uses shorter benchmark prompts (story, explanation, haiku) and weights TTFT and TG speed most heavily.

### Agentic
Optimizes for agent loops and tool-calling workloads — high prompt-processing throughput, large context windows, and reliable structured output. Benchmark suite includes multi-step reasoning tasks and OpenAI-compatible tool-call tests using the `/v1/chat/completions` endpoint.

Custom scoring weights can be adjusted from the profile menu if your workload is somewhere in between.

---

## How the Optimizer Works

1. **Hardware scan** — detects available VRAM, RAM, and CPU core count
2. **Initial estimate** — derives starting values for GPU layers, context size, and thread count from hardware and model file size
3. **Baseline benchmark** — runs warmup prompts then timed completions, recording PP speed, TG speed, and TTFT from `llama-server`'s `/completion` response timings
4. **Recommendation** — sends the benchmark history and current settings to the running model and asks it to suggest one parameter change (JSON response); falls back to heuristic rules if the model doesn't cooperate
5. **Apply and re-benchmark** — restarts `llama-server` with the new setting, re-runs the benchmark suite, scores the result
6. **Iterate** — repeats until the score improvement drops below a threshold for N consecutive rounds, or the iteration cap is reached
7. **Output** — prints the best settings and the exact `llama-server` command to use

Parameters explored (in typical order): GPU layers → flash attention → KV cache quantization (`q8_0`, `q4_0`) → batch size → context size → thread count → mlock.

---

## Output

At the end of a run the tool shows:

- A metrics summary (best score, PP/TG speeds, TTFT)
- A full settings table for the winning configuration
- The ready-to-use `llama-server` command
- Percentage improvement over the baseline

Results are auto-saved to `sessions/<model>_<profile>_<timestamp>.json`. Use menu option 9 to compare across all past runs, or export the comparison as a CSV for external analysis.

---

## Session Files

```
sessions/
  Mistral-7B-Instruct_Chat_20260628_143200.json
  Mistral-7B-Instruct_Agentic_20260628_151045.json
  Llama-3.1-8B-Q4_K_M_Chat_20260629_090312.json
```

Each file contains the full iteration history, hardware snapshot, all benchmark results, and the best settings found.

---

## TurboQuant

The TurboQuant menu (option 7) integrates with [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) to quantize a full-precision or higher-precision model to a smaller format using importance-matrix calibration.

After quantization completes the tool reports original vs. quantized size and compression ratio, and optionally switches to the quantized model for an immediate benchmark comparison against the original.

`turboquant` must be installed separately and available in `PATH` (or a custom path can be specified in the menu).
