# llama-cpp-autosizer

A cross-platform terminal application that automatically benchmarks and tunes `llama-server` settings for any local GGUF model — iterating until the fastest, most stable configuration for your hardware is found.

![.NET 10](https://img.shields.io/badge/.NET-10-512BD4?style=flat-square&logo=dotnet)
![C# 13](https://img.shields.io/badge/C%23-13-239120?style=flat-square&logo=csharp)
![Spectre.Console](https://img.shields.io/badge/Spectre.Console-TUI-00bfff?style=flat-square)
![Cross-platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey?style=flat-square)
![llama.cpp](https://img.shields.io/badge/llama.cpp-server-orange?style=flat-square)
![LLM-guided](https://img.shields.io/badge/recommendations-LLM--guided-blueviolet?style=flat-square)

**Built with:** .NET 10 · Spectre.Console · Microsoft.Extensions.DependencyInjection · Microsoft.Extensions.Http · System.Text.Json · System.Diagnostics.Process

---

## What It Does

Running a local LLM efficiently requires tuning a dozen inter-dependent parameters — GPU layer count, context window, KV cache quantization, flash attention, batch sizes, thread counts — and the best values are different for every model, GPU, and workload. This tool automates that process.

It spawns `llama-server`, runs a battery of timed prompts, collects real metrics (prompt-eval speed, token generation speed, time-to-first-token), then asks the model itself for the next setting to try. Each iteration restarts the server with one targeted change, benchmarks again, and scores the result. It stops when performance converges or a set iteration cap is hit.

---

## Features

- **Auto-discovery** — point to a folder and the tool scans for all `.gguf` files, showing name and size before you pick
- **Two optimization profiles** — Chat (minimize latency, fast token generation) and Agentic (maximize throughput and tool-call reliability for agent loops)
- **LLM-guided recommendations** — uses the running model to suggest the next parameter to test, with a heuristic fallback when the model can't respond
- **Named profiles** — save optimized settings under a name, launch `llama-server` with one keypress, and reuse across sessions
- **Manual overrides** — full interactive settings editor lets you pin any value or inject raw CLI args as hints before or during optimization
- **TurboQuant integration** — set `turbo4`/`turbo3` KV cache types and quantize models via [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- **Live progress display** — iteration table updates in real time showing PP speed, TG speed, TTFT, score, and whether a recommendation came from the model or heuristics
- **Historical comparison** — loads all past sessions and renders a ranked side-by-side table across every model and run; exportable as CSV or JSON
- **Session persistence** — every run is saved to `sessions/` as JSON; reload any past session to view results or apply its best settings
- **Ready-to-use output** — final result includes the exact `llama-server` command to paste into a terminal or script
- **Esc to go back** — every navigation menu supports Escape; Ctrl+C cleanly stops the server and exits

---

## Technical Stack

| Layer | Library / Tool |
|---|---|
| Runtime | .NET 10 / C# 13 |
| Terminal UI | [Spectre.Console](https://spectreconsole.net/) 0.49 — live tables, selection prompts, figlet banner |
| Dependency injection | Microsoft.Extensions.DependencyInjection 9 |
| HTTP client | Microsoft.Extensions.Http 9 — `IHttpClientFactory` for llama-server API calls |
| Logging | Microsoft.Extensions.Logging 9 |
| Serialization | System.Text.Json — session files, config, API DTOs |
| Hardware (Windows) | P/Invoke `GlobalMemoryStatusEx` · `wmic` · `nvidia-smi` · `rocm-smi` |
| Hardware (Linux) | `/proc/meminfo` · `/proc/cpuinfo` · `nvidia-smi` · `rocm-smi` |
| Process control | System.Diagnostics.Process — spawn/kill `llama-server` subprocess |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| .NET 10 SDK | [dotnet.microsoft.com](https://dotnet.microsoft.com/download) |
| llama.cpp `llama-server` | Pre-built releases at [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) |
| At least one `.gguf` model file | Any GGUF-format model works |
| NVIDIA / AMD GPU *(optional)* | Required for GPU layer offload; CPU-only also supported |
| turboquant `llama-server` *(optional)* | Required only for TurboQuant cache types — [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |

---

## Building

```bash
git clone https://github.com/jvanlaar5930/llama-cpp-autosizer
cd llama-cpp-autosizer

# Run directly (development)
dotnet run --project src/LlamaCppAutosizer

# Publish — Windows x64 self-contained single file
dotnet publish src/LlamaCppAutosizer -r win-x64 -c Release -o ./publish

# Publish — Linux x64
dotnet publish src/LlamaCppAutosizer -r linux-x64 -c Release -o ./publish

# Publish — Linux ARM64 (e.g. Raspberry Pi, Ampere)
dotnet publish src/LlamaCppAutosizer -r linux-arm64 -c Release -o ./publish
```

---

## Usage

Launch the app and use the arrow-key menu (Esc goes back, Ctrl+C exits):

```
1. Select Model Folder   — scan a directory for .gguf files and pick one
2. Choose Profile        — Chat or Agentic (see below)
3. Set llama-server Path — full path to llama-server or llama-server.exe
4. Detect Hardware       — reads GPU VRAM, RAM, CPU cores
5. Run Auto-Optimization — start the iterative benchmark loop
6. Edit Settings Manually — override any parameter before or between runs
7. Named Profiles        — run, rename, edit, or delete saved configurations
8. TurboQuant Options    — set turbo KV cache types or quantize a model
9. View / Load Sessions  — browse and reload past optimization runs
H. Historical Comparison — ranked table across all models and sessions
0. Exit
```

Your `llama-server` path, last-used model folder, and selected profile are saved to `autosizer-config.json` so you only configure them once.

---

## Optimization Profiles

### Chat
Optimizes for interactive conversation — low time-to-first-token and fast token generation. Uses shorter benchmark prompts (story, explanation, haiku) and weights TTFT and TG speed most heavily.

### Agentic
Optimizes for agent loops and tool-calling workloads — high prompt-processing throughput, large context windows, and reliable structured output. Benchmark suite includes multi-step reasoning tasks and OpenAI-compatible tool-call tests using the `/v1/chat/completions` endpoint.

---

## Named Profiles

After an optimization run (or after editing settings manually) you can save the configuration under a name. From menu option 7 you can:

- **Run** a saved profile — starts `llama-server` with that configuration and shows the endpoint URL (`http://127.0.0.1:8080/v1`), expected TG/PP speeds, and key settings. The server stays up until you press any key.
- **Edit** settings on a saved profile
- **Rename** or add notes
- **Delete** profiles that are no longer needed

---

## How the Optimizer Works

1. **Hardware scan** — detects available VRAM, RAM, and CPU core count
2. **Initial estimate** — derives starting values for GPU layers, context size, and thread count from hardware and model file size
3. **Baseline benchmark** — runs warmup prompts then timed completions, recording PP speed, TG speed, and TTFT from `llama-server`'s `/completion` response timings
4. **Recommendation** — sends the benchmark history and current settings to the running model and asks it to suggest one parameter change (JSON response); falls back to heuristic rules if the model doesn't cooperate
5. **Apply and re-benchmark** — restarts `llama-server` with the new setting, re-runs the benchmark suite, scores the result
6. **Iterate** — repeats until the score improvement drops below a threshold for N consecutive rounds, or the iteration cap is reached
7. **Output** — prints the best settings and the exact `llama-server` command to use

Parameters explored (in typical order): GPU layers → flash attention → KV cache quantization (`q8_0`, `q4_0`, `turbo4`, `turbo3`) → batch size → context size → thread count → mlock.

---

## Output

At the end of a run the tool shows:

- A metrics summary (best score, PP/TG speeds, TTFT)
- A full settings table for the winning configuration
- The ready-to-use `llama-server` command
- Percentage improvement over the baseline

Results are auto-saved to `sessions/<model>_<profile>_<timestamp>.json`. Use menu option H to compare across all past runs, or export the comparison as a CSV for external analysis.

---

## Session Files

```
sessions/
  Mistral-7B-Instruct_Chat_20260628_143200.json
  Mistral-7B-Instruct_Agentic_20260628_151045.json
  Llama-3.1-8B-Q4_K_M_Chat_20260629_090312.json

profiles/
  a1b2c3d4e5f6....json   ← named profiles saved from optimization runs
```

Each session file contains the full iteration history, hardware snapshot, all benchmark results, and the best settings found.

---

## TurboQuant

The TurboQuant menu (option 8) integrates with [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), a fork of llama.cpp that adds additional KV cache quantization types (`turbo4`, `turbo3`, `turbo2`, `turbo1`, `turbo8`).

**Configure KV cache types** — sets `--cache-type-k turbo4 --cache-type-v turbo3` (or any combination) on the active settings. These are applied whenever you run a profile or optimization with the TurboQuant `llama-server`.

**Quantize a model** — point to the folder containing the TurboQuant-built `llama-server`, select a quantization type and calibration options, and the tool runs the quantization and reports the size reduction.

The TurboQuant `llama-server` can be pointed to via a folder path (the tool searches for `llama-server.exe` / `llama-server` inside it) or a direct path to the executable.
