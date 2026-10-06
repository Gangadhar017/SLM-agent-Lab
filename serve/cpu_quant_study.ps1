# Stage 4 on a CPU, for free: convert a Hugging Face checkpoint to GGUF, quantise it (int8 = Q8_0, int4 = Q4_K_M),
# serve each precision with llama.cpp's OpenAI-compatible llama-server, and run the SAME task harness through the
# server (accuracy + failure taxonomy per precision). Throughput/latency (serve/bench.py) is only meaningful on an
# idle machine, so it is a separate switch.
#
#   powershell -File serve/cpu_quant_study.ps1 -ModelDir <hf snapshot or merged dir> -Tag base -Tokenizer ibm-granite/granite-4.0-350m
#   powershell -File serve/cpu_quant_study.ps1 -ModelDir models/granite-350m-lora-r16 -Tag lora-r16 -Tokenizer ibm-granite/granite-4.0-350m -Bench
#
# Requires: winget install ggml.llamacpp ; pip install gguf ; git clone --depth 1 https://github.com/ggml-org/llama.cpp <LlamaCppSrc>
param(
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Tokenizer = "ibm-granite/granite-4.0-350m",
    [string]$LlamaCppSrc = "C:\Users\om200\tools\llama.cpp",
    [string]$Python = "C:\Users\om200\.venvs\slm-agent-lab\Scripts\python.exe",
    [string]$Precisions = "f16,q8_0,q4_k_m",
    [string]$ConvertType = "f16",   # GGUF type written by the converter; use q8_0 when an f16 file would not fit on disk
    [int]$Limit = 150,
    [int]$Port = 8080,
    [int]$Threads = 4,
    [switch]$Bench,
    [switch]$NoCommit
)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$models = Join-Path $repo "models"
New-Item -ItemType Directory -Force $models | Out-Null
New-Item -ItemType Directory -Force (Join-Path $repo "results\logs") | Out-Null

# locate llama.cpp binaries installed by winget
$pkg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory | Where-Object { $_.Name -like "ggml.llamacpp*" } | Select-Object -First 1
if (-not $pkg) { throw "llama.cpp not found under WinGet packages (winget install ggml.llamacpp)" }
$server = Get-ChildItem $pkg.FullName -Recurse -Filter llama-server.exe | Select-Object -First 1 -ExpandProperty FullName
$quant = Get-ChildItem $pkg.FullName -Recurse -Filter llama-quantize.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $server -or -not $quant) { throw "llama-server.exe / llama-quantize.exe not found in $($pkg.FullName)" }
Write-Output "llama-server: $server"

# 1. convert to GGUF (once); the converted file is the source for the quantisations
$src = Join-Path $models "$Tag-$ConvertType.gguf"
if (-not (Test-Path $src)) {
    Write-Output "converting $ModelDir -> $src"
    & $Python (Join-Path $LlamaCppSrc "convert_hf_to_gguf.py") $ModelDir --outtype $ConvertType --outfile $src 2>&1 | Select-Object -Last 3
}
if (-not (Test-Path $src)) { throw "conversion failed" }

# 2. quantise (once per precision)
$files = @{ $ConvertType = $src }
foreach ($p in $Precisions.Split(",")) {
    if ($p -eq $ConvertType) { continue }
    $out = Join-Path $models "$Tag-$p.gguf"
    if (-not (Test-Path $out)) { & $quant $src $out $p.ToUpper() 2>&1 | Select-Object -Last 2 }
    if (Test-Path $out) { $files[$p] = $out }
}
Get-ChildItem $models -Filter "$Tag-*.gguf" | ForEach-Object { "{0}: {1:N0} MB" -f $_.Name, ($_.Length / 1MB) }

# 3. serve each precision and run the harness (and optionally the throughput bench) through it
foreach ($p in $Precisions.Split(",")) {
    if (-not $files.ContainsKey($p)) { Write-Output "skip $p (no file)"; continue }
    $gguf = $files[$p]
    $label = "$Tag-$p"
    Write-Output "=== serving $label on port $Port"
    $proc = Start-Process -FilePath $server -ArgumentList @("-m", "`"$gguf`"", "--port", "$Port", "-c", "4096", "-t", "$Threads", "--alias", $label, "--log-disable") -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.StatusCode -eq 200) { $ok = $true; break } } catch {}
    }
    if (-not $ok) { Write-Output "server for $label did not become healthy"; Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; continue }
    $runDir = "results\serve\harness_$label"
    cmd /c "`"$Python`" scripts\run_eval.py --backend openai --base-url http://127.0.0.1:$Port --model $label --label $label --tokenizer $Tokenizer --limit $Limit --seed 0 --max-new-tokens 256 --out $runDir > results\logs\serve_harness_$label.log 2>&1"
    Get-Content "results\logs\serve_harness_$label.log" | Where-Object { $_ -match "accuracy=" }
    if ($Bench) {
        cmd /c "`"$Python`" serve\bench.py --base-url http://127.0.0.1:$Port --model $label --tokenizer $Tokenizer --tag $label --concurrency 1,2,4,8 --requests-per-worker 3 --max-tokens 64 > results\logs\serve_bench_$label.log 2>&1"
        Get-Content "results\logs\serve_bench_$label.log" | Where-Object { $_ -match "throughput" }
    }
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    if (-not $NoCommit) {
        git add -- results/serve 2>$null
        if (git diff --cached --name-only) {
            git -c core.safecrlf=false commit -q -m "results: $label served with llama.cpp on CPU, harness through the server (+bench: $Bench)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
            Write-Output "COMMITTED $label"
        }
    }
}
Write-Output "CPU QUANT STUDY DONE ($Tag)"
