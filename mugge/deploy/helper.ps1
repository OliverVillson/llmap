<#
Centipede helper on a Windows PC with an NVIDIA GPU: llama.cpp's llama-server with Qwen3.5, whose
slots the crew on centipede-server uses (deploy/SERVER.md). It installs what it needs the first
time (llama.cpp's CUDA build and the model, under %USERPROFILE%\centipede), sizes the slots to the
GPU's free memory with the same arithmetic as deploy/mac_crew.py, prints this PC's addresses, and
serves until you close the window. The PC stays awake while it serves.

  powershell -ExecutionPolicy Bypass -File helper.ps1
  powershell -ExecutionPolicy Bypass -File helper.ps1 -Model 4b       (Qwen3.5-4B: faster, about 3x the slots)
  -Slots N (instead of what fits)  -Port 8080  -KV q4_0|q8_0  -Ctx 12288 (tokens a slot)

Then on centipede-server: deploy/box.sh helper add http://<an address it printed>:8080
#>
param(
  [string]$Model = $(if ($env:MODEL) { $env:MODEL } else { '9b' }),
  [int]$Port = 8080,
  [int]$Slots = 0,
  [string]$KV = 'q4_0',
  [int]$Ctx = 12288
)
$ErrorActionPreference = 'Stop'
# Invoke-WebRequest is many times slower while it draws a progress bar
$ProgressPreference = 'SilentlyContinue'
$MinBuild = 11669
$Root = Join-Path $env:USERPROFILE 'centipede'
$LlamaDir = Join-Path $Root 'llama.cpp'
New-Item -ItemType Directory -Force -Path $Root, (Join-Path $Root 'models'), $LlamaDir | Out-Null

# The models and the memory arithmetic, as in deploy/mac_crew.py: Qwen3.5 keeps a KV cache in 8 of
# its 32 layers (4 KV heads x 256) and a fixed Gated DeltaNet state in the other 24.
$Models = @{
  '9b' = @{ Repo = 'unsloth/Qwen3.5-9B-GGUF'; File = 'Qwen3.5-9B-Q4_K_M.gguf'; Bytes = 5680522464; Served = 'qwen3.5-9b'; ComputeGB = 1.0 }
  '4b' = @{ Repo = 'unsloth/Qwen3.5-4B-GGUF'; File = 'Qwen3.5-4B-Q6_K.gguf'; Bytes = 3525956768; Served = 'qwen3.5-4b'; ComputeGB = 0.7 }
}
$KvTypeBytes = @{ 'q4_0' = 18 / 32; 'q8_0' = 34 / 32; 'f16' = 2.0 }
if (-not $Models.ContainsKey($Model)) { throw "Model must be 9b or 4b, not $Model" }
if (-not $KvTypeBytes.ContainsKey($KV)) { throw "KV must be q4_0, q8_0 or f16, not $KV" }
$M = $Models[$Model]
$StateBytes = 24 * 4 * (32 * 128 * 128 + 3 * (32 * 128 + 2 * 16 * 128))
$CudaContextGB = 0.3
$MarginGB = 0.25

# 1. the GPU
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi not found: install the NVIDIA driver (nvidia.com/drivers), then run this again' }
$g = ((& nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits | Select-Object -First 1) -split ',') | ForEach-Object { $_.Trim() }
$GpuName = $g[0]; $TotalMiB = [int]$g[1]; $UsedMiB = [int]$g[2]; $Driver = $g[3]
if ([int]($Driver -split '\.')[0] -lt 551) { throw "NVIDIA driver $Driver is too old for llama.cpp's CUDA 12.4 build (551 or newer): update it from nvidia.com/drivers" }
Write-Host "GPU: $GpuName, $TotalMiB MiB ($UsedMiB in use), driver $Driver"

# 2. llama.cpp's release build for CUDA 12.4 (llama-server and the CUDA runtime DLLs side by side)
function Get-Build($exe) {
  # llama-server prints its version on stderr, which Windows PowerShell turns into an error under 'Stop'
  $ErrorActionPreference = 'Continue'
  $v = (& $exe --version 2>&1 | Out-String)
  if ($v -match 'build (\d+)') { return [int]$Matches[1] }
  return 0
}
$Exe = Get-ChildItem -Path $LlamaDir -Filter llama-server.exe -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Exe -or (Get-Build $Exe.FullName) -lt $MinBuild) {
  # llama.cpp's "latest" release is a version (v0.7.0) with no binaries, only nightly-tag.txt naming
  # its b<number> build; the b<number> builds are marked prerelease
  $tag = (Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest').tag_name
  if ($tag -notmatch '^b\d+$') {
    $nt = Join-Path $LlamaDir 'nightly-tag.txt'
    try { Invoke-WebRequest -UseBasicParsing "https://github.com/ggml-org/llama.cpp/releases/download/$tag/nightly-tag.txt" -OutFile $nt; $tag = (Get-Content -Raw $nt).Trim() } catch { $tag = '' }
  }
  if ($tag -notmatch '^b\d+$') {
    # assigned first: Windows PowerShell passes a JSON array down a pipe as one object
    $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30'
    $tag = ($rels | Where-Object { $_.tag_name -match '^b\d+$' } | Select-Object -First 1).tag_name
  }
  if ($tag -notmatch '^b\d+$') { throw "could not find llama.cpp's latest release build on GitHub" }
  if ([int]($tag.TrimStart('b')) -lt $MinBuild) { throw "llama.cpp release $tag is older than build $MinBuild" }
  $base = "https://github.com/ggml-org/llama.cpp/releases/download/$tag"
  $dir = Join-Path $LlamaDir $tag
  Write-Host "llama.cpp $tag for CUDA 12.4 -> $dir"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Invoke-WebRequest -UseBasicParsing "$base/llama-$tag-bin-win-cuda-12.4-x64.zip" -OutFile (Join-Path $LlamaDir 'bin.zip')
  Invoke-WebRequest -UseBasicParsing "$base/cudart-llama-bin-win-cuda-12.4-x64.zip" -OutFile (Join-Path $LlamaDir 'cudart.zip')
  Expand-Archive (Join-Path $LlamaDir 'bin.zip') -DestinationPath $dir -Force
  Expand-Archive (Join-Path $LlamaDir 'cudart.zip') -DestinationPath $dir -Force
  Remove-Item (Join-Path $LlamaDir 'bin.zip'), (Join-Path $LlamaDir 'cudart.zip')
  $Exe = Get-ChildItem -Path $dir -Filter llama-server.exe -Recurse | Select-Object -First 1
  if (-not $Exe) { throw "no llama-server.exe in the $tag archive" }
}
Write-Host "llama-server: $($Exe.FullName) (build $(Get-Build $Exe.FullName))"

# 3. the model (resumes a cut download)
$Gguf = Join-Path (Join-Path $Root 'models') $M.File
if (-not (Test-Path $Gguf) -or (Get-Item $Gguf).Length -ne $M.Bytes) {
  Write-Host ("downloading {0} ({1:N1} GB) from Hugging Face; if it is cut off, run this again and it resumes" -f $M.File, ($M.Bytes / 1e9))
  & curl.exe -fL --retry 5 --retry-delay 3 -C - -o "$Gguf.part" "https://huggingface.co/$($M.Repo)/resolve/main/$($M.File)"
  if ($LASTEXITCODE -ne 0) { throw 'the download stopped: run this again (it resumes)' }
  $got = (Get-Item "$Gguf.part").Length
  if ($got -ne $M.Bytes) { throw "downloaded $got bytes, expected $($M.Bytes): run this again (it resumes)" }
  Move-Item -Force "$Gguf.part" $Gguf
}

# 4. the slots: one per call, each with its own context, as many as the free GPU memory holds
$Free = ($TotalMiB - $UsedMiB) * 1MB
$Slot = $Ctx * (8 * 4 * 256) * (2 * $KvTypeBytes[$KV]) + $StateBytes
$Fixed = $M.Bytes + ($M.ComputeGB + $CudaContextGB) * 1e9 + $MarginGB * 1e9
$Fit = [math]::Floor(($Free - $Fixed) / $Slot)
if ($Slots -le 0) { $Slots = [math]::Min(16, $Fit) }
if ($Slots -lt 1) { throw ("not enough free GPU memory for one slot ({0:N2} GB free, {1:N2} GB needed before slots): close programs using the GPU, or -Model 4b" -f ($Free / 1e9), ($Fixed / 1e9)) }
Write-Host ("slots: {0} of {1} tokens (KV {2}, {3:N0} MiB each); GPU memory planned {4:N1} of {5:N1} GB free" -f $Slots, $Ctx, $KV, ($Slot / 1MB), (($Fixed + $Slots * $Slot) / 1e9), ($Free / 1e9))

# 5. reachable from centipede-server: Windows asks once whether llama-server may use the network (allow it)
try {
  New-NetFirewallRule -DisplayName 'Centipede llama-server' -Direction Inbound -Program $Exe.FullName -Action Allow -Profile Any -ErrorAction Stop | Out-Null
} catch { }
Write-Host ''
Write-Host 'This PC answers on these addresses (use the one on the cable or network centipede-server shares):'
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike '127.*' } | ForEach-Object {
  Write-Host ("  http://{0}:{1}    ({2})" -f $_.IPAddress, $Port, $_.InterfaceAlias)
}
Write-Host 'On centipede-server: cd ~/llmap-demo/mugge && deploy/box.sh helper add http://ADDRESS:8080'
Write-Host 'Keep this window open; closing it stops the helper.'
Write-Host ''

# keep the PC awake (not the display) while this window serves
Add-Type -Namespace Centipede -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Centipede.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null

& $Exe.FullName -m $Gguf --alias $M.Served --host 0.0.0.0 --port $Port -ngl 99 -fa on `
  -np $Slots -c ($Slots * $Ctx) -ctk $KV -ctv $KV `
  --jinja --reasoning-format deepseek --reasoning-budget-message 'Considering the limited time, I have to write the answer now.' `
  --ctx-checkpoints 2 --cache-ram 0 --metrics
