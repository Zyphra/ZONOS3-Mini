# start-zonos3.ps1 — launch zonos3-server from local GGUFs; fetch ffmpeg if needed.
# Launched by start-zonos3.bat (double-click) with -ExecutionPolicy Bypass; mirrors
# scripts/start-zonos3.sh — keep the two in sync.
param(
    [string]$Quant,
    [switch]$Cpu,
    [switch]$Gpu,
    [string]$BindHost,
    [int]$Port,
    [switch]$Yes,
    [switch]$NoBrowser,
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'

$GPU_DEFAULT = 'cpu'   # CI stamps this to 'gpu' in the vulkan release archive
$ScriptDir = $PSScriptRoot

if (-not $Quant)    { $Quant    = if ($env:ZONOS3_QUANT)     { $env:ZONOS3_QUANT }     else { 'qat' } }
$ModelDir           = if ($env:ZONOS3_MODEL_DIR)             { $env:ZONOS3_MODEL_DIR } else { Join-Path $ScriptDir 'models' }
if (-not $BindHost) { $BindHost = if ($env:ZONOS3_HOST)      { $env:ZONOS3_HOST }      else { '127.0.0.1' } }
if (-not $Port)     { $Port     = if ($env:ZONOS3_PORT)      { [int]$env:ZONOS3_PORT } else { 1919 } }
$ServerBin          = if ($env:ZONOS3_SERVER_BIN)            { $env:ZONOS3_SERVER_BIN } else { Join-Path $ScriptDir 'zonos3-server.exe' }
if ($env:ZONOS3_ASSUME_YES) { $Yes = $true }
if ($env:ZONOS3_NO_BROWSER) { $NoBrowser = $true }
$FfmpegUrl = $env:ZONOS3_FFMPEG_URL   # override with a single-binary .exe URL (used by tests); default is BtbN LGPL
$FfmpegDir = Join-Path $ModelDir 'bin'
$FfmpegBin = Join-Path $FfmpegDir 'ffmpeg.exe'
$GpuMode = $GPU_DEFAULT
if ($Cpu) { $GpuMode = 'cpu' }
if ($Gpu) { $GpuMode = 'gpu' }

if ($Help) {
    Write-Host @"
usage: start-zonos3.bat [options] [extra zonos3-server args]
  -Quant Q     backbone file models/zonos3-Q.gguf (default qat)
  -Cpu | -Gpu  backend (archive default: $GPU_DEFAULT)
  -BindHost H  bind address (default 127.0.0.1)
  -Port P      port (default 1919)
  -Yes         don't ask before fetching ffmpeg
  -NoBrowser   don't open the web UI
env overrides: ZONOS3_QUANT ZONOS3_MODEL_DIR ZONOS3_HOST ZONOS3_PORT
               ZONOS3_SERVER_BIN ZONOS3_ASSUME_YES ZONOS3_NO_BROWSER ZONOS3_FFMPEG_URL
ffmpeg (voice cloning only) is fetched into <ModelDir>\bin if not already there or on PATH.
"@
    exit 0
}

# not Write-Error: under ErrorActionPreference=Stop it throws before reaching exit
function Fail([string]$msg) { Write-Host "start-zonos3: error: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $ServerBin)) {
    Fail "zonos3-server.exe not found at $ServerBin (run this script from the extracted release archive, or set ZONOS3_SERVER_BIN)"
}


# Fetch a static ffmpeg into $FfmpegDir — voice cloning only; basic TTS never uses it.
# Windows uses the BtbN LGPL build (a zip nesting the exe at ffmpeg-*/bin/ffmpeg.exe);
# override with ZONOS3_FFMPEG_URL (single-binary .exe URL).
function Download-Ffmpeg {
    New-Item -ItemType Directory -Force -Path $FfmpegDir | Out-Null
    if ($FfmpegUrl) {
        Write-Host "downloading ffmpeg (for voice cloning) ..."
        & curl.exe -L --fail --retry 3 --progress-bar -o "$FfmpegBin.part" $FfmpegUrl
        if ($LASTEXITCODE -ne 0) { return $false }
        Move-Item -Force "$FfmpegBin.part" $FfmpegBin
        return $true
    }
    $url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-lgpl-7.1.zip'
    $zip = Join-Path $FfmpegDir 'ffmpeg-dl.zip'
    $ex  = Join-Path $FfmpegDir '.extract'
    Write-Host "downloading ffmpeg (~90 MB, for voice cloning) ..."
    & curl.exe -L --fail --retry 3 --progress-bar -o $zip $url
    if ($LASTEXITCODE -ne 0) { return $false }
    if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
    Expand-Archive -Path $zip -DestinationPath $ex -Force
    $found = Get-ChildItem -Path $ex -Recurse -Filter ffmpeg.exe |
             Where-Object { $_.FullName -match '[\\/]bin[\\/]' } | Select-Object -First 1
    if (-not $found) { Remove-Item -Recurse -Force $zip, $ex -ErrorAction SilentlyContinue; return $false }
    Move-Item -Force $found.FullName $FfmpegBin
    Remove-Item -Recurse -Force $zip, $ex -ErrorAction SilentlyContinue
    return (Test-Path $FfmpegBin)
}

$Backbone = "zonos3-$Quant.gguf"
$Missing = @()
foreach ($f in @($Backbone, 'zac.gguf')) {
    if (-not (Test-Path (Join-Path $ModelDir $f))) { $Missing += $f }
}
if ($Missing.Count -gt 0) {
    Fail "missing $($Missing -join ', ') in $ModelDir — create them with models/convert-*.py (docs/INTERNALS.md)"
}
if (-not (Test-Path (Join-Path $ModelDir 'spk-encoder.gguf'))) {
    Write-Host "note: spk-encoder.gguf missing — speaker-embedding cloning disabled (raw-reference cloning needs zac-encoder.gguf + ssl-teacher.gguf)"
}
foreach ($companion in @('zac-encoder.gguf', 'ssl-teacher.gguf')) {
    if (-not (Test-Path (Join-Path $ModelDir $companion))) {
        Write-Host "note: $companion not found — raw-reference cloning needs both zac-encoder.gguf and ssl-teacher.gguf"
    }
}

# ffmpeg is only needed for voice cloning: prefer our own copy, then a system ffmpeg
# on PATH, else fetch it alongside the models.
$needFfmpeg = $false
if (Test-Path $FfmpegBin) {
    $env:ZONOS3_FFMPEG = $FfmpegBin
} elseif ($env:ZONOS3_SKIP_PATH_FFMPEG -ne '1' -and (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue)) {
    # a system ffmpeg is on PATH; the server resolves it there
} else {
    $needFfmpeg = $true
}

if ($needFfmpeg) {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        Fail 'curl.exe is required to fetch ffmpeg'
    }
    Write-Host 'ffmpeg (~90 MB, voice cloning only) is not installed'
    if (-not $Yes) {
        $ans = Read-Host 'Fetch ffmpeg now? [Y/n]'
        if ($ans -and $ans -notmatch '^[yY]') { Write-Host 'aborted.'; exit 1 }
    }
    if (Download-Ffmpeg) { $env:ZONOS3_FFMPEG = $FfmpegBin }
    else { Write-Warning 'ffmpeg download failed — voice cloning disabled (basic TTS still works)' }
}

# The default voices ship next to this script in release archives; in a source checkout the script lives in
# servers\ggml\scripts\ and they sit at the repo root. Pass them explicitly because the server default is
# cwd-relative and a double-clicked .bat may not run from the archive dir. The server finds the studio itself
# (beside the executable in an archive, studio\dist in a checkout).
function Get-Sibling([string]$name) {
    $p = Join-Path $ScriptDir $name
    if (Test-Path $p) { $p } else { Join-Path $ScriptDir "..\..\..\$name" }
}

$srvArgs = @((Join-Path $ModelDir $Backbone),
             '--zac', (Join-Path $ModelDir 'zac.gguf'),
             '--host', $BindHost, '--port', $Port)
if (Test-Path (Get-Sibling 'default_voices')) {
    $srvArgs += @('--tts-default-voices-dir', (Get-Sibling 'default_voices'))
}
foreach ($companion in @('spk-encoder', 'zac-encoder', 'ssl-teacher')) {
    if (Test-Path (Join-Path $ModelDir "$companion.gguf")) {
        $srvArgs += @("--$companion", (Join-Path $ModelDir "$companion.gguf"))
    }
}
if ($GpuMode -eq 'gpu') { $srvArgs += '--gpu' }
if ($ExtraArgs) { $srvArgs += $ExtraArgs }

# PS 5.1 joins -ArgumentList elements with spaces without quoting; quote them ourselves
$quotedArgs = $srvArgs | ForEach-Object { if ("$_" -match '\s') { '"{0}"' -f $_ } else { "$_" } }
$proc = Start-Process -FilePath $ServerBin -ArgumentList $quotedArgs -NoNewWindow -PassThru
try {
    $pollHost = if ($BindHost -eq '0.0.0.0' -or $BindHost -eq '::') { '127.0.0.1' } else { $BindHost }
    $url = "http://${pollHost}:$Port/"
    $ready = $false
    for ($i = 0; $i -lt 240; $i++) {
        if ($proc.HasExited) { Fail 'server exited during startup' }
        # -s without -S: stderr output under redirection throws NativeCommandError on PS 5.1
        & curl.exe -fs "${url}v2/health" > $null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { Fail "server did not become ready at $url" }

    Write-Host "zonos3: ready -> $url"
    if (-not $NoBrowser) { Start-Process $url }

    Write-Host 'Press Ctrl-C (or close this window) to stop the server.'
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
