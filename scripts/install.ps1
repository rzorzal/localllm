# scripts/install.ps1 — download + install the right localllm build (Windows).
# Usage:  .\install.ps1          install
#         .\install.ps1 -Print   detect + resolve the asset, print, don't download
param([switch]$Print)
$ErrorActionPreference = 'Stop'

$Repo = 'rzorzal/localllm'
$Api  = "https://api.github.com/repos/$Repo/releases/latest"
$Headers = @{ 'User-Agent' = 'localllm-install'; 'Accept' = 'application/vnd.github+json' }

function Get-Variant {
    $cuda = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        foreach ($c in $caps) {
            $v = 0.0
            $ok = [double]::TryParse(
                $c.Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$v)
            if ($ok -and $v -ge 8.0) { $cuda = $true }
        }
    }
    if ($cuda) { 'windows-x64-cuda' } else { 'windows-x64-cpu' }
}

# Print the detected variant first, so it shows even if the API call below
# fails (e.g. no release published yet -> Invoke-RestMethod throws on 404).
$variant = Get-Variant
Write-Host "Detected variant: $variant"

try {
    $rel = Invoke-RestMethod -Uri $Api -Headers $Headers -UseBasicParsing
} catch {
    Write-Error "No published release yet for $Repo (or the API is unreachable). See https://github.com/$Repo/releases"
    exit 1
}
$asset = $rel.assets | Where-Object { $_.name -like "*-$variant.zip" } | Select-Object -First 1
if (-not $asset) {
    Write-Error "No asset matching *-$variant.zip in the latest release of $Repo."
    exit 1
}
Write-Host "Asset: $($asset.browser_download_url)"
if ($Print) { exit 0 }

$dest = Join-Path $env:LOCALAPPDATA 'localllm'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = Join-Path $env:TEMP "localllm-$variant.zip"
Write-Host "Downloading..."
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $Headers -UseBasicParsing
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip -Force

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$dest", 'User')
    Write-Host "Added $dest to your user PATH (restart the terminal to pick it up)."
}
Write-Host "Installed -> $dest\localllm.exe"
Write-Host "Unsigned: if SmartScreen warns, choose 'More info -> Run anyway'."
