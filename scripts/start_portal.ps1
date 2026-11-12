param(
    [ValidateSet('coordinator','institution')][string]$Mode = 'coordinator',
    [string]$Name = 'FedMedAI Research Network',
    [string]$StateDir = '.portal',
    [string]$WebPython = '.venv-web\Scripts\python.exe',
    [string]$TrainerPython = '',
    [string]$ServerAddress = '127.0.0.1',
    [ValidateSet('pc','jetson_nano','jetson_orin')][string]$DeviceType = 'pc',
    [int]$Port = 0,
    [int]$FlowerPort = 8080,
    [string]$DataRoot = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath 'web\dist\index.html')) { throw 'Build the web interface first: scripts\setup_portal.ps1' }
    if (-not (Test-Path -LiteralPath (Join-Path $StateDir 'settings.json'))) {
        if (-not $TrainerPython) { throw 'For first setup, pass -TrainerPython with the Python executable that runs your Flower clients.' }
        $arguments = @('-m','portal','init','--mode',$Mode,'--state-dir',$StateDir,'--name',$Name,
                       '--trainer-python',$TrainerPython,'--advertise-host',$ServerAddress,'--device-type',$DeviceType,'--flower-port',"$FlowerPort")
        if ($Port -gt 0) { $arguments += @('--port',"$Port") }
        if ($DataRoot) { $arguments += @('--data-root',$DataRoot) }
        & $WebPython @arguments
        if ($LASTEXITCODE -ne 0) { throw 'Instance initialization failed.' }
    }
    Write-Host 'Starting the portal. Keep this terminal open; closing the browser does not stop training.'
    & $WebPython -m portal serve --state-dir $StateDir
    if ($LASTEXITCODE -ne 0) { throw 'Portal exited with an error.' }
} finally { Pop-Location }
