param([string]$Python = 'python', [switch]$SkipWebBuild)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$originalPath = $env:PATH
Push-Location $repoRoot
try {
    if (-not $SkipWebBuild) {
        # Use the Windows command shim, avoiding npx.ps1 execution-policy issues.
        $nodeCommand = Get-Command node.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $npxCommand = Get-Command npx.cmd -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $nodeCommand -or -not $npxCommand) {
            # A terminal opened before the Node installer may have an outdated PATH.
            $nodeDirectories = @(
                "$env:ProgramFiles\nodejs",
                "${env:ProgramFiles(x86)}\nodejs",
                "$env:LOCALAPPDATA\Programs\nodejs"
            )
            foreach ($nodeDirectory in $nodeDirectories) {
                if ((Test-Path -LiteralPath (Join-Path $nodeDirectory 'node.exe')) -and
                    (Test-Path -LiteralPath (Join-Path $nodeDirectory 'npx.cmd'))) {
                    $env:PATH = "$nodeDirectory;$env:PATH"
                    $nodeCommand = Get-Command node.exe -CommandType Application | Select-Object -First 1
                    $npxCommand = Get-Command npx.cmd -CommandType Application | Select-Object -First 1
                    break
                }
            }
        }
        if (-not $nodeCommand -or -not $npxCommand) {
            throw 'Frontend tools missing: install Node.js 24 LTS for Windows from https://nodejs.org/en/download with npm and Add to PATH enabled. Restart VS Code/PowerShell, then check node --version and npx.cmd --version. Alternatively, copy web/dist from the same commit and use -SkipWebBuild. See portal/TEAM_SETUP.md.'
        }
        $nodeVersion = & $nodeCommand.Source --version
        if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v\d+\.\d+\.\d+$') {
            throw 'Could not determine the Node.js version. Check node --version in a new terminal.'
        }
        if ([version]$nodeVersion.Substring(1) -lt [version]'22.12.0') {
            throw "Node.js $nodeVersion is too old. Install Node.js 24 LTS (minimum supported: 22.12), then restart VS Code/PowerShell."
        }
        Write-Host "Frontend tools: Node.js $nodeVersion; $($npxCommand.Source)"
    }
    & $Python -m venv .venv-web
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the web environment. Pass -Python with a working Python 3.11+ executable.' }
    & '.\.venv-web\Scripts\python.exe' -m pip install -r portal/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Web dependency installation failed.' }
    if (-not $SkipWebBuild) {
        & $npxCommand.Source --yes pnpm@11.25.0 --dir web install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed. See the package-manager error above.' }
        & $npxCommand.Source --yes pnpm@11.25.0 --dir web build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    }
    Write-Host 'Portal setup complete. Training dependencies remain in the separate training environment.'
} finally {
    $env:PATH = $originalPath
    Pop-Location
}
