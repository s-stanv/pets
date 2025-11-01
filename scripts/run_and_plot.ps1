Param(
  [ValidateSet('Auto','Ninja','VS2022','VS2019')]
  [string]$Generator = 'Auto'
)

$ErrorActionPreference = 'Stop'

# Paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectRoot 'build'

function Assert-Command([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Required command '$name' not found in PATH."
  }
}

function Get-CMakeVersion() {
  $m = (& cmake --version | Select-String -Pattern 'cmake version ([0-9]+\.[0-9]+\.[0-9]+)' -AllMatches).Matches
  if ($m.Count -gt 0) { return [Version]($m[0].Groups[1].Value) } else { return [Version]'0.0.0' }
}

function Get-ExistingGenerator([string]$buildDir) {
  $cache = Join-Path $buildDir 'CMakeCache.txt'
  if (-not (Test-Path $cache)) { return $null }
  $line = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.+)$' -AllMatches | Select-Object -First 1
  if ($line) {
    $gen = $line.Matches[0].Groups[1].Value
    switch -Regex ($gen) {
      'Ninja' { return 'Ninja' }
      'Visual Studio 17 2022' { return 'VS2022' }
      'Visual Studio 16 2019' { return 'VS2019' }
      default { return $null }
    }
  }
  return $null
}

function Find-Exe([string]$root) {
  $exe = Get-ChildItem -Path (Join-Path $root 'build') -Filter 'cuda_optimizers.exe' -Recurse -File -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty FullName
  if ($exe) { return (Resolve-Path $exe).Path } else { return $null }
}

Assert-Command 'cmake'
$cmakeVer = Get-CMakeVersion
$hasPresets = ($cmakeVer -ge [Version]'3.21.0') -and (Test-Path (Join-Path $projectRoot 'CMakePresets.json'))

# Choose how to build
$hasNinja   = [bool](Get-Command ninja -ErrorAction SilentlyContinue)
$hasCl      = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
$hasClangCl = [bool](Get-Command clang-cl.exe -ErrorAction SilentlyContinue)
$hasGxx     = [bool](Get-Command g++.exe -ErrorAction SilentlyContinue)

if ($Generator -eq 'Auto') {
  $existing = Get-ExistingGenerator -buildDir $buildDir
  if ($existing) { $Generator = $existing }
  elseif ($hasNinja -and ($hasCl -or $hasClangCl -or $hasGxx)) { $Generator = 'Ninja' }
  else { $Generator = 'VS2022' }
}

# Configure + build
Write-Host "Configuring and building in: $buildDir" -ForegroundColor Cyan
if ($hasPresets) {
  $cachePath = Join-Path $buildDir 'CMakeCache.txt'
  $hasExistingCache = Test-Path $cachePath
  if ($hasExistingCache) {
    # If cache belongs to a different source tree, wipe build dir to avoid cmake re-run with wrong source
    try {
      $homeLine = Select-String -Path $cachePath -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$' -AllMatches | Select-Object -First 1
      if ($homeLine) {
        $cachedHome = $homeLine.Matches[0].Groups[1].Value
        $normCached = (Resolve-Path $cachedHome -ErrorAction SilentlyContinue)
        $normRoot   = (Resolve-Path $projectRoot)
        $sameHome = $false
        if ($normCached) { $sameHome = ($normCached.Path.TrimEnd('\\') -ieq $normRoot.Path.TrimEnd('\\')) }
        if (-not $sameHome) {
          Write-Host "Stale CMake cache detected (points to: $cachedHome). Cleaning '$buildDir'." -ForegroundColor Yellow
          Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
          New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
          $hasExistingCache = $false
        }
      }
    } catch { }
  }
  if ($hasExistingCache) {
    # Reuse existing build tree without reconfiguring to avoid generator/platform mismatch
    switch ($Generator) {
      'Ninja'  { & cmake --build $buildDir --parallel }
      default  { & cmake --build $buildDir --config Release --parallel }
    }
    if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }
  }
  else {
    Push-Location $projectRoot
    try {
      switch ($Generator) {
        'Ninja'  { & cmake --preset ninja-release; if ($LASTEXITCODE -ne 0) { throw 'CMake configure (ninja-release) failed.' }
                   & cmake --build --preset build-ninja }
        'VS2022' { & cmake --preset vs2022;        if ($LASTEXITCODE -ne 0) { & cmake --preset vs2019 }
                   if ($LASTEXITCODE -ne 0) { throw 'CMake configure (VS) failed.' }
                   & cmake --build --preset build-vs2022-release; if ($LASTEXITCODE -ne 0) { & cmake --build --preset build-vs2019-release } }
        'VS2019' { & cmake --preset vs2019;        if ($LASTEXITCODE -ne 0) { throw 'CMake configure (VS2019) failed.' }
                   & cmake --build --preset build-vs2019-release }
      }
      if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }
    }
    finally { Pop-Location }
  }
} else {
  New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
  switch ($Generator) {
    'Ninja'  { & cmake -S $projectRoot -B $buildDir -G Ninja -DCMAKE_BUILD_TYPE=Release }
    'VS2022' { & cmake -S $projectRoot -B $buildDir -G 'Visual Studio 17 2022' -A x64 }
    'VS2019' { & cmake -S $projectRoot -B $buildDir -G 'Visual Studio 16 2019' -A x64 }
  }
  if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
  if ($Generator -eq 'Ninja') { & cmake --build $buildDir --parallel }
  else { & cmake --build $buildDir --config Release --parallel }
  if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }
}

$exePath = Find-Exe -root $projectRoot
if (-not $exePath) { throw 'Build completed, but executable not found in local build folder.' }

# Python env and deps
$venvDir = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts/python.exe'

# Resolve a usable system Python as fallback
$pyCmd = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { $null }
$pyArgs = @()
if ($pyCmd -eq 'py') { $pyArgs = @('-3') }

if (-not (Test-Path $venvPython)) {
  if ($pyCmd -eq $null) { throw 'Python not found. Install Python 3.x or ensure it is on PATH.' }
  Write-Host 'Creating Python virtual environment...' -ForegroundColor Cyan
  & $pyCmd @pyArgs -m venv $venvDir
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
    Write-Warning 'Failed to create venv; will use system Python instead.'
  }
}
Write-Host 'Installing Python requirements...' -ForegroundColor Cyan
if (Test-Path $venvPython) {
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install -r (Join-Path $scriptDir 'requirements.txt')
} elseif ($pyCmd) {
  & $pyCmd @pyArgs -m pip install --user --upgrade pip
  & $pyCmd @pyArgs -m pip install --user -r (Join-Path $scriptDir 'requirements.txt')
} else {
  throw 'No Python available to install requirements.'
}

# Run the executable
$workDir = Split-Path -Parent $exePath
Push-Location $workDir
try {
  Write-Host "Running: $exePath" -ForegroundColor Cyan
  & $exePath
} finally { Pop-Location }

# Generate plots
Write-Host 'Generating plots (windows will open)...' -ForegroundColor Cyan
$plotsScript = Join-Path $scriptDir 'plots_convergence.py'
if (Test-Path $venvPython) {
  & $venvPython $plotsScript --logs (Join-Path $workDir 'logs')
} else {
  & $pyCmd @pyArgs $plotsScript --logs (Join-Path $workDir 'logs')
}
Write-Host 'Done. Close plot windows to finish.' -ForegroundColor Green
