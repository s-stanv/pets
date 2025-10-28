$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$buildDir = Join-Path $repoRoot 'build'

$candidates = @(
  (Join-Path $buildDir 'cuda_optimizers.exe'),
  (Join-Path $buildDir 'Release/cuda_optimizers.exe'),
  (Join-Path $buildDir 'Debug/cuda_optimizers.exe'),
  (Join-Path $buildDir 'RelWithDebInfo/cuda_optimizers.exe'),
  (Join-Path $buildDir 'MinSizeRel/cuda_optimizers.exe')
)

$exePath = $null
foreach ($c in $candidates) {
  if (Test-Path $c) { $exePath = (Resolve-Path $c).Path; break }
}

if (-not $exePath) {
  Write-Host "Build not found. Looked for:" -ForegroundColor Yellow
  $candidates | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
  Write-Host "Please build first:" -ForegroundColor Yellow
  Write-Host "  mkdir build; cd build; cmake -G 'Ninja' -T host=x64 -DCMAKE_BUILD_TYPE=Release ..; cmake --build . -j"
  exit 1
}

$workDir = Split-Path -Parent $exePath
Push-Location $workDir
try {
  Write-Host "Running: $exePath" -ForegroundColor Cyan
  & $exePath
}
finally {
  Pop-Location
}

Write-Host "Generating plots..." -ForegroundColor Cyan
python (Join-Path $repoRoot 'cuda\scripts\plots_convergence.py') --logs (Join-Path $workDir 'logs')
Write-Host "Done. See logs/*.csv and logs/*.png" -ForegroundColor Green
