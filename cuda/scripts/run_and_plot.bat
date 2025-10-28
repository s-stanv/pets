@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\") do set "REPO_ROOT=%%~fI"
set "BUILD_DIR=%REPO_ROOT%build"

set "EXE_PATH="
if not defined EXE_PATH if exist "%BUILD_DIR%\cuda_optimizers.exe" set "EXE_PATH=%BUILD_DIR%\cuda_optimizers.exe"
if not defined EXE_PATH if exist "%BUILD_DIR%\Release\cuda_optimizers.exe" set "EXE_PATH=%BUILD_DIR%\Release\cuda_optimizers.exe"
if not defined EXE_PATH if exist "%BUILD_DIR%\Debug\cuda_optimizers.exe" set "EXE_PATH=%BUILD_DIR%\Debug\cuda_optimizers.exe"
if not defined EXE_PATH if exist "%BUILD_DIR%\RelWithDebInfo\cuda_optimizers.exe" set "EXE_PATH=%BUILD_DIR%\RelWithDebInfo\cuda_optimizers.exe"
if not defined EXE_PATH if exist "%BUILD_DIR%\MinSizeRel\cuda_optimizers.exe" set "EXE_PATH=%BUILD_DIR%\MinSizeRel\cuda_optimizers.exe"

if not defined EXE_PATH (
  echo Build not found. Looked for:
  echo   %BUILD_DIR%\cuda_optimizers.exe
  echo   %BUILD_DIR%\Release\cuda_optimizers.exe
  echo   %BUILD_DIR%\Debug\cuda_optimizers.exe
  echo   %BUILD_DIR%\RelWithDebInfo\cuda_optimizers.exe
  echo   %BUILD_DIR%\MinSizeRel\cuda_optimizers.exe
  echo Please build first:
  echo   mkdir build ^&^& cd build ^&^& cmake -G "Ninja" -T host=x64 -DCMAKE_BUILD_TYPE=Release .. ^&^& cmake --build . -j
  exit /b 1
)

for %%I in ("%EXE_PATH%") do set "WORK_DIR=%%~dpI"
pushd "%WORK_DIR%"
echo Running: %EXE_PATH%
"%EXE_PATH%"
popd

echo Generating plots...
python "%REPO_ROOT%cuda\scripts\plots_convergence.py" --logs "%WORK_DIR%logs"
echo Done. See logs\*.csv and logs\*.png

endlocal
