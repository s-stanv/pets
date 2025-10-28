# CUDA Optimizers (SGD, RMSProp, Adam) — GPU vs CPU

Синтетическая линейная регрессия (MSE). Логи в CSV, скрипт визуализации.

## Требования
- CUDA Toolkit 11+ (проверено с 13.x)
- CMake 3.18+
- Python 3 + pandas + matplotlib

## Сборка (Windows, рекомендуемый вариант)
Откройте x64 Native Tools Command Prompt for VS 2022 и выполните:
```bat
mkdir build && cd build
cmake -G "Ninja" -T host=x64 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Запуск и графики

1) Запустите бинарник из консоли, а не двойным кликом, чтобы видеть вывод и дождаться завершения:
```bat
cd build
cuda_optimizers.exe
```

По умолчанию программа пишет логи в `logs/*.csv` и выводит тайминги. Чтобы построить графики, нужен Python с pandas и matplotlib:
```bat
pip install pandas matplotlib
python ..\cuda\scripts\plots_convergence.py
```
Изображения сохраняются в `logs/*.png`.

Альтернатива: один шаг из корня репозитория (Windows PowerShell):
```powershell
./cuda/scripts/run_and_plot.ps1
```