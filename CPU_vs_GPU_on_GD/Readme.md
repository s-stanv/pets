# CUDA Optimizers (SGD, RMSProp, Adam): GPU vs CPU

Сравнение сходимости классических оптимизаторов (SGD, RMSProp, Adam) на CPU и GPU (CUDA) при минимизации MSE. Бинарник пишет логи в CSV, а скрипты строят графики.

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/0d4c4f8c-edf0-4346-92b2-b5ee93277607" />
<img width="640" height="478" alt="Figure_1" src="https://github.com/user-attachments/assets/9204cc1d-ac44-42c4-be4f-c3173071ac59" />
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/b8a7fa1d-ce12-4df7-bf61-ef3db98267cb" />

## Требования
- CUDA Toolkit 11+ (проверено на 13.x)
- Windows + MSVC (Visual Studio 2022/2019) и/или Ninja
- CMake 3.21+ (для пресетов, иначе подойдёт 3.18+)
- Python 3 (скрипт сам установит зависимости: numpy, pandas, matplotlib)

## Быстрый старт (PowerShell, рекомендуется)
```powershell
cd CPU_vs_GPU_on_GD\scripts
./run_and_plot.ps1
```
Скрипт:
- конфигурирует и собирает проект в `CPU_vs_GPU_on_GD/build` (повторно использует уже созданный build и его генератор);
- создаёт виртуальное окружение `CPU_vs_GPU_on_GD/.venv` и ставит зависимости;
- запускает бинарник (генерирует `logs/*.csv`);
- открывает окна с графиками (по умолчанию графики показываются, не сохраняются).

Параметры:
```powershell
./run_and_plot.ps1 -Generator Auto|Ninja|VS2022|VS2019
```
Если билд уже существует, скрипт не будет переконфигурировать его (что исключает конфликт генераторов/платформ).

## Вариант для CMD
```bat
cd CPU_vs_GPU_on_GD\scripts
run_and_plot.bat
```
Использует системный Python и также показывает графики в окнах.

## Ручная сборка
Visual Studio 2022:
```bat
cd CPU_vs_GPU_on_GD
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
build\Release\cuda_optimizers.exe
```
Ninja:
```bat
cd CPU_vs_GPU_on_GD
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
build\cuda_optimizers.exe
```

## Построение графиков вручную
```bat
cd CPU_vs_GPU_on_GD
python scripts\plots_convergence.py --logs build\Release\logs
```
- Добавьте `--save`, чтобы вместо показа окна сохранить PNG в папку логов.
- `--metric loss|excess` — показать сырую метрику или «лишнюю» потерю (ниже нуля не опускается).

## Где искать данные
- Логи: `build/Release/logs` (CSV: `sgd.csv`, `rmsprop.csv`, `adam.csv`).
- PNG сохраняются только с флагом `--save`.

## Примечания
- Архитектуры CUDA задаются автоматически: `native` (CMake ≥ 3.23) или набор `75 86 89`.
- При «generator mismatch» удалите `build` вручную или используйте `run_and_plot.ps1` — он переиспользует существующий билд.

