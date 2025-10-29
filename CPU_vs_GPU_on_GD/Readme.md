# CUDA Optimizers (SGD, RMSProp, Adam): GPU vs CPU

<img width="1500" height="450" alt="Figure_1" src="https://github.com/user-attachments/assets/3fc0bc71-b9e7-4e4e-9557-a444b2ac3db4" />

Сравнение сходимости и скорости классических оптимизаторов (SGD, RMSProp, Adam) на CPU и GPU (CUDA) при решении задачи линейной регрессии методом минимизации средней квадратичной ошибки (MSE).

## Что моделируем

**Данные:**  
Синтетическая линейная регрессия  

![y = X w_true + ε](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20y%20%3D%20X%20w_%7Btrue%7D%20%2B%20%5Cvarepsilon)

где  
![\varepsilon \sim \mathcal{N}(0, 0.1^2)](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20%5Cvarepsilon%20%5Csim%20%5Cmathcal%7BN%7D(0%2C%200.1%5E2))

Размерность по умолчанию:  
![N = 20000](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20N%20%3D%2020000) объектов,  
![D = 128](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20D%20%3D%20128) признаков.

---

**Функция потерь:**  
![MSE(w) = \frac{1}{N} \sum_i (x_i^T w - y_i)^2](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20MSE(w)%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_i%20(x_i%5ET%20w%20-%20y_i)%5E2)

**Градиент:**  
![\nabla_w MSE = \frac{2}{N} X^T (X w - y)](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20%5Cnabla_w%20MSE%20%3D%20%5Cfrac%7B2%7D%7BN%7D%20X%5ET(Xw-y))

---

## Оптимизаторы

- **SGD:** полная (batch) версия — на каждом шаге считается градиент по всему ![X](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20X).  
- **RMSProp:** экспоненциальное сглаживание второго момента градиента (![\beta_2](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20%5Cbeta_2)), деление шага на корень из накопленного момента.  
- **Adam:** адаптивный шаг с моментами первого и второго порядков  
  (![\beta_1, \beta_2](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20%5Cbeta_1%2C%20%5Cbeta_2)) и bias-correction.

**Защита от численной неустойчивости:**  
- глобальный clip градиента (по L2-норме),  
- NaN-trap на GPU.

---

## Логи и метрики

В процессе оптимизации пишется CSV с колонками:
```
time_cpu_ms, loss_cpu, time_gpu_ms, loss_gpu
```
(сэмплируется каждые 10 итераций).

- Время на GPU измеряется `cudaEvent`-ами  
- Время на CPU — высокоточным таймером.

---

**Сравнение ведётся по метрике:**

- ![Loss = MSE](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20Loss%20%3D%20MSE)
- или  
  ![Excess\ Loss = Loss - floor](https://latex.codecogs.com/svg.image?%5Ccolor%7B%23999999%7D%20Excess%5C%20Loss%20%3D%20Loss%20-%20floor)

(разность относительно минимального хвостового уровня).


Визуализация
- Один экран с тремя графиками (SGD, RMSProp, Adam) в один горизонтальный ряд. Заголовок окна содержит названия видеокарты (GPU) и процессора (CPU), определяемые автоматически.
- Оси X/Y подбираются и центрируются индивидуально для каждого графика (масштаб «по своему размеру»). По оси X — время (мс), если есть столбцы time_*; иначе итерации.

## Требования
- CUDA Toolkit 11+ (проверено на 13.x)
- Windows + MSVC (Visual Studio 2022/2019) и/или Ninja
- CMake 3.21+ (для пресетов; подойдёт и 3.18+ без пресетов)
- Python 3 (скрипт установит numpy, pandas, matplotlib)

## Быстрый старт (PowerShell, рекомендуется)
```powershell
cd CPU_vs_GPU_on_GD\scripts
./run_and_plot.ps1
```
Скрипт:
- конфигурирует и собирает проект в `CPU_vs_GPU_on_GD/build` (повторно использует уже созданный build и его генератор);
- создаёт виртуальное окружение `CPU_vs_GPU_on_GD/.venv` и ставит зависимости;
- запускает бинарник (генерирует `logs/*.csv`);
- открывает одно окно с тремя графиками (PNG больше не создаются по умолчанию).

Параметры:
```powershell
./run_and_plot.ps1 -Generator Auto|Ninja|VS2022|VS2019
```

## Вариант для CMD
```bat
cd CPU_vs_GPU_on_GD\scripts
run_and_plot.bat
```
Использует системный Python и также показывает одно окно с тремя графиками.

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
- Флаг `--save` сохранит один PNG `convergence_all.png` вместо показа окна.
- Параметр `--metric loss|excess` переключает метрику по оси Y.

## Где искать данные
- Логи: `build/Release/logs` (CSV: `sgd.csv`, `rmsprop.csv`, `adam.csv`).
- PNG сохраняются только с флагом `--save`.

## Технические детали и оговорки
- CUDA-часть использует простые (не сильно оптимизированные) ядра — цель сравнить тренды, а не показать эталонную производительность.
- Архитектуры CUDA задаются автоматически: `native` (CMake ≥ 3.23) или набор `75 86 89`.
- При конфликте генераторов CMake удалите `build` или используйте `run_and_plot.ps1` — он переиспользует существующий билд.
