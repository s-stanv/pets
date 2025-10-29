import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys
import subprocess
import platform
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter


def find_logs_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        return p
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    candidates = [
        Path.cwd() / 'logs',
        Path.cwd() / 'build' / 'logs',
        repo_root / 'logs',
        repo_root / 'build' / 'logs',
        script_dir / 'logs',
        script_dir.parent / 'logs',
    ]
    for c in candidates:
        if (c / 'sgd.csv').exists() and (c / 'rmsprop.csv').exists() and (c / 'adam.csv').exists():
            return c
    # Fallback to repo_root/logs
    return repo_root / 'logs'


def detect_gpu_cpu() -> Tuple[str, str]:
    # GPU name
    gpu = None
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name',
            '--format=csv,noheader,nounits'
        ], stderr=subprocess.STDOUT, text=True, timeout=3)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            gpu = lines[0]
    except Exception:
        # Try PowerShell for Windows as a fallback
        try:
            out = subprocess.check_output([
                'powershell', '-NoProfile', '-Command',
                'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'
            ], stderr=subprocess.STDOUT, text=True, timeout=3)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # Prefer NVIDIA adapter if present
            nvidia = [l for l in lines if 'NVIDIA' in l.upper()]
            gpu = (nvidia[0] if nvidia else (lines[0] if lines else None))
        except Exception:
            pass
    if not gpu:
        gpu = 'Unknown GPU'

    # CPU name
    cpu = None
    try:
        out = subprocess.check_output(['wmic', 'cpu', 'get', 'Name'], stderr=subprocess.STDOUT, text=True, timeout=3)
        lines = [l.strip() for l in out.splitlines() if l.strip() and 'Name' not in l]
        if lines:
            cpu = lines[0]
    except Exception:
        pass
    if not cpu:
        cpu = platform.processor() or platform.uname().processor or 'Unknown CPU'
    return gpu, cpu


def main() -> int:
    ap = argparse.ArgumentParser(description='Plot CPU vs GPU convergence from CSV logs')
    ap.add_argument('--logs', type=str, default=None, help='Path to logs directory (containing *.csv)')
    ap.add_argument('--metric', type=str, default='loss', choices=['loss','excess'],
                    help="What to plot on Y: raw 'loss' or 'excess' = loss minus noise floor (>=0)")
    ap.add_argument('--save', action='store_true', help='Save plots to PNG instead of showing')
    args = ap.parse_args()

    logs_dir = find_logs_dir(args.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ('SGD', logs_dir / 'sgd.csv'),
        ('RMSProp', logs_dir / 'rmsprop.csv'),
        ('Adam', logs_dir / 'adam.csv'),
    ]

    missing = [name for name, p in series if not p.exists()]
    if missing:
        print('Missing CSV files in', logs_dir)
        for name in missing:
            print(f' - {name}: expected {series[name]}')
        print('Run the optimizer binary first to generate logs.')
        return 1

    # Prepare a single figure with 3 subplots in one horizontal row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15.0, 4.5), constrained_layout=True)
    cpu_style = dict(linestyle='-', linewidth=2.4, color='#d62728', alpha=0.95, zorder=3)
    gpu_style = dict(linestyle='-', linewidth=2.0, color='#1f77b4', alpha=0.9, zorder=2)

    x_is_time = False

    for (idx, (name, p)) in enumerate(series):
        df = pd.read_csv(p)
        ax = axes[idx]
        if {'time_cpu_ms','loss_cpu','time_gpu_ms','loss_gpu'}.issubset(df.columns):
            x_is_time = True
            t_cpu = df['time_cpu_ms'] - float(df['time_cpu_ms'].iloc[0])
            t_gpu = df['time_gpu_ms'] - float(df['time_gpu_ms'].iloc[0])
            y_cpu = df['loss_cpu'].astype(float)
            y_gpu = df['loss_gpu'].astype(float)
            if args.metric == 'excess':
                tail = max(1, int(0.2 * len(df)))
                floor = min(y_cpu.iloc[-tail:].min(), y_gpu.iloc[-tail:].min())
                y_cpu = (y_cpu - floor).clip(lower=0)
                y_gpu = (y_gpu - floor).clip(lower=0)
            ax.plot(t_cpu, y_cpu, label=f'{name} CPU', **cpu_style)
            ax.plot(t_gpu, y_gpu, label=f'{name} GPU', **gpu_style)
            # Per-plot X limits with small padding
            x_min = float(min(t_cpu.min(), t_gpu.min()))
            x_max = float(max(t_cpu.max(), t_gpu.max()))
            x_span = max(1e-9, x_max - x_min)
            pad = 0.02 * x_span
            ax.set_xlim(left=x_min - pad, right=x_max + pad)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=6))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        else:
            need_cols = [c for c in ['iter','loss_cpu','loss_gpu'] if c in df.columns]
            df = df.dropna(subset=need_cols)
            y_cpu = df['loss_cpu'].astype(float)
            y_gpu = df['loss_gpu'].astype(float)
            if args.metric == 'excess':
                tail = max(1, int(0.2 * len(df)))
                floor = min(y_cpu.iloc[-tail:].min(), y_gpu.iloc[-tail:].min())
                y_cpu = (y_cpu - floor).clip(lower=0)
                y_gpu = (y_gpu - floor).clip(lower=0)
            x = df['iter']
            ax.plot(x, y_cpu, label=f'{name} CPU', **cpu_style)
            ax.plot(x, y_gpu, label=f'{name} GPU', **gpu_style)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=6, integer=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            # Per-plot X limits with small padding
            x_min = float(x.min())
            x_max = float(x.max())
            x_span = max(1.0, x_max - x_min)
            pad = 0.02 * x_span
            ax.set_xlim(left=x_min - pad, right=x_max + pad)

        ax.set_ylabel('Excess Loss' if args.metric=='excess' else 'Loss (MSE)')
        # Per-plot Y limits with padding; keep >=0 for 'excess'
        y_min = float(min(y_cpu.min(), y_gpu.min()))
        y_max = float(max(y_cpu.max(), y_gpu.max()))
        if args.metric == 'excess':
            y_min = max(0.0, y_min)
        y_span = max(1e-12, y_max - y_min)
        y_pad = 0.05 * y_span
        ax.set_ylim(bottom=y_min - 0.5*y_pad, top=y_max + 0.5*y_pad)
        ax.set_title(name)
        ax.legend(frameon=True)
        ax.grid(True, which='major', linewidth=0.8, alpha=0.35)
        ax.grid(True, which='minor', linewidth=0.5, alpha=0.18)

    fig.supxlabel('Time (ms)' if x_is_time else 'Iteration')

    gpu_name, cpu_name = detect_gpu_cpu()
    fig.suptitle(f'Convergence: CPU vs GPU — GPU: {gpu_name} | CPU: {cpu_name}', fontsize=12)

    if args.save:
        out_path = logs_dir / 'convergence_all.png'
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    if args.save:
        print(f'Saved plot to {logs_dir}/convergence_all.png')
    else:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
