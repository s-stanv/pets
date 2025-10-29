import argparse
from pathlib import Path
from typing import Optional
import sys
import pandas as pd
import matplotlib

matplotlib.use('Agg')
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


def main() -> int:
    ap = argparse.ArgumentParser(description='Plot CPU vs GPU convergence from CSV logs')
    ap.add_argument('--logs', type=str, default=None, help='Path to logs directory (containing *.csv)')
    ap.add_argument('--metric', type=str, default='loss', choices=['loss','excess'],
                    help="What to plot on Y: raw 'loss' or 'excess' = loss minus noise floor (>=0)")
    args = ap.parse_args()

    logs_dir = find_logs_dir(args.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)

    series = {
        'SGD': logs_dir / 'sgd.csv',
        'RMSProp': logs_dir / 'rmsprop.csv',
        'Adam': logs_dir / 'adam.csv',
    }

    missing = [name for name, p in series.items() if not p.exists()]
    if missing:
        print('Missing CSV files in', logs_dir)
        for name in missing:
            print(f' - {name}: expected {series[name]}')
        print('Run the optimizer binary first to generate logs.')
        return 1

    for name, p in series.items():
        df = pd.read_csv(p)
        fig, ax = plt.subplots()
        cpu_style = dict(linestyle='-', linewidth=2.4, color='#d62728', alpha=0.95, zorder=3)
        gpu_style = dict(linestyle='-', linewidth=2.0, color='#1f77b4', alpha=0.9, zorder=2)

        if {'time_cpu_ms','loss_cpu','time_gpu_ms','loss_gpu'}.issubset(df.columns):
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
            xmax = max(float(t_cpu.max()), float(t_gpu.max()))
            ax.set_xlim(left=0, right=xmax * 1.02)
            ax.set_xlabel('Time (ms)')
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
            ax.plot(df['iter'], y_cpu, label=f'{name} CPU', **cpu_style)
            ax.plot(df['iter'], y_gpu, label=f'{name} GPU', **gpu_style)
            ax.set_xlabel('Iteration')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=6, integer=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))

        ax.set_ylabel('Excess Loss (MSE above floor)' if args.metric=='excess' else 'Loss (MSE)')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=ymax)
        ax.set_title(f'Convergence: {name} (CPU vs GPU)')
        ax.legend(frameon=True)
        ax.grid(True, which='major', linewidth=0.8, alpha=0.35)
        ax.grid(True, which='minor', linewidth=0.5, alpha=0.18)
        fig.tight_layout()
        out_path = logs_dir / f'{name.lower()}_convergence.png'
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    print(f'Saved plots to {logs_dir}/*.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
