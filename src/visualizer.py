# src/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from src.logger import setup_logger

logger = setup_logger(__name__)


def plot_alignment_text(cons, fwd_aln, rev_aln, width=60):
    """Генерирует текстовый блок выравнивания для лога"""
    lines = []
    for i in range(0, len(cons), width):
        lines.append("cons: " + cons[i:i + width])
        lines.append("fwd : " + fwd_aln[i:i + width])
        lines.append("rev : " + rev_aln[i:i + width])
        lines.append("")
    return "\n".join(lines)


def plot_trace_windows(fwd_window, rev_window, fwd_base, rev_base, cons_base,
                       title="Сравнение окон", save_path=None):
    """Рисует пару окон трасс с метками."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    colors = {"A": "green", "C": "blue", "G": "black", "T": "red"}
    bases = ["A", "C", "G", "T"]

    for base in bases:
        ax1.plot(fwd_window[base], color=colors[base], alpha=0.7, label=base)
    ax1.set_title(f"Forward window (called: {fwd_base}, consensus: {cons_base})")
    ax1.legend()

    for base in bases:
        ax2.plot(rev_window[base], color=colors[base], alpha=0.7, label=base)
    ax2.set_title(f"Reverse window (called: {rev_base})")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.debug(f"Сохранён график: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_three_way_alignment(cons_seq, fwd_aln, rev_aln, output_path, max_width=100):
    """
    Сохраняет визуализацию трёх строк выравнивания: cons, fwd, rev.
    Если длина > max_width, разбивает на несколько панелей.
    """
    total_len = len(cons_seq)
    n_panels = (total_len + max_width - 1) // max_width
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2 * n_panels), squeeze=False)

    base_colors = {
        'A': '#4CAF50', 'C': '#2196F3', 'G': '#FFC107', 'T': '#F44336',
        '-': '#E0E0E0', 'N': '#9E9E9E'
    }

    for panel_idx in range(n_panels):
        ax = axes[panel_idx][0]
        start = panel_idx * max_width
        end = min(start + max_width, total_len)
        segment_cons = cons_seq[start:end]
        segment_fwd = fwd_aln[start:end]
        segment_rev = rev_aln[start:end]

        for i, (c, f, r) in enumerate(zip(segment_cons, segment_fwd, segment_rev)):
            # cons
            ax.add_patch(plt.Rectangle((i, 2), 1, 1, color=base_colors.get(c, '#9E9E9E'), ec='white', lw=0.2))
            ax.text(i + 0.5, 2.5, c, ha='center', va='center', fontsize=6, fontweight='bold')
            # fwd
            ax.add_patch(plt.Rectangle((i, 1), 1, 1, color=base_colors.get(f, '#9E9E9E'), ec='white', lw=0.2))
            ax.text(i + 0.5, 1.5, f, ha='center', va='center', fontsize=6)
            # rev
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=base_colors.get(r, '#9E9E9E'), ec='white', lw=0.2))
            ax.text(i + 0.5, 0.5, r, ha='center', va='center', fontsize=6)

        ax.set_xlim(0, len(segment_cons))
        ax.set_ylim(0, 3)
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['rev', 'fwd', 'cons'])
        ax.set_xticks([])
        ax.set_title(f"Three-way alignment (positions {start}-{end})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()