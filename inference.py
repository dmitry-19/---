# inference.py
import os
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from src.logger import setup_logger
from src.reader import AB1Reader
from src.hybrid_aligner import hybrid_align, global_letter_alignment, build_profile, local_dtw, get_conflict_regions, extract_region_peaks, build_gapped_seq
from src.aligner import align_and_map as align_consensus_map
from src.file_utils import find_all_files
from src.model import BaseCallerNet

logger = setup_logger("inference", log_file="inference.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
INV_BASE_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def select_file(files, file_type):
    if not files:
        print(f"Нет доступных файлов типа '{file_type}'.")
        return None
    print(f"\nДоступные {file_type}-файлы:")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {os.path.basename(f)}")
    print("  [0] Пропустить / не использовать")
    while True:
        try:
            choice = input(f"Выберите номер {file_type}-файла: ").strip()
            if choice == '0':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print("Неверный номер, попробуйте снова.")
        except ValueError:
            print("Введите число.")

def plot_trace_with_predictions(reader, peak_indices, predictions, cons_bases=None, title="Trace", save_path=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    colors = {"A": "green", "C": "blue", "G": "black", "T": "red"}
    bases_order = ["A", "C", "G", "T"]

    if not peak_indices:
        return

    start_sample = reader.peaks[peak_indices[0]] - 100
    end_sample = reader.peaks[peak_indices[-1]] + 100
    start_sample = max(0, start_sample)
    end_sample = min(len(reader.traces["A"]), end_sample)

    x = np.arange(start_sample, end_sample)
    for base in bases_order:
        ax.plot(x, reader.traces[base][start_sample:end_sample],
                color=colors[base], alpha=0.8, label=base, linewidth=0.8)

    for i, idx in enumerate(peak_indices):
        if idx < 0 or idx >= len(reader.peaks):
            continue
        peak_pos = reader.peaks[idx]
        if start_sample <= peak_pos < end_sample:
            orig_base = reader.sequence[idx] if idx < len(reader.sequence) else "?"
            pred_base = predictions[i] if i < len(predictions) else "?"
            cons_base = cons_bases[i] if cons_bases and i < len(cons_bases) else None

            if cons_base:
                color = 'green' if pred_base == cons_base else 'red'
            else:
                color = 'blue'

            ax.axvline(x=peak_pos, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
            ax.text(peak_pos, ax.get_ylim()[1]*0.95, orig_base, fontsize=7, ha='center', color='gray', alpha=0.8)
            ax.text(peak_pos, ax.get_ylim()[1]*0.85, pred_base, fontsize=8, ha='center', color=color, fontweight='bold')
            if cons_base:
                ax.text(peak_pos, ax.get_ylim()[1]*0.75, cons_base, fontsize=6, ha='center', color='black', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Sample point")
    ax.set_ylabel("Intensity")
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig

def main():
    logger.info("=== Старт инференса ===")
    # Загружаем модель с мета‑признаками, как она была обучена
    model = BaseCallerNet(use_meta=True).to(DEVICE)
    model_path = "basecaller.pt"
    if not os.path.exists(model_path):
        print("Ошибка: модель basecaller.pt не найдена. Сначала обучите модель.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Модель загружена.\n")

    # Выбор файлов
    data_dir = "data"
    ab1_files, cons_files = find_all_files(data_dir)
    if len(ab1_files) < 2:
        print("Нужно минимум два AB1-файла.")
        return

    print("=== ВЫБОР ПРЯМОГО РИДА ===")
    fwd_path = select_file(ab1_files, "прямой рид (AB1)")
    if not fwd_path: return
    ab1_rem = [f for f in ab1_files if f != fwd_path]
    print("=== ВЫБОР ОБРАТНОГО РИДА ===")
    rev_path = select_file(ab1_rem, "обратный рид (AB1)")
    if not rev_path: return

    cons_seq = None
    cons_path = None
    if cons_files:
        print("=== КОНСЕНСУС (опционально) ===")
        cons_path = select_file(cons_files, "консенсус (FASTA)")
        if cons_path:
            cons_record = next(SeqIO.parse(cons_path, "fasta"))
            cons_seq = str(cons_record.seq).upper()
            logger.info(f"Консенсус загружен: длина {len(cons_seq)}")

    # Чтение и обрезка ридов
    fwd_reader = AB1Reader(fwd_path)
    rev_reader = AB1Reader(rev_path)
    f_start, f_end = fwd_reader.find_quality_trim_bounds()
    r_start, r_end = rev_reader.find_quality_trim_bounds()
    fwd_reader = fwd_reader.trim(f_start, f_end)
    rev_reader = rev_reader.trim(r_start, r_end)
    logger.info(f"Обрезано: fwd {f_start}-{f_end}, rev {r_start}-{r_end}")

    # Получаем карту соответствия пиков
    if cons_seq:
        mapping_cons = align_consensus_map(
            cons_seq,
            fwd_reader.sequence,
            str(Seq(rev_reader.sequence).reverse_complement()),
            fwd_reader.peaks,
            rev_reader.peaks
        )
    else:
        mapping_hybrid, _ = hybrid_align(fwd_reader, rev_reader)
        mapping_cons = []
        for m in mapping_hybrid:
            if m['fwd_peak_idx'] is not None and m['rev_peak_idx'] is not None:
                mapping_cons.append({
                    'fwd_peak_idx': m['fwd_peak_idx'],
                    'rev_peak_idx': m['rev_peak_idx'],
                    'cons_base': None,
                    'fwd_base': m.get('fwd_base'),
                    'rev_base': m.get('rev_base')
                })

    # Списки для отчёта и визуализации
    consensus_calls = []
    orig_fwd_bases = []
    orig_rev_bases = []
    ref_bases = []
    pred_fwd_idxs = []
    pred_rev_idxs = []

    with torch.no_grad():
        for m in mapping_cons:
            if m['fwd_peak_idx'] is None or m['rev_peak_idx'] is None:
                continue
            # Извлекаем окна
            fwd_win, _ = fwd_reader.get_window(m['fwd_peak_idx'], 120)
            rev_win, _ = rev_reader.get_window(m['rev_peak_idx'], 120)
            fwd_arr = np.array([fwd_win[base] for base in ["A","C","G","T"]], dtype=np.float32)
            rev_arr = np.array([rev_win[base] for base in ["A","C","G","T"]], dtype=np.float32)
            max_val = max(fwd_arr.max(), rev_arr.max(), 1e-6)
            fwd_arr /= max_val; rev_arr /= max_val
            X = np.concatenate([fwd_arr, rev_arr], axis=0)

            # Мета‑признаки
            match = 0
            fwd_ratio = fwd_reader.get_peak_ratio(m['fwd_peak_idx'])
            rev_ratio = rev_reader.get_peak_ratio(m['rev_peak_idx'])
            fwd_qual = fwd_reader.get_peak_quality(m['fwd_peak_idx'])
            rev_qual = rev_reader.get_peak_quality(m['rev_peak_idx'])
            meta = np.array([
                match,
                fwd_ratio,
                rev_ratio,
                fwd_qual if fwd_qual is not None else -1,
                rev_qual if rev_qual is not None else -1
            ], dtype=np.float32)

            X_t = torch.tensor(X).unsqueeze(0).to(DEVICE)
            meta_t = torch.tensor(meta).unsqueeze(0).to(DEVICE)
            out = model(X_t, meta_t)
            pred = torch.argmax(out, dim=1).item()
            consensus_calls.append(INV_BASE_MAP[pred])
            orig_fwd_bases.append(m.get('fwd_base', '?'))
            orig_rev_bases.append(m.get('rev_base', '?'))
            ref_bases.append(m.get('cons_base', None))
            pred_fwd_idxs.append(m['fwd_peak_idx'])
            pred_rev_idxs.append(m['rev_peak_idx'])

    final_seq = ''.join(consensus_calls)
    print(f"\nИтоговая консенсусная последовательность (длина {len(final_seq)}):")
    print(final_seq)

    out_fasta = "inference_result.fasta"
    with open(out_fasta, "w") as f:
        f.write(f">inference_consensus\n{final_seq}\n")
    print(f"Сохранено в {out_fasta}")

    # Сравнение с консенсусом, если он есть
    if cons_seq and ref_bases:
        correct = sum(1 for i, p in enumerate(consensus_calls) if p == ref_bases[i])
        total = len(consensus_calls)
        print(f"\nСравнение с загруженным консенсусом:")
        print(f"  Совпадений: {correct}/{total} (точность {correct/total:.3f})")
        fwd_correct = sum(1 for i, b in enumerate(orig_fwd_bases) if b == ref_bases[i])
        rev_correct = sum(1 for i, b in enumerate(orig_rev_bases) if b == ref_bases[i])
        print(f"  Исходный прямой рид совпадает с консенсусом: {fwd_correct}/{total}")
        print(f"  Исходный обратный рид совпадает с консенсусом: {rev_correct}/{total}")

        fixed = 0
        new_errors = 0
        for i in range(total):
            fwd_wrong = (orig_fwd_bases[i] != ref_bases[i])
            rev_wrong = (orig_rev_bases[i] != ref_bases[i])
            model_right = (consensus_calls[i] == ref_bases[i])
            if (fwd_wrong or rev_wrong) and model_right:
                fixed += 1
            elif (not fwd_wrong and not rev_wrong) and not model_right:
                new_errors += 1
        print(f"  Ошибок исправлено моделью: {fixed}")
        print(f"  Новых ошибок (было верно, модель ошиблась): {new_errors}")

    # Визуализация с синхронизированными списками
    if pred_fwd_idxs:
        fwd_preds_vis = consensus_calls
        fwd_cons_vis = ref_bases if ref_bases else None
        plot_trace_with_predictions(
            fwd_reader, pred_fwd_idxs, fwd_preds_vis,
            cons_bases=fwd_cons_vis,
            title="Forward trace with model predictions",
            save_path="output/inference_fwd.png"
        )
    if pred_rev_idxs:
        rev_preds_vis = consensus_calls
        rev_cons_vis = ref_bases if ref_bases else None
        plot_trace_with_predictions(
            rev_reader, pred_rev_idxs, rev_preds_vis,
            cons_bases=rev_cons_vis,
            title="Reverse trace with model predictions",
            save_path="output/inference_rev.png"
        )
    print("Графики сохранены в output/inference_fwd.png и inference_rev.png")
    logger.info("=== Инференс завершён ===")

if __name__ == "__main__":
    main()