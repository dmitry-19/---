# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from src.logger import setup_logger
from src.reader import AB1Reader
from src.hybrid_aligner import hybrid_align, global_letter_alignment, build_profile, local_dtw, get_conflict_regions, \
    extract_region_peaks, build_gapped_seq
from src.aligner import align_and_map as align_consensus_map
from src.file_utils import find_all_files, get_unprocessed_triples
from src.dataset_builder import build_dataset
from src.visualizer import plot_three_way_alignment


# ========== Функции визуализации (без изменений) ==========
def plot_chromatogram(reader, region_start, region_end, title="Chromatogram", save_path=None):
    """Рисует участок хроматограммы с отмеченными пиками и вызванными основаниями."""
    fig, ax = plt.subplots(figsize=(15, 5))
    colors = {"A": "green", "C": "blue", "G": "black", "T": "red"}
    bases_order = ["A", "C", "G", "T"]

    if region_start < 0 or region_end >= len(reader.peaks):
        print(f"Предупреждение: индексы пиков выходят за пределы (всего пиков: {len(reader.peaks)})")
        return

    start_sample = reader.peaks[region_start] - 100
    end_sample = reader.peaks[region_end] + 100
    start_sample = max(0, start_sample)
    end_sample = min(len(reader.traces["A"]), end_sample)

    x = np.arange(start_sample, end_sample)
    for base in bases_order:
        ax.plot(x, reader.traces[base][start_sample:end_sample],
                color=colors[base], alpha=0.8, label=base, linewidth=0.8)

    for i in range(region_start, region_end + 1):
        peak_pos = reader.peaks[i]
        if start_sample <= peak_pos < end_sample:
            ax.axvline(x=peak_pos, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
            base = reader.sequence[i] if i < len(reader.sequence) else "?"
            ax.text(peak_pos, ax.get_ylim()[1] * 0.9, base, fontsize=8, ha='center', color=colors.get(base, 'black'))

    ax.set_title(title)
    ax.set_xlabel("Sample point")
    ax.set_ylabel("Intensity")
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_dtw_path(profiles_fwd, profiles_rev, path, region_start=None, region_end=None, save_path=None):
    """Визуализирует DTW-путь между двумя последовательностями профилей."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fwd_sum = profiles_fwd.sum(axis=1) if len(profiles_fwd) > 0 else np.array([])
    rev_sum = profiles_rev.sum(axis=1) if len(profiles_rev) > 0 else np.array([])

    ax1.plot(fwd_sum, 'b-', label='Forward peaks', alpha=0.7)
    ax1.plot(rev_sum, 'r-', label='Reverse peaks', alpha=0.7)
    ax1.set_title("Summed intensity profiles")
    ax1.legend()
    ax1.set_xlabel("Peak index")
    ax1.set_ylabel("Summed intensity")

    if path and len(path) > 0:
        i_idx, j_idx = zip(*path)
        ax2.plot(j_idx, i_idx, 'k-', linewidth=0.5)
        ax2.scatter(j_idx, i_idx, c='blue', s=2, alpha=0.5)
        ax2.set_xlabel("Reverse peak index")
        ax2.set_ylabel("Forward peak index")
        ax2.set_title("DTW alignment path")
        max_i = max(i_idx)
        max_j = max(j_idx)
        ax2.plot([0, max_j], [0, max_i], 'r--', alpha=0.3, label='Identity')
        ax2.legend()

    if region_start is not None and region_end is not None:
        fig.suptitle(f"DTW alignment for region [{region_start}, {region_end}]")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_alignment_overview(fwd_aln, rev_aln, conflict_regions, save_path=None):
    """Текстовая схема выравнивания с подсветкой конфликтных регионов."""
    fig, ax = plt.subplots(figsize=(16, 3))

    n_cols = len(fwd_aln)
    matrix = np.zeros((2, n_cols))

    base_to_val = {"A": 0.2, "C": 0.4, "G": 0.6, "T": 0.8, "-": 0.0, "N": 0.5}
    for i, (f, r) in enumerate(zip(fwd_aln, rev_aln)):
        matrix[0, i] = base_to_val.get(f, 0.5)
        matrix[1, i] = base_to_val.get(r, 0.5)

    ax.imshow(matrix, aspect='auto', cmap='Set3', interpolation='nearest')

    for start, end in conflict_regions:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.2, color='red')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Forward', 'Reverse'])
    ax.set_title("Alignment overview (red = conflict regions)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def select_file(files, file_type):
    """Позволяет пользователю выбрать один файл из списка."""
    if not files:
        print(f"Нет доступных файлов типа '{file_type}'.")
        return None
    print(f"\nДоступные {file_type}-файлы:")
    for i, f in enumerate(files):
        print(f"  [{i + 1}] {os.path.basename(f)}")
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


# ========== Основная функция обработки одной тройки ==========
def process_triple(fwd_path, rev_path, cons_path, output_dir, global_X, global_y, global_meta):
    """
    Обрабатывает одну тройку файлов, визуализирует результаты и добавляет
    примеры в глобальные списки X, y, meta (in-place).
    Возвращает обновлённые списки.
    """
    logger = setup_logger("main", log_file="pipeline.log")
    prefix = os.path.splitext(os.path.basename(cons_path))[0]  # базовое имя консенсуса
    sample_out = os.path.join(output_dir, prefix)
    os.makedirs(sample_out, exist_ok=True)

    # 1. Чтение данных
    fwd_reader = AB1Reader(fwd_path)
    rev_reader = AB1Reader(rev_path)
    cons_record = next(SeqIO.parse(cons_path, "fasta"))
    cons_seq = str(cons_record.seq).upper()
    logger.info(f"[{prefix}] Консенсус загружен: длина {len(cons_seq)}")

    # 2. Обрезка
    fwd_start, fwd_end = fwd_reader.find_quality_trim_bounds()
    rev_start, rev_end = rev_reader.find_quality_trim_bounds()
    fwd_reader = fwd_reader.trim(fwd_start, fwd_end)
    rev_reader = rev_reader.trim(rev_start, rev_end)
    logger.info(f"[{prefix}] Обрезка: fwd={fwd_start}-{fwd_end}, rev={rev_start}-{rev_end}")

    # 3. Глобальное выравнивание
    seq_fwd = fwd_reader.sequence
    seq_rev = str(Seq(rev_reader.sequence).reverse_complement())
    fwd_aln, rev_aln = global_letter_alignment(seq_fwd, seq_rev)

    # 4. Гибридное выравнивание
    mapping, indels = hybrid_align(fwd_reader, rev_reader)

    # 5. Визуализации
    conflict_regions = get_conflict_regions(fwd_aln, rev_aln)
    plot_alignment_overview(fwd_aln, rev_aln, conflict_regions,
                            save_path=os.path.join(sample_out, "alignment_overview.png"))
    for idx, (start, end) in enumerate(conflict_regions[:3]):
        f_peaks_region, r_peaks_region = extract_region_peaks(
            fwd_reader, rev_reader, fwd_aln, rev_aln, (start, end)
        )
        f_valid = [p for p in f_peaks_region if p is not None]
        r_valid = [p for p in r_peaks_region if p is not None]
        if f_valid and r_valid:
            plot_chromatogram(fwd_reader, f_valid[0], f_valid[-1],
                              title=f"Forward - Region {start}-{end}",
                              save_path=os.path.join(sample_out, f"chromatogram_fwd_region{idx + 1}.png"))
            plot_chromatogram(rev_reader, r_valid[0], r_valid[-1],
                              title=f"Reverse - Region {start}-{end}",
                              save_path=os.path.join(sample_out, f"chromatogram_rev_region{idx + 1}.png"))
            prof_f = build_profile(fwd_reader, f_valid)
            prof_r = build_profile(rev_reader, r_valid)
            path, cost = local_dtw(prof_f, prof_r)
            plot_dtw_path(prof_f, prof_r, path, start, end,
                          save_path=os.path.join(sample_out, f"dtw_path_region{idx + 1}.png"))

    if indels:
        first_indel = indels[0]
        # упрощённая логика для визуализации контекста первого индела
        if first_indel['type'] == 'del_in_fwd':
            center_rev = first_indel['rev_pos']
            region_fwd_start = max(0, (center_rev - 10)) if center_rev is not None else 0
            region_fwd_end = min(len(fwd_reader.peaks) - 1, (center_rev + 10)) if center_rev is not None else 20
            region_rev_start = max(0, center_rev - 10) if center_rev is not None else 0
            region_rev_end = min(len(rev_reader.peaks) - 1, center_rev + 10) if center_rev is not None else 20
        elif first_indel['type'] == 'del_in_rev':
            center_fwd = first_indel['fwd_pos']
            region_fwd_start = max(0, center_fwd - 10) if center_fwd is not None else 0
            region_fwd_end = min(len(fwd_reader.peaks) - 1, center_fwd + 10) if center_fwd is not None else 20
            region_rev_start = max(0, (center_fwd - 10)) if center_fwd is not None else 0
            region_rev_end = min(len(rev_reader.peaks) - 1, (center_fwd + 10)) if center_fwd is not None else 20
        else:
            region_fwd_start, region_fwd_end = 0, 20
            region_rev_start, region_rev_end = 0, 20
        plot_chromatogram(fwd_reader, region_fwd_start, region_fwd_end,
                          title=f"Forward near indel ({first_indel['type']})",
                          save_path=os.path.join(sample_out, "indel_fwd_context.png"))
        plot_chromatogram(rev_reader, region_rev_start, region_rev_end,
                          title=f"Reverse near indel ({first_indel['type']})",
                          save_path=os.path.join(sample_out, "indel_rev_context.png"))

    # 6. Привязка к консенсусу и построение датасета
    mapping_cons = align_consensus_map(
        cons_seq,
        fwd_reader.sequence,
        str(Seq(rev_reader.sequence).reverse_complement()),
        fwd_reader.peaks,
        rev_reader.peaks
    )

    # Трёхсторонняя визуализация
    aln_fwd = PairwiseAligner().align(cons_seq, fwd_reader.sequence)[0]
    aln_rev = PairwiseAligner().align(cons_seq, str(Seq(rev_reader.sequence).reverse_complement()))[0]
    cons_fwd, fwd_aln_str = build_gapped_seq(cons_seq, fwd_reader.sequence, aln_fwd.aligned)
    cons_rev, rev_aln_str = build_gapped_seq(cons_seq, str(Seq(rev_reader.sequence).reverse_complement()),
                                             aln_rev.aligned)
    min_len = min(len(cons_fwd), len(cons_rev))
    cons_vis = cons_fwd[:min_len]
    fwd_vis = fwd_aln_str[:min_len]
    rev_vis = rev_aln_str[:min_len]
    plot_three_way_alignment(cons_vis, fwd_vis, rev_vis,
                             os.path.join(sample_out, "three_way_alignment.png"))

    # Датасет (без сохранения в файл)
    X, y, meta = build_dataset(mapping_cons, fwd_reader, rev_reader, output_dir=None)
    if X is not None and len(y) > 0:
        global_X.append(X)
        global_y.append(y)
        global_meta.append(meta)
        logger.info(f"[{prefix}] Добавлено {len(y)} примеров (total mismatched: {int(np.sum(meta[:, 0] == 0))})")
    else:
        logger.info(f"[{prefix}] Нет примеров для добавления.")

    return global_X, global_y, global_meta


# ========== Главная функция с циклом ==========
def main():
    logger = setup_logger("main", log_file="pipeline.log")
    logger.info("=== Старт множественного пайплайна ===")

    data_dir = "data"
    ab1_files, cons_files = find_all_files(data_dir)

    if len(ab1_files) < 2:
        logger.error("Недостаточно AB1-файлов.")
        return
    if not cons_files:
        logger.warning("Нет FASTA-файлов консенсуса, работа без консенсуса невозможна.")
        return

    # Глобальные накопители
    all_X = []
    all_y = []
    all_meta = []
    processed_prefixes = set()

    while True:
        triples = get_unprocessed_triples(data_dir, processed_prefixes)
        if not triples:
            print("Все доступные тройки обработаны.")
            break

        print("\n=== ВЫБОР ТРОЙКИ (ПРЯМОЙ + ОБРАТНЫЙ) ===")
        print("Доступные тройки:")
        for i, t in enumerate(triples):
            print(f"  [{i + 1}] {t['prefix']}")
        print("  [0] Завершить обработку")

        choice = input("Выберите номер тройки: ").strip()
        if choice == '0':
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(triples):
                selected = triples[idx]
            else:
                print("Неверный номер.")
                continue
        except ValueError:
            print("Введите число.")
            continue

        fwd_path = selected['fwd']
        rev_path = selected['rev']
        cons_path = selected['cons']
        prefix = selected['prefix']

        all_X, all_y, all_meta = process_triple(
            fwd_path, rev_path, cons_path, "output",
            all_X, all_y, all_meta
        )
        processed_prefixes.add(prefix)

    # Сохранение объединённого датасета
    if all_X:
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        meta_combined = np.concatenate(all_meta, axis=0)
        os.makedirs("datasets", exist_ok=True)
        np.save("datasets/X.npy", X_combined)
        np.save("datasets/y.npy", y_combined)
        np.save("datasets/meta.npy", meta_combined)
        logger.info(f"=== Итоговый датасет сохранён: {len(y_combined)} примеров ===")
    else:
        logger.info("Датасет пуст.")

    logger.info("=== Пайплайн завершён ===")


if __name__ == "__main__":
    main()