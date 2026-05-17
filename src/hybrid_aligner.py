# src/hybrid_aligner.py
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from src.logger import setup_logger

logger = setup_logger(__name__)


def build_gapped_seq(ref, query, aligned):
    """
    Восстанавливает выровненные строки с гэпами.
    Аналогична функции из aligner.py, используется в global_letter_alignment.
    """
    ref_gapped = []
    query_gapped = []
    ref_pos = 0
    query_pos = 0

    for (r_start, r_end), (q_start, q_end) in zip(aligned[0], aligned[1]):
        while ref_pos < r_start:
            ref_gapped.append(ref[ref_pos])
            query_gapped.append("-")
            ref_pos += 1
        while query_pos < q_start:
            ref_gapped.append("-")
            query_gapped.append(query[query_pos])
            query_pos += 1
        while ref_pos < r_end:
            ref_gapped.append(ref[ref_pos])
            ref_pos += 1
        while query_pos < q_end:
            query_gapped.append(query[query_pos])
            query_pos += 1

    # Хвосты
    while ref_pos < len(ref):
        ref_gapped.append(ref[ref_pos])
        query_gapped.append("-")
        ref_pos += 1
    while query_pos < len(query):
        ref_gapped.append("-")
        query_gapped.append(query[query_pos])
        query_pos += 1

    return "".join(ref_gapped), "".join(query_gapped)


def build_profile(reader, peak_indices):
    """
    Строит профили пиков — массив (n_peaks, 4) нормированных интенсивностей
    каналов A, C, G, T в точках пиков.
    """
    profiles = []
    for idx in peak_indices:
        if idx is None or idx < 0 or idx >= len(reader.peaks):
            profiles.append(np.zeros(4))
            continue
        center = reader.peaks[idx]
        vals = np.array([reader.traces[base][center] for base in ["A", "C", "G", "T"]], dtype=np.float64)
        s = vals.sum()
        if s > 0:
            vals /= s
        profiles.append(vals)
    return np.array(profiles)


def global_letter_alignment(seq_fwd, seq_rev):
    """
    Глобальное выравнивание двух последовательностей (уже реверс-комплементированной)
    с аффинными штрафами и свободными концами.
    Возвращает строки выравнивания.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    aligner.open_end_gap_score = 0      # свободные концы
    aligner.extend_end_gap_score = 0

    aln = aligner.align(seq_fwd, seq_rev)[0]
    fwd_aln, rev_aln = build_gapped_seq(seq_fwd, seq_rev, aln.aligned)
    logger.debug(f"Глобальное выравнивание длиной {len(fwd_aln)}")
    return fwd_aln, rev_aln


def get_conflict_regions(fwd_aln, rev_aln, context=5):
    """
    Находит регионы, где есть гэпы или несовпадения, расширяет на context позиций.
    Возвращает список кортежей (start, end) в координатах выравнивания.
    """
    conflicts = []
    in_conflict = False
    start = 0
    for i, (f, r) in enumerate(zip(fwd_aln, rev_aln)):
        if f != r or f == '-' or r == '-':
            if not in_conflict:
                start = max(0, i - context)
                in_conflict = True
        else:
            if in_conflict:
                conflicts.append((start, min(len(fwd_aln)-1, i + context)))
                in_conflict = False
    if in_conflict:
        conflicts.append((start, len(fwd_aln)-1))

    # Объединение пересекающихся регионов
    merged = []
    for region in sorted(conflicts):
        if not merged or region[0] > merged[-1][1] + 1:
            merged.append(region)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]))
    logger.info(f"Найдено конфликтных регионов: {len(merged)}")
    return merged


def extract_region_peaks(fwd_reader, rev_reader, fwd_aln, rev_aln, region):
    """
    По координатам выравнивания определяет, какие индексы пиков прямого и обратного ридов
    попадают в регион [start, end] (включая позиции с гэпами).
    Возвращает два списка индексов пиков (целые числа или None).
    """
    start, end = region
    fwd_peaks = []
    rev_peaks = []
    fwd_pos = 0
    rev_pos = 0

    for i in range(len(fwd_aln)):
        if i < start:
            if fwd_aln[i] != '-':
                fwd_pos += 1
            if rev_aln[i] != '-':
                rev_pos += 1
            continue
        if i > end:
            break

        if fwd_aln[i] != '-':
            fwd_peaks.append(fwd_pos)
            fwd_pos += 1
        else:
            fwd_peaks.append(None)

        if rev_aln[i] != '-':
            rev_peaks.append(rev_pos)
            rev_pos += 1
        else:
            rev_peaks.append(None)

    return fwd_peaks, rev_peaks


def _dtw_path(C, bandwidth=10):
    """
    Вычисляет DTW-путь для матрицы локальных расстояний C.
    bandwidth: максимальное отклонение от диагонали (Sakoe–Chiba band).
    Возвращает список пар (i, j) — оптимальный путь.
    """
    n, m = C.shape
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        j_min = max(1, i - bandwidth)
        j_max = min(m, i + bandwidth)
        for j in range(j_min, j_max+1):
            cost = C[i-1, j-1]
            D[i, j] = cost + min(D[i-1, j],      # вертикальный (пропуск j)
                                 D[i, j-1],      # горизонтальный (пропуск i)
                                 D[i-1, j-1])    # диагональный

    # Обратный ход
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        diag = D[i-1, j-1] if i>0 and j>0 else np.inf
        up   = D[i-1, j]   if i>0 else np.inf
        left = D[i, j-1]   if j>0 else np.inf
        if diag <= up and diag <= left:
            i -= 1; j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def local_dtw(profiles_fwd, profiles_rev, bandwidth=10):
    """
    DTW между двумя последовательностями профилей.
    Возвращает путь (список пар индексов) и стоимость.
    """
    n = len(profiles_fwd)
    m = len(profiles_rev)
    if n == 0 or m == 0:
        return [], 0.0

    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(profiles_fwd[i] - profiles_rev[j])

    path = _dtw_path(C, bandwidth=bandwidth)
    cost = sum(C[i, j] for i, j in path)
    return path, cost


def hybrid_align(fwd_reader, rev_reader):
    """
    Главная функция гибридного выравнивания.
    Возвращает:
        mapping: список словарей с ключами fwd_peak_idx, rev_peak_idx, fwd_base, rev_base
        indels: список словарей с ключами type, fwd_pos, rev_pos
    """
    # 1. Буквенное выравнивание
    seq_fwd = fwd_reader.sequence
    seq_rev = str(Seq(rev_reader.sequence).reverse_complement())
    fwd_aln, rev_aln = global_letter_alignment(seq_fwd, seq_rev)

    # 2. Конфликтные регионы
    conflict_regions = get_conflict_regions(fwd_aln, rev_aln)

    if not conflict_regions:
        # Нет конфликтов — простое сопоставление
        mapping = []
        fwd_idx = 0
        rev_idx = 0
        for f, r in zip(fwd_aln, rev_aln):
            f_peak = fwd_idx if f != '-' else None
            r_peak = rev_idx if r != '-' else None
            mapping.append({
                'fwd_peak_idx': f_peak,
                'rev_peak_idx': r_peak,
                'fwd_base': f if f != '-' else None,
                'rev_base': r if r != '-' else None
            })
            if f != '-': fwd_idx += 1
            if r != '-': rev_idx += 1
        return mapping, []

    # 3. Построение гибридного сопоставления
    mapping = []
    indels = []
    fwd_pos = 0
    rev_pos = 0
    last_end = -1

    for start, end in conflict_regions:
        # Добавляем участок до региона по буквенному выравниванию
        for i in range(last_end + 1, start):
            f = fwd_aln[i]; r = rev_aln[i]
            f_peak = fwd_pos if f != '-' else None
            r_peak = rev_pos if r != '-' else None
            mapping.append({
                'fwd_peak_idx': f_peak,
                'rev_peak_idx': r_peak,
                'fwd_base': f if f != '-' else None,
                'rev_base': r if r != '-' else None
            })
            if f != '-': fwd_pos += 1
            if r != '-': rev_pos += 1

        # Извлекаем пики для региона
        f_peaks_region, r_peaks_region = extract_region_peaks(
            fwd_reader, rev_reader, fwd_aln, rev_aln, (start, end)
        )
        f_valid = [p for p in f_peaks_region if p is not None]
        r_valid = [p for p in r_peaks_region if p is not None]

        if len(f_valid) == 0 or len(r_valid) == 0:
            # Одна сторона полностью из гэпов – оставляем буквенное
            for i in range(start, end+1):
                f = fwd_aln[i]; r = rev_aln[i]
                f_peak = fwd_pos if f != '-' else None
                r_peak = rev_pos if r != '-' else None
                mapping.append({
                    'fwd_peak_idx': f_peak,
                    'rev_peak_idx': r_peak,
                    'fwd_base': f if f != '-' else None,
                    'rev_base': r if r != '-' else None
                })
                if f != '-': fwd_pos += 1
                if r != '-': rev_pos += 1
        else:
            # DTW уточнение
            prof_f = build_profile(fwd_reader, f_valid)
            prof_r = build_profile(rev_reader, r_valid)
            path, cost = local_dtw(prof_f, prof_r)
            logger.debug(f"Регион {start}-{end}: DTW cost={cost:.3f}, длина пути={len(path)}")

            # Разбор пути
            i_prev, j_prev = -1, -1
            for i, j in path:
                if i == i_prev + 1 and j == j_prev + 1:
                    # диагональный
                    f_peak = f_valid[i]
                    r_peak = r_valid[j]
                    mapping.append({
                        'fwd_peak_idx': f_peak,
                        'rev_peak_idx': r_peak,
                        'fwd_base': fwd_reader.sequence[f_peak],
                        'rev_base': rev_reader.sequence[r_peak]
                    })
                elif i == i_prev + 1 and j == j_prev:
                    # вертикальный: пик в fwd, гэп в rev -> делеция в rev
                    f_peak = f_valid[i]
                    mapping.append({
                        'fwd_peak_idx': f_peak,
                        'rev_peak_idx': None,
                        'fwd_base': fwd_reader.sequence[f_peak],
                        'rev_base': None
                    })
                    indels.append({
                        'type': 'del_in_rev',
                        'fwd_pos': f_peak,
                        'rev_pos': None
                    })
                elif i == i_prev and j == j_prev + 1:
                    # горизонтальный: пик в rev, гэп в fwd -> делеция в fwd
                    r_peak = r_valid[j]
                    mapping.append({
                        'fwd_peak_idx': None,
                        'rev_peak_idx': r_peak,
                        'fwd_base': None,
                        'rev_base': rev_reader.sequence[r_peak]
                    })
                    indels.append({
                        'type': 'del_in_fwd',
                        'fwd_pos': None,
                        'rev_pos': r_peak
                    })
                i_prev, j_prev = i, j

            # Обновляем позиционные счётчики
            max_f = max((p for p in f_peaks_region if p is not None), default=None)
            max_r = max((p for p in r_peaks_region if p is not None), default=None)
            if max_f is not None:
                fwd_pos = max_f + 1
            if max_r is not None:
                rev_pos = max_r + 1

        last_end = end

    # Оставшиеся позиции после последнего региона
    for i in range(last_end + 1, len(fwd_aln)):
        f = fwd_aln[i]; r = rev_aln[i]
        f_peak = fwd_pos if f != '-' else None
        r_peak = rev_pos if r != '-' else None
        mapping.append({
            'fwd_peak_idx': f_peak,
            'rev_peak_idx': r_peak,
            'fwd_base': f if f != '-' else None,
            'rev_base': r if r != '-' else None
        })
        if f != '-': fwd_pos += 1
        if r != '-': rev_pos += 1

    logger.info(f"Гибридное выравнивание завершено. Инделов найдено: {len(indels)}")
    return mapping, indels