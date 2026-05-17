# src/dataset_builder.py
import os
import numpy as np
from src.logger import setup_logger

logger = setup_logger(__name__)

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

def build_dataset(mapping_cons, fwd_reader, rev_reader, window_size=120, output_dir="datasets"):
    """
    Создаёт обучающий датасет из карты сопоставления консенсуса с пиками.
    mapping_cons: список словарей (результат align_and_map)
        каждый содержит: cons_base, fwd_peak_idx, rev_peak_idx, fwd_base, rev_base
    fwd_reader, rev_reader: объекты AB1Reader (уже обрезанные)
    window_size: полуширина окна в сэмплах (итоговое окно 2*window_size)
    output_dir: папка для сохранения .npy файлов. Если None – возвращает X, y, meta.
    Возвращает X, y, meta (numpy массивы) или (None, None, None), если примеров нет.
    """
    X_list, y_list, meta_list = [], [], []

    for m in mapping_cons:
        if m['fwd_peak_idx'] is None or m['rev_peak_idx'] is None:
            continue
        if m['cons_base'] not in BASE_MAP:
            continue

        # Извлекаем окна
        fwd_win, _ = fwd_reader.get_window(m['fwd_peak_idx'], window_size)
        rev_win, _ = rev_reader.get_window(m['rev_peak_idx'], window_size)

        fwd_arr = np.array([fwd_win[base] for base in ["A", "C", "G", "T"]], dtype=np.float32)
        rev_arr = np.array([rev_win[base] for base in ["A", "C", "G", "T"]], dtype=np.float32)

        max_val = max(fwd_arr.max(), rev_arr.max(), 1e-6)
        fwd_arr /= max_val
        rev_arr /= max_val

        X = np.concatenate([fwd_arr, rev_arr], axis=0)  # (8, 2*window_size)
        y = BASE_MAP[m['cons_base']]

        match = 1 if (m['fwd_base'] == m['cons_base'] and m['rev_base'] == m['cons_base']) else 0
        fwd_ratio = fwd_reader.get_peak_ratio(m['fwd_peak_idx'])
        rev_ratio = rev_reader.get_peak_ratio(m['rev_peak_idx'])
        fwd_qual = fwd_reader.get_peak_quality(m['fwd_peak_idx'])
        rev_qual = rev_reader.get_peak_quality(m['rev_peak_idx'])

        meta = [
            match,
            fwd_ratio,
            rev_ratio,
            fwd_qual if fwd_qual is not None else -1,
            rev_qual if rev_qual is not None else -1
        ]
        X_list.append(X)
        y_list.append(y)
        meta_list.append(meta)

    if not X_list:
        logger.warning("Не создано ни одного примера для датасета!")
        return None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = np.array(meta_list, dtype=np.float32)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y.npy"), y)
        np.save(os.path.join(output_dir, "meta.npy"), meta)
        logger.info(f"Датасет сохранён в {output_dir}")
    else:
        logger.info(f"Датасет возвращён без сохранения: {len(y)} примеров")

    return X, y, meta