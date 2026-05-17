# src/reader.py
import numpy as np
from Bio import SeqIO
from src.logger import setup_logger

logger = setup_logger(__name__)


class AB1Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.record = None
        self.traces = None  # dict: {'A': array, 'C': array, 'G': array, 'T': array}
        self.peaks = None  # массив индексов пиков
        self.sequence = None  # строка вызванных оснований
        self.qualities = None  # массив Phred-качеств (или None, если нет)
        self._load()

    def _load(self):
        logger.info(f"Чтение файла: {self.file_path}")
        self.record = SeqIO.read(self.file_path, "abi")
        raw = self.record.annotations["abif_raw"]

        # Порядок каналов
        fwo = raw.get("FWO_1", None)
        if fwo is not None:
            order = fwo.decode("ascii") if isinstance(fwo, bytes) else fwo
            logger.debug(f"Порядок каналов из FWO_1: {order}")
        else:
            order = "GATC"
            logger.debug("FWO_1 отсутствует, используется стандартный порядок GATC")

        self.traces = {
            "G": np.array(raw["DATA9"], dtype=np.float64),
            "A": np.array(raw["DATA10"], dtype=np.float64),
            "T": np.array(raw["DATA11"], dtype=np.float64),
            "C": np.array(raw["DATA12"], dtype=np.float64),
        }
        self.peaks = np.array(raw["PLOC2"])
        self.sequence = str(self.record.seq)

        # Качества: обрабатываем оба формата
        raw_qual = raw.get("PCON2", None)
        if raw_qual is not None:
            if isinstance(raw_qual, bytes):
                try:
                    qual_arr = np.array([ord(c) - 33 for c in raw_qual.decode('ascii', errors='replace')],
                                        dtype=np.float64)
                except Exception:
                    logger.warning("Не удалось декодировать PCON2 как ASCII-строку, качества отключены.")
                    qual_arr = np.array([])
            elif isinstance(raw_qual, str):
                qual_arr = np.array([ord(c) - 33 for c in raw_qual], dtype=np.float64)
            else:
                qual_arr = np.array(raw_qual, dtype=np.float64)

            if qual_arr.ndim == 0:
                self.qualities = None
            elif qual_arr.size != len(self.peaks):
                logger.warning(
                    f"Размер PCON2 ({qual_arr.size}) не совпадает с числом пиков ({len(self.peaks)}). Качества отключены.")
                self.qualities = None
            else:
                self.qualities = qual_arr
        else:
            self.qualities = None

        logger.info(f"Успешно загружено: seq_len={len(self.sequence)}, "
                    f"peaks={len(self.peaks)}, traces_len={len(self.traces['A'])}, "
                    f"qualities={'есть' if self.qualities is not None else 'нет'}")

    def get_window(self, peak_index, window_size=120):
        if peak_index < 0 or peak_index >= len(self.peaks):
            raise IndexError(f"Индекс пика {peak_index} за пределами")
        center = self.peaks[peak_index]
        start = max(0, center - window_size)
        end = min(len(self.traces["A"]), center + window_size)
        window = {}
        for base in ["A", "C", "G", "T"]:
            win = self.traces[base][start:end]
            if len(win) < 2 * window_size:
                padded = np.zeros(2 * window_size, dtype=np.float64)
                offset = (2 * window_size - len(win)) // 2
                padded[offset:offset + len(win)] = win
                win = padded
            window[base] = win
        return window, center

    def get_peak_quality(self, peak_index):
        if self.qualities is not None and 0 <= peak_index < self.qualities.size:
            return self.qualities[peak_index]
        return None

    def get_peak_ratio(self, peak_index):
        center = self.peaks[peak_index]
        vals = [self.traces[base][center] for base in ["A", "C", "G", "T"]]
        vals.sort()
        # Если второй пик близок к нулю, возвращаем большое конечное число
        if vals[-2] < 1e-6:
            return 100.0
        return vals[-1] / vals[-2]

    def find_quality_trim_bounds(self, qual_threshold=20, min_ratio=1.5, margin=2):
        n_peaks = len(self.peaks)
        if self.qualities is not None and self.qualities.size == n_peaks:
            good = np.where(self.qualities >= qual_threshold)[0]
        else:
            ratios = np.array([self.get_peak_ratio(i) for i in range(n_peaks)])
            good = np.where(ratios >= min_ratio)[0]

        if len(good) == 0:
            return 0, n_peaks - 1

        start = max(0, good[0] - margin)
        end = min(n_peaks - 1, good[-1] + margin)
        return start, end

    def trim(self, start_peak, end_peak):
        import copy
        new_reader = copy.copy(self)
        new_reader.sequence = self.sequence[start_peak:end_peak + 1]
        new_reader.peaks = self.peaks[start_peak:end_peak + 1]
        if self.qualities is not None:
            new_reader.qualities = self.qualities[start_peak:end_peak + 1]
        return new_reader