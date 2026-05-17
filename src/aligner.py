# src/aligner.py
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
import numpy as np
from src.logger import setup_logger

logger = setup_logger(__name__)

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

def build_gapped_seq(ref, query, aligned):
    """
    Восстанавливает выровненные строки из ref, query и списка aligned-блоков.
    aligned = [(ref_blocks), (query_blocks)], где каждый блок — (start, end)
    """
    ref_gapped = []
    query_gapped = []
    ref_pos = 0
    query_pos = 0

    for (r_start, r_end), (q_start, q_end) in zip(aligned[0], aligned[1]):
        # Символы до блока
        while ref_pos < r_start:
            ref_gapped.append(ref[ref_pos])
            query_gapped.append("-")
            ref_pos += 1
        while query_pos < q_start:
            ref_gapped.append("-")
            query_gapped.append(query[query_pos])
            query_pos += 1
        # Блок совпадения
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


def map_cons_to_peaks(cons_seq, query_seq):
    """
    Возвращает список длиной len(cons_seq), где для каждой позиции консенсуса
    указан индекс в query_seq (0-based) или None, если делеция.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    aligner.open_end_gap_score = 0
    aligner.extend_end_gap_score = 0

    aln = aligner.align(cons_seq, query_seq)[0]
    cons_gapped, query_gapped = build_gapped_seq(cons_seq, query_seq, aln.aligned)

    mapping = [None] * len(cons_seq)
    cons_idx = 0  # позиция в исходном cons_seq
    query_idx = 0 # позиция в исходном query_seq

    for c, q in zip(cons_gapped, query_gapped):
        if c != '-' and q != '-':
            mapping[cons_idx] = query_idx
            cons_idx += 1
            query_idx += 1
        elif c != '-' and q == '-':
            # делеция в query: основание консенсуса, но нет нуклеотида в риде
            cons_idx += 1
            # query_idx не меняется
        elif c == '-' and q != '-':
            # вставка в query: консенсус не имеет этой позиции
            query_idx += 1
        # случай c=='-' и q=='-' не встречается

    return mapping


def align_and_map(cons_seq, fwd_seq, rev_seq, fwd_peaks, rev_peaks):
    """
    Строит карту: для каждой позиции консенсуса возвращает словарь с cons_base,
    fwd_peak_idx, rev_peak_idx и вызванными основаниями (если есть).
    """
    logger.info("Сопоставление консенсуса с прямым ридом...")
    fwd_map = map_cons_to_peaks(cons_seq, fwd_seq)

    logger.info("Сопоставление консенсуса с обратным ридом...")
    rev_map = map_cons_to_peaks(cons_seq, rev_seq)

    mapping = []
    for i, base in enumerate(cons_seq):
        fwd_idx = fwd_map[i]
        rev_idx = rev_map[i]
        mapping.append({
            "cons_pos": i,
            "cons_base": base,
            "fwd_peak_idx": fwd_idx,
            "rev_peak_idx": rev_idx,
            "fwd_base": fwd_seq[fwd_idx] if fwd_idx is not None else None,
            "rev_base": rev_seq[rev_idx] if rev_idx is not None else None,
        })

    logger.info(f"Построена карта для {len(mapping)} позиций консенсуса.")
    return mapping