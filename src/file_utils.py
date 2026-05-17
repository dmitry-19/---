# src/file_utils.py
import os
import re
from src.logger import setup_logger

logger = setup_logger(__name__)

FASTA_EXTENSIONS = {'.fasta', '.fas', '.fa', '.fna', '.txt'}

def find_all_files(data_dir):
    """Возвращает списки всех AB1 и FASTA-файлов."""
    if not os.path.isdir(data_dir):
        return [], []
    all_files = os.listdir(data_dir)
    ab1_files = []
    cons_files = []
    for f in all_files:
        full = os.path.join(data_dir, f)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext == '.ab1':
            ab1_files.append(full)
        elif ext in FASTA_EXTENSIONS:
            cons_files.append(full)
    return sorted(ab1_files), sorted(cons_files)

def find_all_triples(data_dir):
    """
    Ищет все возможные тройки (fwd, rev, cons) по шаблону:
    префикс_sc1.ab1, префикс_sc2.ab1, префикс_*consensus.fasta (или другие расширения)
    Возвращает список словарей с ключами fwd, rev, cons, prefix.
    """
    ab1_files, cons_files = find_all_files(data_dir)
    triples = []
    ab1_map = {}
    for f in ab1_files:
        basename = os.path.basename(f)
        match = re.match(r'(.+)(_sc[12])\.ab1$', basename)
        if match:
            prefix = match.group(1)
            suf = match.group(2)
            if prefix not in ab1_map:
                ab1_map[prefix] = {}
            ab1_map[prefix][suf] = f
    for prefix, parts in ab1_map.items():
        if '_sc1' not in parts or '_sc2' not in parts:
            continue
        cons_candidates = [
            f for f in cons_files
            if os.path.basename(f).startswith(prefix) and
            any(ext in f for ext in FASTA_EXTENSIONS)
        ]
        if cons_candidates:
            cons = cons_candidates[0]
            triples.append({
                'fwd': parts['_sc1'],
                'rev': parts['_sc2'],
                'cons': cons,
                'prefix': prefix
            })
    return triples

def get_unprocessed_triples(data_dir, processed_prefixes):
    """Возвращает тройки, чьи префиксы не в processed_prefixes."""
    all_triples = find_all_triples(data_dir)
    return [t for t in all_triples if t['prefix'] not in processed_prefixes]