import numpy as np
import torch
import torch.utils.data as Data
from typing import List, Dict
from atom_feature import aa_dict        
from morgan_fp import (
    morgan_from_seq_by_residue_or,
    maccs_from_seq_by_residue_or,
)

Pep_residue2idx = {
    '[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
    'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10,
    'K': 11, 'L': 12, 'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22,
}

def transform_Pep_to_index(sequences: List[str], residue2idx: Dict[str, int]):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq]
        token_index.append(seq_id)
    return token_index

def pad_sequence(token_list: List[List[int]], max_len=51):
    data = []
    for tokens in token_list:
        seq = [Pep_residue2idx['[CLS]']] + tokens
        n_pad = max_len - len(seq)
        seq.extend([Pep_residue2idx['[PAD]']] * max(n_pad, 0))
        data.append(seq[:max_len])
    return data

def load_data_from_fasta(file_path: str, max_len=51, ecfp_bits=2048, ecfp_radius=2, use_maccs=True):
    sequences = []
    labels = []
    with open(file_path, 'r') as f:
        current_seq = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                header = line[1:]
                label = 1 if header.lower().startswith('pos') else 0
                labels.append(label)
                current_seq = ''
                sequences.append(current_seq)
            else:
                sequences[-1] += line

    indexed_sequences = transform_Pep_to_index(sequences, Pep_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences, max_len=max_len)

    cache_morgan, cache_maccs = {}, {}
    ecfp_list = [
        morgan_from_seq_by_residue_or(seq, aa_smiles=aa_dict, n_bits=ecfp_bits, radius=ecfp_radius, cache=cache_morgan)
        for seq in sequences
    ]  
    if use_maccs:
        maccs_list = [
            maccs_from_seq_by_residue_or(seq, aa_smiles=aa_dict, cache=cache_maccs)
            for seq in sequences
        ]  
        fp_features = np.concatenate([np.stack(ecfp_list, axis=0), np.stack(maccs_list, axis=0)], axis=1)
    else:
        fp_features = np.stack(ecfp_list, axis=0)

    fp_features = fp_features.astype(np.float32)

    return padded_sequences, fp_features, labels

def load_esm_features(path):

    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32)

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, esm_features, fp_features, labels):
        assert len(input_ids) == len(esm_features) == len(fp_features) == len(labels), \
            f"length mismatch: ids={len(input_ids)}, esm={len(esm_features)}, fp={len(fp_features)}, y={len(labels)}"
        self.input_ids = input_ids
        self.esm_features = esm_features
        self.fp_features = fp_features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.esm_features[idx], dtype=torch.float32),
            torch.tensor(self.fp_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

