from typing import List, Tuple, Dict
import numpy as np

def _lazy_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        return Chem, AllChem, MACCSkeys
    except Exception as e:
        raise ImportError(
        ) from e

def smiles_to_morgan_bits(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    Chem, AllChem, _ = _lazy_import_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def smiles_to_maccs_bits(smiles: str) -> np.ndarray:
    Chem, _, MACCSkeys = _lazy_import_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((167,), dtype=np.uint8)
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def morgan_from_seq_by_residue_or(seq: str, aa_smiles: Dict[str, str],
                                  n_bits: int = 2048, radius: int = 2,
                                  cache: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    if cache is None:
        cache = {}
    out = np.zeros((n_bits,), dtype=np.uint8)
    for ch in seq.lower():
        smi = aa_smiles.get(ch, None)
        if not smi: continue
        if smi not in cache:
            cache[smi] = smiles_to_morgan_bits(smi, n_bits=n_bits, radius=radius)
        out |= cache[smi]
    return out.astype(np.float32)

def maccs_from_seq_by_residue_or(seq: str, aa_smiles: Dict[str, str],
                                 cache: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    if cache is None:
        cache = {}
    out = np.zeros((167,), dtype=np.uint8)
    for ch in seq.lower():
        smi = aa_smiles.get(ch, None)
        if not smi: continue
        if smi not in cache:
            cache[smi] = smiles_to_maccs_bits(smi)
        out |= cache[smi]
    return out.astype(np.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BitConvGRUBranch(nn.Module):
    def __init__(
        self,
        in_bits: int,
        out_dim: int = 256,
        conv_channels: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        conv_stride: int = 4,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_bits = in_bits

        self.input_bn = nn.BatchNorm1d(1)

        self.convs = nn.ModuleList([
            nn.Conv1d(1, conv_channels, k, stride=conv_stride, padding=k // 2)
            for k in kernel_sizes
        ])
        self.conv_bns = nn.ModuleList([nn.BatchNorm1d(conv_channels) for _ in kernel_sizes])

        self.total_c = conv_channels * len(kernel_sizes)

        self.gru = nn.GRU(
            input_size=self.total_c,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = gru_hidden * (2 if bidirectional else 1)

        self.proj = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        if x.dim() != 2 or x.size(1) != self.in_bits:
            raise ValueError(f"Expected x of shape [B, {self.in_bits}], got {tuple(x.shape)}")

        x = x.unsqueeze(1)
        x = self.input_bn(x)

        conv_outs = []
        for conv, bn in zip(self.convs, self.conv_bns):
            h = F.gelu(bn(conv(x)))      
            conv_outs.append(h)
        h = torch.cat(conv_outs, dim=1)  

        h = h.transpose(1, 2)            
        h, _ = self.gru(h)               

        avg = h.mean(dim=1)              
        mx  = h.max(dim=1).values        
        z = torch.cat([avg, mx], dim=-1) 
        return self.proj(z)              


