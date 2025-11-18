import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from morgan_fp import BitConvGRUBranch  

class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class AttentionPooler(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.q)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

    def forward(self, x):  
        B = x.size(0)
        q = self.q.expand(B, -1, -1)   
        h, _ = self.mha(q, x, x)       
        return h.squeeze(1)           

class ConvBlock1D(nn.Module):
   
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            groups=d_model, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):        
        res = x
        z = self.ln(x)
        z = z.transpose(1, 2)    
        z = self.dw(z)
        z = self.bn(z)
        z = self.act(z)
        z = self.pw(z)           
        z = z.transpose(1, 2)    
        z = self.drop(z)
        return res + z

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, conv_kernel=5, conv_dropout=0.1, conv_layers=1):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncodingBF(d_model, max_len=512, dropout=0.5)
        self.conv = nn.Sequential(
            *[ConvBlock1D(d_model, kernel_size=conv_kernel, dropout=conv_dropout) for _ in range(conv_layers)]
        )
    def forward(self, input_ids):
        x = self.src_emb(input_ids)   
        x = self.pos_emb(x)          
        x = self.conv(x)             
        return x

class peptide(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads,
                 max_len=50, conv_kernel=5, conv_dropout=0.1, conv_layers=1):
        super().__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model,
                                  conv_kernel=conv_kernel,
                                  conv_dropout=conv_dropout,
                                  conv_layers=conv_layers)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.pooler = AttentionPooler(d_model, n_heads=max(1, n_heads // 2), dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, d_model)
        )
    def forward(self, input_ids):
        x = self.emb(input_ids)                     
        x = self.transformer_blocks(x)              
        x = self.pooler(x)                         
        feats = self.fc(x)                         
        return feats

class ESM_MLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class GlobalRefineGatedSum(nn.Module):

    def __init__(self, feature_dim: int):
        super().__init__()
        D = feature_dim
   
        self.mix = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1, bias=True)
       
        self.gate1 = nn.Linear(2*D, D)
        self.gate2 = nn.Linear(2*D, D)
        self.gate3 = nn.Linear(2*D, D)
        self.ln = nn.LayerNorm(D)     
        self.proj = nn.Linear(D, D)   

    def forward(self, feats):           
        f1, f2, f3 = feats[:,0,:], feats[:,1,:], feats[:,2,:]          


        x = feats                    
        refined = self.mix(x)        

        h_sum = refined.sum(dim=1)   
        h_sum = self.ln(h_sum)      

        r1, r2, r3 = refined[:,0,:], refined[:,1,:], refined[:,2,:]  
        g1 = self.gate1(torch.cat([r1, h_sum], dim=-1))   
        g2 = self.gate2(torch.cat([r2, h_sum], dim=-1))  
        g3 = self.gate3(torch.cat([r3, h_sum], dim=-1))   

        gates = torch.stack([g1, g2, g3], dim=1)          
        gates = torch.softmax(gates, dim=1)               
        a, b, c = gates[:,0,:], gates[:,1,:], gates[:,2,:] 

        fused = a*f1 + b*f2 + c*f3                        
        return self.proj(fused)                          

class piv_Model(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len,
                 esm_in_dim, fp_bits_total, fusion_heads=4):
        super().__init__()
        self.peptide_model = peptide(vocab_size, d_model, d_ff, n_layers, n_heads, max_len)
        self.esm_branch = ESM_MLPBranch(in_dim=esm_in_dim, out_dim=d_model)

        self.bit_branch = BitConvGRUBranch(
            in_bits=fp_bits_total,
            out_dim=d_model,
            conv_channels=64,
            kernel_sizes=(3,5,7,9),
            conv_stride=4,      
            gru_hidden=128,
            gru_layers=2,
            bidirectional=True,
            dropout=0.2
        )

        self.post_ln1 = nn.LayerNorm(d_model)
        self.post_ln2 = nn.LayerNorm(d_model)
        self.post_ln3 = nn.LayerNorm(d_model)

        self.fusion = GlobalRefineGatedSum(feature_dim=d_model)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, input_ids, esm_features, fp_features, device=None):
        f1 = self.post_ln1(self.peptide_model(input_ids))     
        f2 = self.post_ln2(self.esm_branch(esm_features))     
        f3 = self.post_ln3(self.bit_branch(fp_features))      
        feats = torch.stack([f1, f2, f3], dim=1)              
        fused = self.fusion(feats)                           
        logits = self.classifier(fused)                      
        return logits








