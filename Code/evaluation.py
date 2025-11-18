import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix

from dataset import (
    Pep_residue2idx,
    load_data_from_fasta,
    load_esm_features,
    MyDataSet
)
from model import piv_Model

def compute_metrics(all_labels, all_preds, all_probs):
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
        sen = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    else:
        sen = spe = 0.0
    return acc, sen, spe, mcc, auc

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in data_loader:
        input_ids, esm_features, fp_features, labels = batch
        input_ids = input_ids.to(device)
        esm_features = esm_features.to(device)
        fp_features = fp_features.to(device)
        labels = labels.to(device)

        logits = model(input_ids, esm_features, fp_features, device)
        probs = F.softmax(logits, dim=1)  # [B,2]
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds, all_probs)

def main():
    vocab_size = len(Pep_residue2idx)
    d_model = 256
    d_ff = 512
    n_layers = 2
    n_heads = 2
    max_len = 50

    ecfp_bits = 2048
    maccs_bits = 167
    fp_bits_total = ecfp_bits + maccs_bits

    test_sequences,  test_fp,  test_labels  = load_data_from_fasta(
        '.txt', max_len=max_len+1, ecfp_bits=ecfp_bits, ecfp_radius=2, use_maccs=True
    )
    test_esm  = load_esm_features('.txt')

    assert len(test_sequences) == len(test_fp) == len(test_labels) == len(test_esm), \
        f"length mismatch: seq={len(test_sequences)}, fp={len(test_fp)}, y={len(test_labels)}, esm={len(test_esm)}"

    esm_in_dim = test_esm.shape[1]
    test_dataset  = MyDataSet(test_sequences,  test_esm,  test_fp,  test_labels)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = piv_Model(
        vocab_size=vocab_size,
        d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
        esm_in_dim=esm_in_dim, fp_bits_total=fp_bits_total,  
        fusion_heads=4
    ).to(device)

    ckpt_path = ".pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)

    acc, sen, spe, mcc, auc = evaluate(model, test_loader, device)

    print("===== Evaluation (Test) =====")
    print(f"ACC : {acc:.4f}")
    print(f"SEN : {sen:.4f}")
    print(f"SPE : {spe:.4f}")
    print(f"MCC : {mcc:.4f}")
    print(f"AUC : {auc:.4f}")

if __name__ == "__main__":
    main()


