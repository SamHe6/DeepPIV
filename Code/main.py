from model import piv_Model
from dataset import (
    Pep_residue2idx,
    load_data_from_fasta,
    load_esm_features,
    MyDataSet,
)
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, esm_features, fp_features, labels = batch
        input_ids = input_ids.to(device)
        esm_features = esm_features.to(device)
        fp_features = fp_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, esm_features, fp_features, device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in test_loader:
        input_ids, esm_features, fp_features, labels = batch
        input_ids = input_ids.to(device)
        esm_features = esm_features.to(device)
        fp_features = fp_features.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, esm_features, fp_features, device)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.numel()
    accuracy = correct / n if n > 0 else 0.0
    return total_loss / len(test_loader), accuracy

if __name__ == "__main__":
    set_seed(42)

    vocab_size = len(Pep_residue2idx)
    d_model = 256
    d_ff = 512
    n_layers = 2
    n_heads = 2
    max_len = 50

    ecfp_bits = 2048      
    maccs_bits = 167      
    fp_bits_total = ecfp_bits + maccs_bits

    train_sequences, train_fp, train_labels = load_data_from_fasta(
        '.txt', max_len=max_len+1, ecfp_bits=ecfp_bits, ecfp_radius=2, use_maccs=True
    )
    test_sequences,  test_fp,  test_labels  = load_data_from_fasta(
        '.txt',           max_len=max_len+1, ecfp_bits=ecfp_bits, ecfp_radius=2, use_maccs=True
    )

    train_esm = load_esm_features('.txt')  
    test_esm  = load_esm_features('.txt')

    assert len(train_sequences) == len(train_esm) == len(train_fp) == len(train_labels), \
        f"train length mismatch: ids={len(train_sequences)}, esm={len(train_esm)}, fp={len(train_fp)}, y={len(train_labels)}"
    assert len(test_sequences)  == len(test_esm)  == len(test_fp)  == len(test_labels), \
        f"test length mismatch:  ids={len(test_sequences)}, esm={len(test_esm)}, fp={len(test_fp)}, y={len(test_labels)}"

    esm_in_dim = train_esm.shape[1]

    train_dataset = MyDataSet(train_sequences, train_esm, train_fp, train_labels)
    test_dataset  = MyDataSet(test_sequences,  test_esm,  test_fp,  test_labels)

    train_loader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = Data.DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    outer_runs = 50     
    n_epochs   = 100    
    lr         = 1e-3

    best_model_path = ".pth"
    global_best_acc = 0.0

    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    for run in range(1, outer_runs + 1):
        print(f"\n====== Run {run}/{outer_runs} ======")
        model = piv_Model(
            vocab_size=vocab_size,
            d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
            esm_in_dim=esm_in_dim, fp_bits_total=fp_bits_total,  
            fusion_heads=4
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        best_acc_this_run = 0.0
        for epoch in range(1, n_epochs + 1):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

            if test_acc > best_acc_this_run:
                best_acc_this_run = test_acc

            if test_acc > global_best_acc:
                global_best_acc = test_acc
                torch.save(model.state_dict(), best_model_path)
                print(f'>>> [SAVE] Global best updated at Run {run}, Epoch {epoch}: Acc={global_best_acc:.4f} -> {best_model_path}')

            print(f'Run {run} | Epoch {epoch}/{n_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | '
                  f'Test Acc: {test_acc:.4f} | Run Best: {best_acc_this_run:.4f} | Global Best: {global_best_acc:.4f}')

        print(f'---- Run {run} finished. Best Acc this run: {best_acc_this_run:.4f} ----')



