import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve

# Paths
ROOT = Path("/workspace")
OUT_DIR = ROOT / 'project'
SPLITS_DIR = OUT_DIR / "splits"
MODELS_DIR = OUT_DIR / "models"
CORE_COLS = ["ENANTEMA", "PÓLIPO", "ÚLCERA", "EROSÃO", "MICRONODULARIDADE"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(fold):
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    
    class EndoDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            self.labels = self.df[CORE_COLS].values.astype(np.float32)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            img_path = str(ROOT / "Dev" / "Data" / "Imgs" / self.df.loc[idx, "img_name"])
            try: img = Image.open(img_path).convert("RGB")
            except: img = Image.new("RGB", (224, 224))
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]

    df = pd.read_csv(SPLITS_DIR / f"fold_{fold}.csv")
    tf_te = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    df_va = df[df['split'] == 'val'].reset_index(drop=True)
    df_te = df[df['split'] == 'test'].reset_index(drop=True)
    
    ds_va = EndoDataset(df_va, tf_te)
    ds_te = EndoDataset(df_te, tf_te)
    
    ldr_va = DataLoader(ds_va, batch_size=32, shuffle=False)
    ldr_te = DataLoader(ds_te, batch_size=32, shuffle=False)
    return ldr_va, ldr_te, df_te

def inference(model, loader):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(DEVICE)))
            all_p.append(p.cpu().numpy())
            all_t.append(y.numpy())
    return np.vstack(all_p), np.vstack(all_t)

def get_thresholds(p_va, t_va):
    thrs = []
    for c in range(p_va.shape[1]):
        pr, rc, th = precision_recall_curve(t_va[:,c], p_va[:,c])
        f1 = 2 * (pr * rc) / (pr + rc + 1e-8)
        best_th = th[np.argmax(f1)] if len(th) > 0 else 0.5
        thrs.append(best_th)
    return np.array(thrs)

# Evaluaremos Fold 4 (ou 0, tentemos 0)
fold = 0
ldr_va, ldr_te, df_te = get_dataloaders(fold)

# Load M0
m0 = models.resnet50(weights=None)
m0.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m0.fc.in_features, len(CORE_COLS)))
m0.load_state_dict(torch.load(MODELS_DIR / f"M0_BCE_fold{fold}.pth", map_location=DEVICE, weights_only=True))
m0.to(DEVICE)

p0_va, t0_va = inference(m0, ldr_va)
p0_te, t0_te = inference(m0, ldr_te)
th0 = get_thresholds(p0_va, t0_va)
y0_pred = (p0_te >= th0).astype(int)

# Load M2
m2 = models.resnet50(weights=None)
m2.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m2.fc.in_features, len(CORE_COLS)))
ckpt2 = torch.load(MODELS_DIR / f"M2_fold{fold}.pt", map_location=DEVICE, weights_only=True)
m2.load_state_dict(ckpt2["model_state_dict"])
m2.to(DEVICE)

p2_va, t2_va = inference(m2, ldr_va)
p2_te, t2_te = inference(m2, ldr_te)
th2 = get_thresholds(p2_va, t2_va)
y2_pred = (p2_te >= th2).astype(int)

true_labels = t0_te # same as t2_te

# Find cases
cases = []

def labels_to_str(binary_vec):
    active = [CORE_COLS[i] for i in range(5) if binary_vec[i] == 1]
    return " + ".join(active) if active else "none"

for i in range(len(true_labels)):
    gt = true_labels[i]
    p0 = y0_pred[i]
    p2 = y2_pred[i]
    
    # Condição 1: M2 acerta classe rara e M0 erra (Pólipo ou Micronod)
    if (gt[1]==1 and p2[1]==1 and p0[1]==0) or (gt[4]==1 and p2[4]==1 and p0[4]==0):
        cases.append((i, "M2 recuperou classe rara", labels_to_str(gt), labels_to_str(p0), labels_to_str(p2)))
        continue
        
    # Condição 2: Falso positivo do M2 em classe rara
    if (gt[1]==0 and p2[1]==1 and p0[1]==0) or (gt[4]==0 and p2[4]==1 and p0[4]==0):
        cases.append((i, "M2 gerou falso positivo", labels_to_str(gt), labels_to_str(p0), labels_to_str(p2)))
        continue
        
    # Condição 3: M2 acerta co-ocorrência que M0 quebra
    if sum(gt) > 1 and sum(p2) == sum(gt) and np.array_equal(p2, gt) and not np.array_equal(p0, gt):
        cases.append((i, "M2 acertou co-ocorrência", labels_to_str(gt), labels_to_str(p0), labels_to_str(p2)))
        continue
        
df_cases = pd.DataFrame(cases, columns=["img_idx", "category", "Ground truth", "M0_BCE", "M2_coo"])
print(df_cases.groupby("category").first())
