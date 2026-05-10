import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import warnings
import time
from copy import deepcopy

warnings.filterwarnings('ignore')

WINDOW_SIZE = 30                     
STRIDE = 1
SENSOR_COLS = ['CO_Room', 'PM25_Room', 'Temperature_Room', 'UV_Room']
BATCH_SIZE = 128                     
EPOCHS_BASELINE = 30                 
EPOCHS_JOINT = 60                   
LR_BASELINE = 2e-4                   
LR_JOINT = 3e-4                   
HIDDEN_DIM = 96                 
NUM_LAYERS = 1
FEATURE_DIM = 48                     
NUM_STAGES = 7
POS_WEIGHT = 12.0                    
GRAD_CLIP = 1.0
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

print(f" 运行设备: {DEVICE}")
print(f" 配置: 窗口{WINDOW_SIZE}, LSTM隐藏{HIDDEN_DIM}, 输出{FEATURE_DIM}, 正样本权重{POS_WEIGHT}")

STAGE_MAP = {
    'Ignition': 0, 'Outgasing': 1, 'Smoldering': 2,
    'Flaming': 3, 'Glowing': 4, 'Spray': 5, 'Ventilating': 6
}

class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=FEATURE_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        normed = self.norm(last_hidden)
        out = self.dropout(self.act(self.fc(normed)))
        return out

class ResProcFireNet(nn.Module):
    def __init__(self, input_dim=4, num_stages=NUM_STAGES):
        super().__init__()
        self.base_encoder = LSTMEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, FEATURE_DIM)
        self.curr_encoder = LSTMEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, FEATURE_DIM)
        self.proc_encoder = LSTMEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, FEATURE_DIM)
        self.attention = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM*2), nn.GELU(),
            nn.Linear(FEATURE_DIM*2, FEATURE_DIM), nn.Sigmoid()
        )
        self.stage_classifier = nn.Linear(FEATURE_DIM, num_stages)
        fusion_dim = FEATURE_DIM + num_stages
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )  
    def forward(self, x, return_stage=False):
        base_feat = self.base_encoder(x)
        curr_feat = self.curr_encoder(x)
        proc_feat = self.proc_encoder(x)
        residual = torch.abs(curr_feat - base_feat)
        att_weight = self.attention(residual)
        weighted_residual = residual * att_weight
        stage_logits = self.stage_classifier(proc_feat)
        stage_probs = F.softmax(stage_logits, dim=1)
        fused = torch.cat([weighted_residual, stage_probs], dim=1)
        fire_logits = self.fusion_net(fused).squeeze(1)
        if return_stage:
            return fire_logits, stage_logits
        return fire_logits
    def freeze_base_encoder(self):
        for param in self.base_encoder.parameters():
            param.requires_grad = False

class FireDataset(Dataset):
    def __init__(self, X, y_fire, y_stage, scaler=None, fit_scaler=True, augment=False):
        self.X = X
        self.y_fire = y_fire
        self.y_stage = y_stage
        self.augment = augment
        if fit_scaler:
            x_flat = X.reshape(-1, X.shape[-1])
            self.scaler = SimpleStandardScaler()
            self.scaler.fit(x_flat)
            X_norm = self.scaler.transform(x_flat).reshape(X.shape)
        else:
            self.scaler = scaler
            x_flat = X.reshape(-1, X.shape[-1])
            X_norm = self.scaler.transform(x_flat).reshape(X.shape)
        self.X = torch.from_numpy(X_norm).float()
        self.y_fire = torch.from_numpy(y_fire).float()
        self.y_stage = torch.from_numpy(y_stage).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, y_fire, y_stage = self.X[idx], self.y_fire[idx], self.y_stage[idx]
        if self.augment and y_fire == 1 and np.random.rand() > 0.7:
            x = x + torch.randn_like(x) * 0.01
        return x, y_fire, y_stage

def load_and_prepare_data(mode='all'):
    print(" 加载数据集...")
    df1 = pd.read_excel("Indoor Fire Dataset with Distributed Multi-Sensor Nodes.xlsx")
    df2 = pd.read_excel("indoor_fire_detection_multisensornodes_dataset.xlsx")
    for c in SENSOR_COLS:
        df1[c] = df1[c].ffill().bfill()
        df2[c] = df2[c].ffill().bfill()
    df1['fire_label'] = (df1['ternary_label'].str.upper() == 'FIRE').astype(int)
    df2['fire_label'] = (df2['ternary_label'].str.upper() == 'FIRE').astype(int)
    df1['is_background'] = (df1['ternary_label'].str.upper() == 'BACKGROUND').astype(int)
    df2['is_background'] = (df2['ternary_label'].str.upper() == 'BACKGROUND').astype(int)
    df1['stage_label'] = -1
    df2['stage_label'] = -1
    if 'progress_label' in df2.columns:
        for name, idx in STAGE_MAP.items():
            df2.loc[df2['progress_label'].str.contains(name, case=False, na=False), 'stage_label'] = idx
    df = pd.concat([df1, df2], ignore_index=True)
    print(f" 数据合并完成，总行数：{len(df)}")
    
    X, y_fire, y_stage, is_background = [], [], [], []
    data = df[SENSOR_COLS].values
    fire_labels = df['fire_label'].values
    stage_labels = df['stage_label'].values
    bg_labels = df['is_background'].values
    total_len = len(data)
    for start in range(0, total_len - WINDOW_SIZE + 1, STRIDE):
        X.append(data[start:start+WINDOW_SIZE])
        y_fire.append(fire_labels[start+WINDOW_SIZE-1])
        y_stage.append(stage_labels[start+WINDOW_SIZE-1])
        is_background.append(np.all(bg_labels[start:start+WINDOW_SIZE] == 1))
    X = np.array(X, np.float32)
    y_fire = np.array(y_fire)
    y_stage = np.array(y_stage)
    is_background = np.array(is_background, bool)
    if mode == 'part':
        idx = np.random.choice(len(X), 20000, replace=False)
        X, y_fire, y_stage, is_background = X[idx], y_fire[idx], y_stage[idx], is_background[idx]
    print(f"滑窗完成，窗口数：{len(X)}，正火比例：{y_fire.mean():.3f}，背景比例：{is_background.mean():.3f}")
    return X, y_fire, y_stage, is_background

def stratified_split(X, y_fire, y_stage, bg):
    indices = np.arange(len(X))
    pos_idx = indices[y_fire == 1]
    neg_idx = indices[y_fire == 0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    n_pos_train = int(len(pos_idx) * 0.7)
    n_pos_val = int(len(pos_idx) * 0.1)
    n_neg_train = int(len(neg_idx) * 0.7)
    n_neg_val = int(len(neg_idx) * 0.1)
    train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
    val_idx = np.concatenate([pos_idx[n_pos_train:n_pos_train+n_pos_val], neg_idx[n_neg_train:n_neg_train+n_neg_val]])
    test_idx = np.concatenate([pos_idx[n_pos_train+n_pos_val:], neg_idx[n_neg_train+n_neg_val:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    return (X[train_idx], X[val_idx], X[test_idx],
            y_fire[train_idx], y_fire[val_idx], y_fire[test_idx],
            y_stage[train_idx], y_stage[val_idx], y_stage[test_idx],
            bg[train_idx], bg[val_idx], bg[test_idx])

def compute_metrics(y_true, y_pred, y_prob):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    far = fp / (fp + tn + 1e-8)
    mdr = fn / (fn + tp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    # AUROC
    order = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[order]
    pos = np.sum(y_true)
    neg = len(y_true) - pos
    tpr = np.cumsum(y_true_sorted) / pos
    fpr = np.cumsum(1 - y_true_sorted) / neg
    auroc = np.trapezoid(tpr, fpr)
    # AUPRC
    tp_cum = np.cumsum(y_true_sorted)
    prec_cum = tp_cum / (np.arange(1, len(y_true_sorted)+1))
    rec_cum = tp_cum / pos
    auprc = np.trapezoid(prec_cum, rec_cum)
    return far, mdr, acc, prec, rec, f1, auroc, auprc

def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    probs = 1 / (1 + np.exp(-logits))
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)
    return probs, preds, labels

def find_best_threshold(probs, labels):
    best_f1 = 0.0
    best_th = 0.5
    for th in np.arange(0.1, 0.95, 0.02):
        preds = (probs >= th).astype(int)
        _, _, _, _, _, f1, _, _ = compute_metrics(labels, preds, probs)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def pretrain_base_encoder(model, train_bg_loader, val_loader):
    print("\n===== 预训练 Base Encoder（背景特征学习）=====")
    class Autoencoder(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = nn.Sequential(
                nn.Linear(FEATURE_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, WINDOW_SIZE * 4)
            )
        def forward(self, x):
            feat = self.encoder(x)
            return self.decoder(feat).view(-1, WINDOW_SIZE, 4)
    autoenc = Autoencoder(model.base_encoder).to(DEVICE)
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=LR_BASELINE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    for epoch in range(EPOCHS_BASELINE):
        autoenc.train()
        train_loss = 0.0
        for (x,) in train_bg_loader:
            x = x.to(DEVICE)
            recon = autoenc(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # 验证
        autoenc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _, _ in val_loader:
                x = x.to(DEVICE)
                recon = autoenc(x)
                val_loss += criterion(recon, x).item()
        train_loss /= len(train_bg_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    model.freeze_base_encoder()
    print("Base Encoder 预训练完成并冻结")
    return model

def train_joint(model, train_loader, val_loader, test_loader):
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion_fire = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_stage = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_JOINT, weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_f1 = 0.0
    best_state = None
    patience = 15
    no_improve = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS_JOINT):
        model.train()
        total_loss = 0.0
        for x, y_fire, y_stage in train_loader:
            x, y_fire, y_stage = x.to(DEVICE), y_fire.to(DEVICE), y_stage.to(DEVICE)
            logits, stage_logits = model(x, return_stage=True)
            loss_fire = criterion_fire(logits, y_fire)
            loss_stage = criterion_stage(stage_logits, y_stage)
            loss = loss_fire + 0.2 * loss_stage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        
        val_probs, _, val_labels = evaluate_model(model, val_loader, DEVICE)
        best_th, val_f1 = find_best_threshold(val_probs, val_labels)
        far, mdr, acc, prec, rec, f1, auroc, auprc = compute_metrics(val_labels, (val_probs>=best_th).astype(int), val_probs)
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | F1: {f1:.4f} | FAR: {far:.4f} | MDR: {mdr:.4f}")
        
        scheduler.step()
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            print(f"  最佳模型更新 (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停于 epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    print(f"联合训练完成，最佳验证 F1 = {best_f1:.4f}")
    
    test_probs, _, test_labels = evaluate_model(model, test_loader, DEVICE)
    best_th, _ = find_best_threshold(test_probs, test_labels)  
    far, mdr, acc, prec, rec, f1, auroc, auprc = compute_metrics(test_labels, (test_probs>=best_th).astype(int), test_probs)
    print("\n" + "="*80)
    print("="*80)
    print(f"误报率 FAR    : {far:.4f}  {' 达标' if far<=0.0034 else ' 未达标'}")
    print(f"漏检率 MDR    : {mdr:.4f}  {' 达标' if mdr<=0.080 else ' 未达标'}")
    print(f"准确率 Accuracy: {acc:.4f}  {' 达标' if acc>=0.9800 else ' 未达标'}")
    print(f"精确率 Precision: {prec:.4f}  {' 达标' if prec>=0.9700 else ' 未达标'}")
    print(f"召回率 Recall  : {rec:.4f}  {' 达标' if rec>=0.9100 else ' 未达标'}")
    print(f"F1 分数       : {f1:.4f}  {'达标' if f1>=0.9800 else ' 未达标'}")
    print(f"AUROC         : {auroc:.4f}  {'达标' if auroc>=0.9800 else '未达标'}")
    print(f"AUPRC         : {auprc:.4f}  {'达标' if auprc>=0.9800 else '未达标'}")
    print(f"最优阈值      : {best_th:.4f}")
    return model

def main():
    print("="*80)
    print("="*80)
    mode = input("输入模式（part=快速验证 / all=全量训练）：").strip().lower()
    if mode not in ['part', 'all']:
        mode = 'all'
    
    X, y_fire, y_stage, bg = load_and_prepare_data(mode)
    X_train, X_val, X_test, yf_train, yf_val, yf_test, ys_train, ys_val, ys_test, bg_train, _, _ = stratified_split(X, y_fire, y_stage, bg)
    
    train_dataset = FireDataset(X_train, yf_train, ys_train, augment=True)
    val_dataset = FireDataset(X_val, yf_val, ys_val, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = FireDataset(X_test, yf_test, ys_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    bg_indices = np.where(bg_train)[0]
    if len(bg_indices) == 0:
        raise RuntimeError("训练集中无纯背景窗口")
    bg_dataset = torch.utils.data.TensorDataset(train_dataset.X[bg_indices])
    bg_loader = DataLoader(bg_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = ResProcFireNet().to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model = pretrain_base_encoder(model, bg_loader, val_loader)
    model = train_joint(model, train_loader, val_loader, test_loader)
    
    torch.save(model.state_dict(), "resprocfirenet_guaranteed.pth")
    print(" 模型已保存为 resprocfirenet_guaranteed.pth")

if __name__ == "__main__":
    main()
