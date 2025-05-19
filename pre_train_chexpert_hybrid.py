import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from torchvision import models
from datetime import datetime
from PIL import Image
import timm
from pathlib import Path
import copy
from tqdm import tqdm
import h5py

# ====== Config ======
save_dir = f"./chexpert_models/chexpert_pretrain_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
IMG_DIR = Path("/projects/eclarson/stems/STEMC/EHR/cheXpert/chexpertchestxrays-u20210408")
CSV_PATH = Path("/projects/eclarson/stems/STEMC/EHR/cheXpert/chexpertchestxrays-u20210408/train_visualCheXbert.csv")
VALID_CSV_PATH = Path("/projects/eclarson/stems/STEMC/EHR/cheXpert/chexpertchestxrays-u20210408/valid.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 60
NUM_CLASSES = 14
USE_SMALL_SAMPLE = False
EARLY_STOPPING_PATIENCE = 3

# ====== Dataset ======
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Path'])

        self.labels = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
            'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]
        df[self.labels] = df[self.labels].fillna(0).replace(-1, 0)

        df["PatientID"] = df["Path"].apply(lambda p: p.split("/")[2])
        df["StudyIndex"] = df["Path"].apply(lambda p: int(p.split("/")[3].replace("study", "")))
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        if USE_SMALL_SAMPLE:
            df = df.groupby("PatientID").head(1).iloc[:100]
        self.patient_ids = df["PatientID"].unique()

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patient_df = self.df[self.df["PatientID"] == pid].sort_values("StudyIndex")
        paths = patient_df["Path"].tolist()
        labels = patient_df[self.labels].values

        if len(paths) == 0:
            raise ValueError(f"Patient {pid} has no valid frontal images.")
        elif len(paths) == 1:
            selected = [paths[0]] * 3
            label = labels[0]
        elif len(paths) == 2:
            selected = [paths[0], paths[1], paths[1]]
            label = labels[1]
        else:
            mid_idx = len(paths) // 2
            selected = [paths[0], paths[mid_idx], paths[-1]]
            label = labels[-1] 

        images = []
        for p in selected:
            full_path = os.path.join(self.img_dir, p)
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Skipping corrupted image: {full_path} due to error: {e}")
                image = Image.new("RGB", (224, 224)) 
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)  # (T, C, H, W)
        return images, torch.tensor(label, dtype=torch.float32)


# ====== Transforms ======
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====== ViT Patch Embedding ======
class Hybrid3DPatchEmbedding(nn.Module):
    def __init__(self, time_steps=3, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.time_steps = time_steps
        self.patch_size = patch_size

        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.projection(x)       # (B, D, T, H', W')
        D, T_out, H_out, W_out = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T_out, H_out * W_out, D)  # (B, T, N, D)
        return x + self.pos_embedding.unsqueeze(1)  # (B, T, N, D)

# ====== Transformer ======
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

# ====== ResNet  ======
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        res_tokens = []
        for t in range(T):
            xt = x[:, t]  # (B, C, H, W)
            ft = self.backbone(xt)  # (B, 2048, H', W')
            ft = self.proj(ft)     # (B, D, H', W')
            res_tokens.append(ft.flatten(2).transpose(1, 2))  # (B, N, D)
        return torch.stack(res_tokens, dim=1)  # (B, T, N, D)

# ====== Vision Transformer Hybrid ======
class VisionTransformerHybrid(nn.Module):
    def __init__(self, time_steps=3, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, load_pretrained=False, vit_weight_path=None):
        super().__init__()
        self.time_steps = time_steps
        self.patch_embed = Hybrid3DPatchEmbedding(time_steps, img_size, patch_size, in_channels, embed_dim)
        self.transformers = nn.ModuleList([ 
            nn.Sequential(*[TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)]) 
            for _ in range(time_steps)
        ])
        self.resnet = ResNetFeatureExtractor(embed_dim)

        if load_pretrained and vit_weight_path is not None:
            state_dict = torch.load(vit_weight_path, map_location='cpu')
            self.patch_embed.load_state_dict(state_dict, strict=False)
            print("✅ Patch embedding loaded")

    def forward(self, x):  # x: (B, T, C, H, W)
        x_vit = self.patch_embed(x)  # (B, T, N, D)
        B, T, N, D = x_vit.shape
        vit_out = []
        for t in range(T):
            xt = x_vit[:, t]  # (B, N, D)
            vt = self.transformers[t](xt)  # (B, N, D)
            vit_out.append(vt)
        vit_out = torch.stack(vit_out, dim=1)  # (B, T, N, D)

        resnet_out = self.resnet(x)  # (B, T, 49, D)
        fused = torch.cat([vit_out, resnet_out], dim=2)  # (B, T, 245, D)
        return fused

# ====== Vision Transformer Hybrid Classifier ======
class VisionTransformerHybridClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes=14):
        super().__init__()
        self.backbone = backbone  # VisionTransformerHybrid
        self.pool = nn.AdaptiveAvgPool2d((1, backbone.patch_embed.projection.out_channels))  # (B, 1, D)
        self.classifier = nn.Sequential(
            nn.Linear(backbone.patch_embed.projection.out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        x = self.backbone(x)         # (B, T, 245, D)
        x = x.mean(dim=2)            # (B, T, D) 
        x = x.mean(dim=1)            # (B, D)  
        return self.classifier(x)    # (B, num_classes)

# ====== AUC evaluation ======
def compute_auc(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    aucs = []
    for i in range(y_true.shape[1]):
        if len(set(y_true[:, i])) < 2:
            continue 
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except:
            continue
    return float('nan') if not aucs else sum(aucs) / len(aucs)

# ====== Training Function ======
def train_model():
    vit_model = VisionTransformerHybrid(
    time_steps=3,               
    img_size=224,               
    patch_size=16,              
    in_channels=3,             
    embed_dim=768,              
    num_heads=12,               
    num_layers=12,              
    load_pretrained=False,      
    vit_weight_path=None        
).to(DEVICE)
    model = VisionTransformerHybridClassifier(backbone=vit_model).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = CheXpertDataset(CSV_PATH, IMG_DIR, transform)
    val_dataset = CheXpertDataset(VALID_CSV_PATH, IMG_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    best_auc = 0
    patience = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits))
            all_labels.append(labels)

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_labels, dim=0)
        train_auc = compute_auc(trues, preds)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits))
                val_labels.append(labels)

        val_preds = torch.cat(val_preds, dim=0)
        val_trues = torch.cat(val_labels, dim=0)
        val_auc = compute_auc(val_trues, val_preds)

        print(f"Loss={total_loss/len(train_loader):.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0
            best_path = os.path.join(save_dir, "hybrid_encoder_chexpert_best.pt")
            torch.save(model.state_dict(), best_path)
            print("✅ Saved best model (AUC improved)")

        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    final_path = os.path.join(save_dir, "hybrid_encoder_chexpert_final.pt")
    torch.save(model.state_dict(), final_path)
    print("✅ Final model saved as hybrid_encoder_chexpert_final.pt")

    with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
        f.write(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}\n")
        f.write(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}\n")

    torch.save(preds, os.path.join(save_dir, "train_embeddings.pt"))
    torch.save(val_preds, os.path.join(save_dir, "val_embeddings.pt"))

# Run
train_model()
