import os
import random
from easydict import EasyDict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from multiprocessing import freeze_support

# -------------------------------------------------------------------------
# 1. 配置
# -------------------------------------------------------------------------
config = EasyDict({
    "num_classes":      26,
    "image_size":       224,
    "batch_size":       24,
    "eval_batch_size":  32,
    "head_epochs":      5,      # 先训练 head 的 epoch
    "finetune_epochs":  15,     # 微调全网的 epoch
    "lr_head":          1e-3,   # 训练 head 时的学习率
    "lr_ft":            1e-4,   # 微调时的学习率
    "weight_decay":     0.05,
    "dropout_prob":     0.2,
    "seed":             42,
    "dataset_path":     "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "pretrained_ckpt":  "pretrain_model/seresnext50_32x4d.pth",
    "save_dir":         "./results/ckpt_timm",
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":      4,      # Windows 下可适当调小或设为 0
})

# -------------------------------------------------------------------------
# 2. 随机种子
# -------------------------------------------------------------------------
torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
if config.device == "cuda":
    torch.cuda.manual_seed_all(config.seed)

# -------------------------------------------------------------------------
# 3. 自定义 Dataset 与 DataLoader
# -------------------------------------------------------------------------
class SubsetDS(Dataset):
    def __init__(self, samples, labels, transform):
        self.samples = samples
        self.labels = labels
        self.tf = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        lbl = self.labels[idx]
        img = transforms.functional.pil_to_tensor(__import__("PIL").Image.open(path).convert("RGB"))
        img = transforms.functional.convert_image_dtype(img, dtype=torch.float)
        img = transforms.functional.resize(img, [config.image_size, config.image_size])
        img = self.tf(transforms.functional.to_pil_image(img))
        return img, lbl

def make_dataloaders():
    # 数据增强
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(config.image_size * 1.15)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 扫描所有文件和标签
    base = datasets.ImageFolder(config.dataset_path, transform=None)
    samples = [s[0] for s in base.samples]
    labels  = [s[1] for s in base.samples]

    # 70/30 随机划分
    N = len(samples)
    idxs = list(range(N))
    random.shuffle(idxs)
    split = int(0.7 * N)
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_samples = [samples[i] for i in train_idxs]
    train_labels  = [labels[i]  for i in train_idxs]
    val_samples   = [samples[i] for i in val_idxs]
    val_labels    = [labels[i]  for i in val_idxs]

    train_ds = SubsetDS(train_samples, train_labels, train_tf)
    val_ds   = SubsetDS(val_samples,   val_labels,   val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

# -------------------------------------------------------------------------
# 4. 模型构建与权重加载
# -------------------------------------------------------------------------
def build_model():
    device = torch.device(config.device)

    # 创建 backbone + 自定义 head
    model = timm.create_model("seresnext50_32x4d", pretrained=False, num_classes=0)
    in_feats = model.num_features
    model.reset_classifier(0)
    model.classifier = nn.Sequential(
        nn.Dropout(config.dropout_prob),
        nn.Linear(in_feats, config.num_classes)
    )
    model = model.to(device)

    # 加载本地 backbone 权重（忽略 fc 层）
    if os.path.exists(config.pretrained_ckpt):
        sd = torch.load(config.pretrained_ckpt, map_location=device)
        sd.pop("fc.weight", None)
        sd.pop("fc.bias",   None)
        model.load_state_dict(sd, strict=False)
        print("✅ Loaded backbone weights (fc ignored).")
    else:
        print("⚠️  Pretrained checkpoint not found, training from scratch.")

    return model

# -------------------------------------------------------------------------
# 5. 训练与验证
# -------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, count = 0.0, 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total_loss / count

def validate(model, loader, criterion):
    model.eval()
    correct, count = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            val_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)
    return val_loss / count, correct / count

# -------------------------------------------------------------------------
# 6. 主函数 (必须在 Windows 下这样写)
# -------------------------------------------------------------------------
def main():
    freeze_support()  # Windows 多进程必需
    os.makedirs(config.save_dir, exist_ok=True)

    train_loader, val_loader = make_dataloaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 阶段一：只训练 head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.classifier.parameters(), lr=config.lr_head, weight_decay=config.weight_decay)
    best_acc = 0.0
    for epoch in range(1, config.head_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"[Head ] Epoch {epoch}/{config.head_epochs}  loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_head.pth"))

    # 阶段二：微调整个模型
    for p in model.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=config.lr_ft, weight_decay=config.weight_decay)
    for epoch in range(1, config.finetune_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"[Fine ] Epoch {epoch}/{config.finetune_epochs}  loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_finetune.pth"))

    print(f"🎉 Training complete, best val_acc = {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
