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
    "epochs":           20,
    "lr":               1e-3,
    "weight_decay":     0.05,
    "dropout_prob":     0.2,
    "seed":             42,
    "dataset_path":     "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "model_name":       "seresnext50_32x4d",  # timm 内置名
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":      8,
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
# 3. 数据增强 & DataLoader
# -------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.1),
])
val_transform = transforms.Compose([
    transforms.Resize(int(config.image_size*1.15)),
    transforms.CenterCrop(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

base_ds = datasets.ImageFolder(config.dataset_path, transform=None)
N = len(base_ds)
idxs = list(range(N))
random.shuffle(idxs)
split = int(0.7 * N)
train_idx, val_idx = idxs[:split], idxs[split:]

class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.ds = base_ds
        self.idx = indices
        self.tf = transform
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        path, lbl = self.ds.samples[self.idx[i]]
        img = self.ds.loader(path).convert("RGB")
        return self.tf(img), lbl

train_ds = SubsetWithTransform(base_ds, train_idx, train_transform)
val_ds   = SubsetWithTransform(base_ds, val_idx,   val_transform)

train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                          shuffle=True,  num_workers=config.num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=config.eval_batch_size,
                          shuffle=False, num_workers=config.num_workers, pin_memory=True)

# -------------------------------------------------------------------------
# 4. 模型 & 损失 & 优化器 & 调度
# -------------------------------------------------------------------------
device = torch.device(config.device)
# 4.1 创建带 dropout 的 head
class HeadWithDropout(nn.Sequential):
    def __init__(self, in_features, num_classes, p=0.2):
        super().__init__(
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

# 4.2 从 timm 加载模型（pretrained=True）
model = timm.create_model(
    config.model_name,
    pretrained=True,
    num_classes=config.num_classes,
)
# 替换 classifier
in_feats = model.get_classifier().in_features
model.reset_classifier(0)  # 清掉原来的 head
model.classifier = HeadWithDropout(in_feats, config.num_classes, p=config.dropout_prob)
model = model.to(device)

# 4.3 损失函数（带 label smoothing）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 4.4 优化器 & 调度
optimizer = AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)
# Cosine 重启调度，t_initial = total epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-6
)

# -------------------------------------------------------------------------
# 5. 训练 & 验证函数
# -------------------------------------------------------------------------
def train_one_epoch(e):
    model.train()
    total_loss, count = 0., 0
    loop = tqdm(train_loader, desc=f"[Train] Epoch {e}", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)
        loop.set_postfix(loss=total_loss/count)
    scheduler.step()
    return total_loss / count

def validate(e):
    model.eval()
    correct, count = 0, 0
    loop = tqdm(val_loader, desc=f"[ Val ] Epoch {e}", leave=False)
    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            count += imgs.size(0)
            loop.set_postfix(acc=correct/count)
    return correct / count

# -------------------------------------------------------------------------
# 6. 主训练循环
# -------------------------------------------------------------------------
os.makedirs("./results/ckpt_timm", exist_ok=True)
best_acc = 0.0
for epoch in range(1, config.epochs+1):
    train_loss = train_one_epoch(epoch)
    val_acc    = validate(epoch)
    print(f"Epoch {epoch}/{config.epochs}  loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")
    # 保存最好模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"./results/ckpt_timm/best.pth")
print(f"Training finished, best val_acc={best_acc*100:.2f}%")
