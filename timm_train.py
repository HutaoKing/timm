import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
import ttach as tta

# -----------------------------
# 配置
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data', help='根数据目录，下面应有 train/ 和 val/ 子目录')
parser.add_argument('--model-name', type=str, default='seresnext50_32x4d.racm_in1k',
                    help='timm 模型名')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save-path', type=str, default='checkpoints', 
                    help='模型权重保存目录')
args = parser.parse_args()

# -----------------------------
# 随机种子
# -----------------------------
torch.manual_seed(args.seed)
random.seed(args.seed)

# -----------------------------
# 数据增强与加载
# -----------------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),   transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

num_classes = len(train_ds.classes)

# -----------------------------
# 模型构建
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
model = model.to(device)

# -----------------------------
# 损失、优化器、调度器
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# -----------------------------
# 训练与验证函数
# -----------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"[Train {epoch}]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(train_loader.dataset)

def validate(epoch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[Val {epoch}]"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# -----------------------------
# 主训练循环
# -----------------------------
os.makedirs(args.save_path, exist_ok=True)
best_acc = 0.0

for epoch in range(1, args.epochs+1):
    loss = train_one_epoch(epoch)
    acc  = validate(epoch)
    scheduler.step()

    print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  val_acc={acc:.4%}")

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save({'state_dict': model.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch},
                   os.path.join(args.save_path, f"model_seed{args.seed}.pth"))
        print(f"  --> 新最佳: {best_acc:.4%}, 已保存到 model_seed{args.seed}.pth")
