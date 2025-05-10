import os
import random
from easydict import EasyDict

import numpy as np
import torch
from torch import nn, optim
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
    "image_height":     224,
    "image_width":      224,
    "batch_size":       24,
    "eval_batch_size":  10,
    "epochs":           10,
    "lr_max":           0.01,
    "decay_type":       'constant',  # 'constant' or 'cosine'
    "momentum":         0.8,
    "weight_decay":     3.0,
    "dataset_path":     "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "save_ckpt_epochs": 1,
    "save_ckpt_path":   "./results/ckpt_timm",
    "model_name":       "seresnext50_32x4d.racm_in1k",
    "seed":             42,
})

# -------------------------------------------------------------------------
# 2. 定义 SubsetWithTransform 在模块顶层以支持多进程 DataLoader
# -------------------------------------------------------------------------
class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base_ds.samples[self.indices[idx]]
        img = self.base_ds.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------------------------------------------------
# 3. 主函数
# -------------------------------------------------------------------------
def main():
    # 3.1 随机种子
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 3.2 数据增强与加载
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_height),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(config.image_height * 1.15)),
        transforms.CenterCrop(config.image_height),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    base_ds = datasets.ImageFolder(config.dataset_path, transform=None)
    dataset_size = len(base_ds)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_ds = SubsetWithTransform(base_ds, train_idx, train_transform)
    val_ds = SubsetWithTransform(base_ds, val_idx, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3.3 模型 & 预训练权重加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(config.model_name, pretrained=False, num_classes=config.num_classes)
    state_dict = torch.load('pretrain_model/seresnext50_32x4d.pth', map_location=device)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print("Successfully loaded pretrained weights!")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr_max,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    scheduler = None
    if config.decay_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 3.4 训练与验证函数
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch") as t:
            for imgs, labels in t:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                t.set_postfix(loss=running_loss / ((t.n + 1) * config.batch_size))
        return running_loss / len(train_loader.dataset)

    def validate(epoch):
        model.eval()
        correct = 0
        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch") as t:
            for imgs, labels in t:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                t.set_postfix(acc=correct / ((t.n + 1) * config.eval_batch_size))
        return correct / len(val_loader.dataset)

    # 3.5 主训练循环
    os.makedirs(config.save_ckpt_path, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(epoch)
        acc = validate(epoch)
        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch}/{config.epochs}    loss={loss:.4f}    val_acc={acc:.4%}")
        if epoch % config.save_ckpt_epochs == 0:
            ckpt_name = f"{config.model_name.replace('.', '_')}-epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': acc
            }, os.path.join(config.save_ckpt_path, ckpt_name))
            print(f"  --> Saved checkpoint: {ckpt_name}")
        best_acc = max(best_acc, acc)
    print(f"Training done, best val_acc={best_acc:.4%}")

if __name__ == "__main__":
    freeze_support()
    main()
