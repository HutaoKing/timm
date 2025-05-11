import torch
import os
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# -------------------- 配置 --------------------
data_dir = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100'
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')
pretrained_model_path = './pretrain_model/seresnext50_32x4d.pth'

batch_size    = 32
epochs        = 20
learning_rate = 1e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on device: {device.upper()}")

# ---------------- 数据准备 ----------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset   = ImageFolder(val_dir,   transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=(device=='cuda'))
val_loader   = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=(device=='cuda'))

# ---------------- 模型构建 ----------------
# 1) 先加载原始的 1000 类模型
model = timm.create_model('seresnext50_32x4d', pretrained=False)
#print("Original model.fc:", model.fc)  # debug: 查看原始 fc

# 2) 读取 checkpoint
checkpoint = torch.load(pretrained_model_path, map_location=device)
# 如果 checkpoint 是个 dict 且有 'state_dict' 键，就取它
state_dict = checkpoint.get('state_dict', checkpoint)
#print("Checkpoint keys sample:", list(state_dict.keys())[:5])  # debug: 看几条 key

# 3) 删除最后一层 fc 的权重，以避免大小不匹配
for k in ['fc.weight', 'fc.bias']:
    if k in state_dict:
        #print(f"Removing key from checkpoint: {k}")  # debug
        del state_dict[k]

# 4) 加载剩余权重
model.load_state_dict(state_dict, strict=False)
#print("Loaded pretrained weights except fc.")

# 5) 替换分类头为 26 类
model.fc = nn.Linear(model.num_features, 26)
#print("New model.fc:", model.fc)  # debug: 查看新 fc

model = model.to(device)

# ---------------- 损失 & 优化器 ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------- 训练函数 ----------------
def train_model():
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader,
                    desc=f'Epoch[{epoch}/{epochs}]',
                    ncols=100)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc':  f'{100*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc  = 100. * correct / total
        print(f'>>> Epoch {epoch:02d}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%')

        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

# ---------------- 验证函数 ----------------
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader,
                                   desc='Validation',
                                   ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100. * correct / total
    print(f'>>> Validation Accuracy: {val_acc:.2f}%')

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    train_model()
    evaluate_model()
