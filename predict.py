import os
# os.system(f'pip install timm') 
# os.system(f'pip install ttach')
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import ttach as tta

# 类别映射
inverted = {
    0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle',
    6: 'Cardboard', 7: 'Basketball', 8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks',
    11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush', 15: 'Dirty Cloth',
    16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery',
    20: 'Fluorescent lamp', 21: 'Tablet capsules', 22: 'Orange Peel',
    23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'
}

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 你想加载的模型权重列表
model_paths = ['results/ckpt_timm/epoch15.pth']  # 举例
models = []

for p in model_paths:
    print(f"\nLoading checkpoint from {p}")
    try:
        ckpt = torch.load(p, map_location=device)
    except Exception as e:
        print(f"Failed to torch.load('{p}'): {e}")
        raise
    # 调试输出一下 ckpt 类型和 keys
    print(f"  checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"  checkpoint keys: {list(ckpt.keys())[:5]}{'...' if len(ckpt)>5 else ''}")
    # 兼容多种格式
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt  # 直接就是 state_dict

    # 创建模型
    m = timm.create_model('seresnext50_32x4d', pretrained=False, num_classes=len(inverted))
    # 加载权重
    m.load_state_dict(state_dict, strict=True)
    m.to(device).eval()
    print("  loaded OK")

    # 包装 TTA
    tta_m = tta.ClassificationTTAWrapper(
        m,
        transforms=tta.aliases.flip_transform(),
        merge_mode='mean'
    )
    models.append(tta_m)

# 预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def predict(img_input):
    # 支持文件路径或 numpy
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input)
    else:
        img = Image.open(img_input)
    img = img.convert('RGB')
    # BGR->RGB
    img = Image.fromarray(np.array(img)[:, :, ::-1])

    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        total_logits = None
        for m in models:
            logits = m(inp)
            total_logits = logits if total_logits is None else total_logits + logits

    idx = total_logits.argmax(dim=1).item()
    return inverted[idx]

if __name__ == '__main__':
    test_image = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_01/00037.jpg'
    print("预测结果：", predict(test_image))
