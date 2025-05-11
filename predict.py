import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
os.system(f'pip install timm') 
os.system(f'pip install ttach')
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

# 全局加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_paths = ['results/ckpt_timm/seresnext50_32x4d_racm_in1k-epoch1.pth']
models = []

for p in model_paths:
    ckpt = torch.load(p, map_location=device)
    m = timm.create_model('seresnext50_32x4d.racm_in1k',
                         pretrained=False, num_classes=len(inverted))
    m.load_state_dict(ckpt['state_dict'])
    m.to(device).eval()
    # 包装 TTA（仅做水平翻转示例）
    tta_m = tta.ClassificationTTAWrapper(m, tta.aliases.flip_transform(), merge_mode='mean')
    models.append(tta_m)

# 统一的图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def predict(img_input):
    # 支持文件路径或 numpy 数组
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input)
    else:
        img = Image.open(img_input)
    img = img.convert('RGB')

    # BGR → RGB
    img = Image.fromarray(np.array(img)[:, :, ::-1])

    # 标准预处理
    inp = transform(img).unsqueeze(0).to(device)

    # 双模型 TTA 投票
    with torch.no_grad():
        total_logits = None
        for m in models:
            logits = m(inp)
            total_logits = logits if total_logits is None else total_logits + logits

    pred = total_logits.argmax(dim=1).item()
    return inverted[pred]

if __name__ == '__main__':
    test_image = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_05/00011.jpg'
    print("预测结果：", predict(test_image))
