# src/dataset.py
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import CHARS, CHAR2IDX, IMAGE_HEIGHT, IMAGE_WIDTH

class CaptchaDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, '*.png'))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label_str = os.path.basename(path).split('_')[0]
        img = Image.open(path)
        img = self.transform(img)
        label = [CHAR2IDX[c] for c in label_str]
        label = torch.tensor(label, dtype=torch.long)

        # ---------- 调试代码（放在这里！） ----------
        if idx == 0:   # 注意缩进：属于 __getitem__ 方法体
            img_pil = Image.open(path)
            img_pil.save('debug_train_sample.png')
            print(f"保存调试图片: debug_train_sample.png, 标签: {label_str}")
        # ------------------------------------------

        return img, label, label_str