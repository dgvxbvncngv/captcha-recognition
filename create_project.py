# create_project.py
import os

# 项目根目录
PROJECT_NAME = "captcha_system"
BASE_DIR = os.path.join(os.getcwd(), PROJECT_NAME)

# 要创建的文件和内容映射（相对路径 -> 文件内容）
FILES = {
    # 根目录文件
    "requirements.txt": """torch>=1.9.0
torchvision>=0.10.0
opencv-python
pillow
captcha
flask
numpy
""",
    "data_gen.py": """from captcha.image import ImageCaptcha
import random
import os
from src.config import CHARS, IMAGE_WIDTH, IMAGE_HEIGHT

def generate_dataset(out_dir, num_samples=5000, length=4):
    os.makedirs(out_dir, exist_ok=True)
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    for i in range(num_samples):
        text = ''.join(random.choices(CHARS, k=length))
        image.write(text, os.path.join(out_dir, f'{text}_{i}.png'))
    print(f'生成 {num_samples} 张图片到 {out_dir}')

if __name__ == '__main__':
    generate_dataset('data/train', num_samples=8000)
    generate_dataset('data/val', num_samples=2000)
""",
    "app.py": """from flask import Flask, request, jsonify
import torch
from src.predict import load_model, predict

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device)

@app.route('/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片文件'}), 400
    file = request.files['image']
    image_bytes = file.read()
    try:
        text, conf = predict(image_bytes, model, device)
        return jsonify({'result': text, 'confidence': round(conf, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
""",
    # src 包内文件
    "src/__init__.py": "",
    "src/config.py": """import string

CHARS = string.digits + string.ascii_uppercase
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i+1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_DATA_DIR = 'data/train'
VAL_DATA_DIR = 'data/val'
MODEL_SAVE_PATH = 'models/crnn_captcha.pth'
""",
    "src/dataset.py": """import os
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
        return img, label, label_str
""",
    "src/model.py": """import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channel=1, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, (2, 2)),
        )
        self.feature_height = 4
        self.feature_width = 20
        self.cnn_output_size = 256 * self.feature_height
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
""",
    "src/utils.py": """import torch
from src.config import IDX2CHAR

def ctc_greedy_decode(logits):
    preds = logits.argmax(dim=-1)
    batch_size = preds.size(0)
    results = []
    for i in range(batch_size):
        prev = 0
        decoded = []
        for p in preds[i].tolist():
            if p != prev and p != 0:
                decoded.append(IDX2CHAR.get(p, ''))
            prev = p
        results.append(''.join(decoded))
    return results
""",
    "src/train.py": """import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from src.config import (
    TRAIN_DATA_DIR, VAL_DATA_DIR, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, NUM_CLASSES, MODEL_SAVE_PATH
)
from src.dataset import CaptchaDataset
from src.model import CRNN
from src.utils import ctc_greedy_decode

def train():
    train_dataset = CaptchaDataset(TRAIN_DATA_DIR)
    val_dataset = CaptchaDataset(VAL_DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels, label_strs in train_loader:
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            targets = torch.cat(labels)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, _, label_strs in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = ctc_greedy_decode(logits.cpu())
                for pred, true in zip(preds, label_strs):
                    if pred == true:
                        correct += 1
                    total += 1
        acc = correct / total if total > 0 else 0
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'模型已保存到 {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train()
""",
    "src/predict.py": """import torch
import cv2
import numpy as np
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, MODEL_SAVE_PATH
from src.model import CRNN
from src.utils import ctc_greedy_decode

def load_model(device='cpu'):
    model = CRNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img_tensor

def predict(image_bytes, model, device='cpu'):
    img_tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_text = ctc_greedy_decode(logits.cpu())[0]
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
    return pred_text, confidence
""",
}

def create_project():
    # 创建根目录
    os.makedirs(BASE_DIR, exist_ok=True)
    os.chdir(BASE_DIR)

    # 创建所有文件
    for filepath, content in FILES.items():
        # 确保子目录存在
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'创建文件: {filepath}')

    print(f'\n✅ 项目创建成功！目录: {BASE_DIR}')
    print('\n接下来请执行以下命令：')
    print(f'  cd {PROJECT_NAME}')
    print('  pip install -r requirements.txt')
    print('  python data_gen.py')
    print('  python -m src.train')
    print('  python app.py')

if __name__ == '__main__':
    create_project()