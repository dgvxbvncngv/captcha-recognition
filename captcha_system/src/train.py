# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
from src.config import (
    TRAIN_DATA_DIR, VAL_DATA_DIR, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, NUM_CLASSES, MODEL_SAVE_PATH
)
from src.dataset import CaptchaDataset
from src.model import CRNN
from src.utils import ctc_greedy_decode

LOG_FILE = 'train_progress.log'

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def train():
    # 清空日志
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('')

    # 加载数据集
    log('加载训练集...')
    train_dataset = CaptchaDataset(TRAIN_DATA_DIR)
    val_dataset = CaptchaDataset(VAL_DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log(f'训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'使用设备: {device}')
    model = CRNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        num_batches = len(train_loader)
        for batch_idx, (images, labels, label_strs) in enumerate(train_loader):
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

            # 每 50 批打印一次进度
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                log(f'Epoch {epoch+1}/{EPOCHS} - 批次 {batch_idx+1}/{num_batches}, '
                    f'loss={loss.item():.4f}, 速度={speed:.1f}批/秒, '
                    f'已用={elapsed:.0f}s')

        train_time = time.time() - epoch_start
        log(f'Epoch {epoch+1} 训练完成, 耗时 {train_time:.0f}s, 平均loss={total_loss/num_batches:.4f}')

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_start = time.time()
        with torch.no_grad():
            for batch_idx, (images, _, label_strs) in enumerate(val_loader):
                images = images.to(device)
                logits = model(images)
                preds = ctc_greedy_decode(logits.cpu())

                if batch_idx == 0:
                    log('----- 验证集样本对比 -----')
                    for i in range(min(5, len(label_strs))):
                        log(f'  真实: {label_strs[i]}  ->  预测: {preds[i]}')
                    argmax_seq = logits.argmax(dim=-1)[0][:20].tolist()
                    log(f'  argmax 前20索引: {argmax_seq}')
                    log('---------------------------')

                for pred, true in zip(preds, label_strs):
                    if pred == true:
                        correct += 1
                    total += 1

        acc = correct / total if total > 0 else 0
        val_time = time.time() - val_start
        log(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/num_batches:.4f}, '
            f'Val Acc: {acc:.4f}, 验证耗时: {val_time:.0f}s')

    # 保存模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log(f'模型已保存到 {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train()