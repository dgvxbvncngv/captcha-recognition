# src/test.py
# 验证码识别模型测试脚本
# 用法:
#   1. 测试整个验证集准确率:  python -m src.test
#   2. 测试单张图片:          python -m src.test --image path/to/captcha.png
#   3. 测试并展示 N 个样本:    python -m src.test --show 10
import os
import sys
import glob
import argparse
import time
import torch
from torch.utils.data import DataLoader
from src.config import VAL_DATA_DIR, BATCH_SIZE, NUM_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT
from src.dataset import CaptchaDataset
from src.model import CRNN
from src.predict import load_model, preprocess_image, predict
from src.utils import ctc_greedy_decode


def evaluate(model, device, val_dir=VAL_DATA_DIR):
    """在整个验证集上评估准确率"""
    dataset = CaptchaDataset(val_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    start = time.time()
    wrong_samples = []

    with torch.no_grad():
        for batch_idx, (images, _, label_strs) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            preds = ctc_greedy_decode(logits.cpu())

            for pred, true in zip(preds, label_strs):
                if pred == true:
                    correct += 1
                else:
                    if len(wrong_samples) < 10:
                        wrong_samples.append((true, pred))
                total += 1
                # 逐字符统计
                for c_t, c_p in zip(true, pred):
                    char_total += 1
                    if c_t == c_p:
                        char_correct += 1

    acc = correct / total if total > 0 else 0
    char_acc = char_correct / char_total if char_total > 0 else 0
    elapsed = time.time() - start

    print('=' * 50)
    print(f'验证集评估结果')
    print('=' * 50)
    print(f'样本总数: {total}')
    print(f'整体准确率 (整串匹配): {acc*100:.2f}%  ({correct}/{total})')
    print(f'字符准确率 (逐字符):   {char_acc*100:.2f}%  ({char_correct}/{char_total})')
    print(f'耗时: {elapsed:.1f}s, 平均 {elapsed/total*1000:.1f}ms/张')
    print('=' * 50)

    if wrong_samples:
        print('\n错误样本示例 (真实 -> 预测):')
        for true, pred in wrong_samples:
            print(f'  {true}  ->  {pred}')

    return acc, char_acc


def predict_single(image_path, model, device):
    """测试单张图片"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    true_label = os.path.basename(image_path).split('_')[0]
    start = time.time()
    pred_text, confidence = predict(image_bytes, model, device)
    elapsed = (time.time() - start) * 1000

    print('=' * 50)
    print(f'单张图片预测')
    print('=' * 50)
    print(f'图片路径: {image_path}')
    print(f'真实标签: {true_label}')
    print(f'预测结果: {pred_text}')
    print(f'置信度:   {confidence:.4f}')
    print(f'是否正确: {"✓ 正确" if pred_text == true_label else "✗ 错误"}')
    print(f'耗时:     {elapsed:.1f}ms')
    print('=' * 50)
    return pred_text, confidence


def predict_batch(image_paths, model, device):
    """批量预测图片"""
    print('=' * 60)
    print(f'批量预测 {len(image_paths)} 张图片')
    print('=' * 60)
    correct = 0
    for path in image_paths:
        with open(path, 'rb') as f:
            image_bytes = f.read()
        true_label = os.path.basename(path).split('_')[0]
        pred_text, confidence = predict(image_bytes, model, device)
        ok = pred_text == true_label
        if ok:
            correct += 1
        flag = '✓' if ok else '✗'
        print(f'  {flag} 真实: {true_label}  预测: {pred_text}  置信度: {confidence:.3f}')
    print('-' * 60)
    print(f'准确率: {correct}/{len(image_paths)} = {correct/len(image_paths)*100:.1f}%')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='验证码识别模型测试')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径 (测试单张时使用)')
    parser.add_argument('--dir', type=str, default=VAL_DATA_DIR,
                        help=f'验证集目录 (默认: {VAL_DATA_DIR})')
    parser.add_argument('--show', type=int, default=0,
                        help='展示前 N 个样本的预测结果')
    parser.add_argument('--model', type=str, default=None,
                        help='模型文件路径 (默认使用 config 中的路径)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 加载模型
    model = load_model(device)
    print(f'模型加载成功\n')

    if args.image:
        # 单张图片测试
        if not os.path.exists(args.image):
            print(f'错误: 图片不存在: {args.image}')
            sys.exit(1)
        predict_single(args.image, model, device)
    elif args.show > 0:
        # 展示 N 个样本
        files = sorted(glob.glob(os.path.join(args.dir, '*.png')))[:args.show]
        if not files:
            print(f'错误: 目录中没有 png 图片: {args.dir}')
            sys.exit(1)
        predict_batch(files, model, device)
    else:
        # 整个验证集评估
        if not os.path.isdir(args.dir):
            print(f'错误: 验证集目录不存在: {args.dir}')
            sys.exit(1)
        evaluate(model, device, args.dir)


if __name__ == '__main__':
    main()
