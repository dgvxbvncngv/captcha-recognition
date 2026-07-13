from captcha.image import ImageCaptcha
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
