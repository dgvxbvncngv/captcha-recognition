import string

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
EPOCHS = 20
LEARNING_RATE = 0.0001   # 更小，更稳定
BATCH_SIZE = 32