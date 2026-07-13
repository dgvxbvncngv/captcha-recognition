import torch
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
