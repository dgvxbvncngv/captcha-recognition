import torch
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
