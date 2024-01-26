import torch.nn as nn
import torch
import pandas as pd

def inference(model, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    output = model(imgs)
    return output
