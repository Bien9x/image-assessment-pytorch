import torch
from models import Nima
import numpy as np
import os
import argparse
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(img,base_model_name, model_path, dropout=0.75, n_classes=10):
    model = Nima(base_model_name, dropout, n_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    prediction = model(img).detach().numpy()
    score = (prediction * np.arange(1,n_classes + 1)).sum()
    return score
from PIL import Image
if __name__ == '__main__':
    score = predict_image(Image.open("data/test_images/test.JPG"),"resnet18","cp/checkpoint_ava_resnet18.pt")
    print(score)

