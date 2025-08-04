import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset_preparation import SemanticSegmentationDataset

# --------------- Preparing dataset for training ---------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Resize to desired input size
    transforms.ToTensor()
])

train_dataset = SemanticSegmentationDataset(
    image_dir='E:\\datatestJ03\\train\\input',  # Path to dataset
    label_dir='E:\\datatestJ03\\train\\label',  # Path to dataset
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir='E:\\datatestJ03\\test\\input',  # Path to dataset
    label_dir='E:\\datatestJ03\\test\\label',  # Path to dataset
    transform=train_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# ------------------------------------

import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from model_preparation import Discriminator

classes = 3
# --------------- Preparing models for training ---------------------
# === Generator (G) - UNET++ ===
G = smp.UnetPlusPlus(
    encoder_name="resnet18",
    encoder_weights=None,
    classes=classes,
    activation='softmax'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = G.to(device)
D = Discriminator().to(device)

# --------------- Training models ---------------------
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassRecall, MulticlassAccuracy, MulticlassPrecision
from torchmetrics import ConfusionMatrix
import copy
from trainingFunctions import train_gan, evaluate

criterion_G1 = nn.CrossEntropyLoss()
criterion_G2 = nn.BCELoss()
criterion_D = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
optimizer_D = optim.Adam(D.parameters(), lr=1e-6)
num_epochs = 100
epoch_save = 0
best_val_accuracy = 0.0
best_model_state = None

for epoch in range(num_epochs):
    _, lossG_train, lossD_train, mAcc_train, mIoU_train, mF1_train, mPre_train = train_gan(G, D, train_dataloader, criterion_G1, criterion_G2, criterion_D, optimizer_G, optimizer_D, device, classes, lambda_adv=0.1)
    _, lossG_val, mAcc_val, mIoU_val, mF1_val, mPre_val = evaluate(G, D, val_dataloader, criterion_G1, device, classes)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss G: {lossG_train:.4f}, Loss D: {lossD_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, mIoU: {mIoU_train:.4f}, Mean F1 Score: {mF1_train:.4f}, Mean Precision: {mPre_train:.4f}")
    print(f"Validation Loss G: {lossG_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, mIoU: {mIoU_val:.4f}, Mean F1 Score: {mF1_val:.4f}, Mean Precision: {mPre_val:.4f}")
    f = open('training.txt', 'a')
    f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
    f.write(f"Train Loss G: {lossG_train:.4f}, Loss D: {lossD_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, mIoU: {mIoU_train:.4f}, Mean F1 Score: {mF1_train:.4f}, Mean Precision: {mPre_train:.4f}\n")
    f.write(f"Validation Loss G: {lossG_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, mIoU: {mIoU_val:.4f}, Mean F1 Score: {mF1_val:.4f}, Mean Precision: {mPre_val:.4f}\n")
    f.close()

    if mAcc_val >= best_val_accuracy:
        epoch_saved = epoch + 1
        best_val_accuracy = mAcc_val
        best_model_state = copy.deepcopy(G.state_dict())

print("===================")
print(f"Best Model at epoch : {epoch_saved}")
f = open('training.txt', 'a')
f.write("===================\n")
f.write(f"Best Model at epoch : {epoch_saved}\n")
f.close()
torch.save(best_model_state, "UNetpp.pth")
  