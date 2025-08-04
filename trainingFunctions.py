from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassRecall, MulticlassAccuracy, MulticlassPrecision
from torchmetrics import ConfusionMatrix
import copy
import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np


def train_gan(G, D, dataloader, criterion_G1, criterion_G2, criterion_D, optimizer_G, optimizer_D, device, num_classes, lambda_adv):
    G.train()
    D.train()
    running_loss_g = 0.0
    running_loss_d = 0.0
    
    # Metrics
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    
    pbar = tqdm(dataloader, desc='Training cGAN ', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        # --------------------  Calculate G loss at Step 1  ----------------
        optimizer_G.zero_grad()
        seg = G(images)    
        loss_G1 = criterion_G1(seg, labels)
        # --------------------- Train Discriminator ---------------------
        optimizer_D.zero_grad()
        seg_single_channel = seg.argmax(dim=1, keepdim=True).float()
        confidence_map_fake = D(labels.unsqueeze(1).float(), seg_single_channel)
        loss_D_fake = criterion_D(confidence_map_fake, torch.zeros_like(confidence_map_fake))
        confidence_map_real = D(labels.unsqueeze(1).float(), labels.unsqueeze(1).float())
        loss_D_real = criterion_D(confidence_map_real, torch.ones_like(confidence_map_real))
        loss_D = (loss_D_fake + loss_D_real)/2
        loss_D.backward()
        optimizer_D.step()
        # --------------------- Calculate G loss at Step 2 ---------------------
        optimizer_G.zero_grad()
        confidence_map = D(labels.unsqueeze(1).float(), seg_single_channel)
        loss_G2 = criterion_G2(confidence_map, torch.ones_like(confidence_map))
        # --------------------- Training G ---------------------
        loss_G = loss_G1 + lambda_adv*loss_G2
        loss_G.backward()
        optimizer_G.step()

        # Metrics update
        confmat(seg, labels)
        iou_metric(seg, labels)
        f1_metric(seg, labels)
        accuracy_metric(seg, labels)
        precision_metric(seg, labels)
        
        pbar.set_postfix({
            'Loss G1': f'{loss_G1.item():.4f}',
            'Loss G2': f'{loss_G2.item():.4f}',
            'Loss G': f'{loss_G.item():.4f}',
            'Loss D': f'{loss_D.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}'
        })

        running_loss_d += loss_D.item() * images.size(0)
        running_loss_g += loss_G.item() * images.size(0)
    
    epoch_loss_g = running_loss_g / len(dataloader.dataset)
    epoch_loss_d = running_loss_d / len(dataloader.dataset)
    cm_normalized = confmat.compute().cpu().numpy().astype('float') / confmat.compute().cpu().numpy().sum(axis=1)[:, np.newaxis]
    confmat.reset()
    
    return cm_normalized, epoch_loss_g, epoch_loss_d, accuracy_metric.compute().cpu().numpy(), iou_metric.compute().cpu().numpy(), f1_metric.compute().cpu().numpy(), precision_metric.compute().cpu().numpy()


def evaluate(G, D, dataloader, criterionG, device, num_classes):
    G.eval()
    running_loss_G = 0.0
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)


    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = G(images)
            lossG = criterionG(outputs, labels)

            running_loss_G += lossG.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            confmat(preds, labels)
            iou_metric(preds, labels)
            f1_metric(preds, labels)
            accuracy_metric(preds, labels)
            precision_metric(preds, labels)

            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss G': f'{lossG.item():.4f}',
                'Accuracy': f'{accuracy_metric.compute():.4f}',
                'mIoU': f'{iou_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}'
            })
    
    epoch_lossG = running_loss_G / len(dataloader.dataset)
    cm = confmat.compute().cpu().numpy()  # Convert to numpy for easy usage
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    confmat.reset()

    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()

    return cm_normalized, epoch_lossG, mean_accuracy, mean_iou, mean_f1, mean_precision     
