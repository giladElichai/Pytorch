import os 
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#import albumentations as A
#from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as tt


from dataset import *
from segUtils import *
#from models import UNET
from unetModel import UnetModel

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def train_epoch( model, optimizer, criterion, loader, epoch, epochs ):

    losses = []
    num_correct = num_samples = 0
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch}/{epochs}")
    for batch_idx, (data, labels) in enumerate(loop):

        data = data.to(device)
        labels = labels.long().squeeze(1).to(device)

        preds = model(data)
        loss = criterion(preds, labels)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=np.average(losses))

    return np.average(losses)


def train_model( train_loader, val_loader, model, optimizer, criterion, epochs, chackpoint_dir="."):

    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []

    best_val_loss = 10**10

    for epoch in range(epochs):

        model.train()
        loss = train_epoch(model, optimizer, criterion, train_loader, epoch, epochs)
        all_train_loss.append(loss)

        model.eval()
        val_loss = chack_acc(model, val_loader, criterion, device)
        all_val_loss.append(val_loss)

        if val_loss < best_val_loss and epoch > 1:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, os.path.join(chackpoint_dir,"best_model.pb") )

        #print(f"{epoch}/{epochs}: loss: {loss:.3f}, acc: {acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}")   
        print(f"{epoch}/{epochs}: loss: {loss:.3f}, val_loss: {val_loss:.3f}")   

    history = {"train_loss":all_train_loss, "train_acc":all_train_acc, "val_loss":all_val_loss, "val_acc":all_val_acc}
    return model, history



def main():

    datapath = r"D:\programing\DataSets\segmentain\CamVid"
    dataset_path = Path(datapath)
    train_data, valid_data, test_data = get_data_pairs(dataset_path)
    class_map = get_class_map(dataset_path)
    num_classes = len(class_map)

    #data = np.load(dataset_path/"images.npy", allow_pickle=True)

    learning_rate = 0.001
    epochs = 35
    batch_size = 4
    input_size = (256,256)

    transforms = tt.Compose([tt.ToTensor()])

    train_ds = segmentationDataset(train_data+test_data, class_map, input_size, transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_ds = segmentationDataset(valid_data, class_map, input_size, transforms)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = UnetModel(3, num_classes, [64, 128, 256, 512]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    chackpoint_dir = dataset_path/"model_chackpoints"
    os.makedirs(chackpoint_dir, exist_ok=True)
    model, history = train_model( train_loader, val_loader, model, optimizer, criterion, epochs, chackpoint_dir=chackpoint_dir)

    x = 0




if __name__ == "__main__":
    main()