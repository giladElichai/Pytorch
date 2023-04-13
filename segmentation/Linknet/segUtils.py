
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def make_pair(dataset, set):
    train_imgs = list((dataset / set).glob("*.png"))
    label = f"{set}_labels"
    pairs = []
    for im in train_imgs:
        if (dataset / label / (im.stem +"_L.png" )).exists():
            pairs.append((im , dataset / label / (im.stem +"_L.png")))
    return pairs

def get_data_pairs(dataset_path):
    train_pairs = make_pair(dataset_path, "train")
    val_pairs = make_pair(dataset_path, "val")
    test_pairs = make_pair(dataset_path, "test")

    return train_pairs, val_pairs, test_pairs

def get_class_map(dataset_path):
    class_map_df = pd.read_csv( dataset_path / "class_dict.csv")
    class_map = []
    for index,item in class_map_df.iterrows():
        class_map.append(np.array([item['r'], item['g'], item['b']]))
    return class_map




def chack_acc(model, loader, criterion, device):
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for (data, labels) in loader:
            outputs = model(data.to(device))
            loss= criterion(outputs, labels.long().squeeze(1).to(device)).item()
            losses.append(loss)
    return np.average(losses)



ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #first compute binary cross-entropy 
        BCE = nn.CrossEntropyLoss()(inputs, targets)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss