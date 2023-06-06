import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as tt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torch.utils.data import dataloader


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

from dataset import ButterfliesDataset
from model import ViT


def plot_history(history, out_dir):

  plt.plot(history['train_acc'])
  plt.plot(history['val_acc'])
  plt.title(f"model accuracy")
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(out_dir,"acc graph.jpg"))

  plt.clf()

  plt.plot(history['train_loss'])
  plt.plot(history['val_loss'])
  plt.title(f"model loss")
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig( os.path.join(out_dir,"loss graph model.jpg") )


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def chack_acc(model, loader, criterion):
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for (data, labels) in loader:
            outputs = model(data.to(device))
            loss= criterion(outputs, labels.to(device)).item()
            losses.append(loss)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    return np.average(losses), correct / total


def train_epoch( model, optimizer, criterion, loader, epoch, epochs ):

    losses = []
    num_correct = num_samples = 0
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch}/{epochs}")
    for batch_idx, (data, labels) in enumerate(loop):

        data = data.to(device)
        labels = labels.to(device)

        preds = model(data)
        loss = criterion(preds, labels)

        _, predictions = preds.max(1)
        num_correct += (predictions == labels).sum().item()
        num_samples += predictions.size(0)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=np.average(losses), acc= num_correct/num_samples)

    return np.average(losses), num_correct/num_samples




def train_model( train_loader, val_loader, model, optimizer, criterion, epochs, chackpoint_dir="."):

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc= []

    best_val_loss = 10**10

    for epoch in range(epochs):

        model.train()

        loss, acc = train_epoch(model, optimizer, criterion, train_loader, epoch, epochs)
        all_train_loss.append(loss)
        all_train_acc.append(acc)

        model.eval()
        val_loss, val_acc = chack_acc(model, val_loader, criterion)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        if val_loss < best_val_loss and epoch > 1:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, os.path.join(chackpoint_dir,"best_model.pb") )

        print(f"{epoch}/{epochs}: loss: {loss:.3f}, acc: {acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}")   

    history = {"train_loss":all_train_loss, "train_acc":all_train_acc, "val_loss":all_val_loss, "val_acc":all_val_acc}
    return model, history


def get_dataset(path, class_dict):
    train_path = Path(path)
    X, Y = [], []
    for i, class_dir in enumerate( train_path.iterdir() ):
        images = list((class_dir).glob("*.jpg"))
        X.extend(images)
        Y.extend([class_dict[class_dir.name]]*len(images))

    return X, Y



def main():

    data_path = r"D:\data\butterflies"
    out_path = r"D:\data\butterflies\results"
    os.makedirs(out_path, exist_ok = True)

    class_dict_path = os.path.join(data_path, "class_dict.csv")
    class_dict_df = np.array(pd.read_csv(class_dict_path))[:,0:2]
    class_dict = {k: v for v, k in class_dict_df }

    trainset = get_dataset(os.path.join(data_path, "train"), class_dict)
    valset = get_dataset(os.path.join(data_path, "valid"), class_dict)
    testset = get_dataset(os.path.join(data_path, "test"), class_dict)

    train_tf = tt.Compose([         
        tt.Resize(size=(256,256)),
        tt.RandomHorizontalFlip(0.5),
        tt.RandomRotation(degrees=45),
        tt.RandomGrayscale(p=0.2),
        tt.ToTensor(),  
    ])
    test_tf= tt.Compose([   
        tt.Resize(size=(256,256)),
        tt.ToTensor(),
    ])

    batch_size = 64
    learning_rate = 0.001
    epochs = 3

    train_ds = ButterfliesDataset(trainset[0], trainset[1], transforms=train_tf)
    val_ds = ButterfliesDataset(valset[0], valset[1], transforms=test_tf)
    test_ds = ButterfliesDataset(testset[0], testset[1], transforms=test_tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=2)

    model =  ViT(   image_size = 256,
                    patch_size = 16,
                    num_classes = len(class_dict),
                    dim = 1024,
                    depth = 8,
                    heads = 16,
                    mlp_dim = 2048
                ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, history = train_model( train_loader, val_loader, model, optimizer, criterion, epochs, chackpoint_dir=out_path)

    plot_history(history, out_path)

    model.eval()
    test_loss, test_acc = chack_acc(model, test_loader, criterion)
    print(f"test_loss:{test_loss:.3f}, test_acc:{test_acc:.3f}")
        



if __name__ == '__main__':
  main()