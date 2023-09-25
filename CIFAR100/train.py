import torch
import argparse
from tqdm import tqdm
#from model import build_model
from torch import nn as nn
from dataset import dataloader
from matplotlib import pyplot as plt


class Trainer(object):
    def __init__(self, dataloader=None, model=None, 
            lr=1e-5, opt=None, epochs=10, 
            checkpoint="../fire_classifier.pt", 
            lr_scheduler=None, batch_size=4, 
            device='cpu'):
        
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.model = model.to(device)
        self.lr = lr
        self.opt = opt
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.lr_schedular = lr_scheduler
        self.data_size = len(dataloader)


    def train(self):
        for i in range(epoch):
            epoch_loss = self.train_one_epoch()
            # print epoch loss


    def criterion(self, pred, target):
        loss = loss_func(pred, target)
        return loss


    def train_one_epoch(self):
        epoch_loss = 0
        
        for i, data in enumerate(self.dataloader):
            img, label = data
            self.opt.zer_grad()
            pred = self.model(img)
            loss = self.criterion(pred, target)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            print(f'running loss : {loss.item()}')
            print(f'batch number: {i}')

        return epoch_loss / self.data_size


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0)
    parser.add_argument("--epoch", type=int, default="round")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="../fire_classifier.pt")
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()
    
   
