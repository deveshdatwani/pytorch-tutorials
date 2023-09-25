import torch
import argparse
from tqdm import tqdm
#from model import build_model
from torch import nn as nn
from dataset import dataloader
from matplotlib import pyplot as plt


class Trainer(object):
    def __init__(self, dataloader=None, model=None, lr=1e-5, opt=None, epochs=10, checkpoint="../fire_classifier.pt", lr_scheduler=None, batch_size=4):
        
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.model = model
        self.lr = lr
        self.opt = opt
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.lr_schedular = lr_scheduler


    def train(self):
        for i in range(epoch):
            self.train_one_epoch()
            # print epoch loss

    
    def train_one_epoch(self):
        for i, data in enumerate(self.dataloader):
            img, label = data
            continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0)
    parser.add_argument("--epoch", type=int, default="round")
    args = parser.parse_args()
    print(f'learning rate: {args.lr}')
    print(f'epoch: {args.epoch}')
    # model = build_model()
    # model_trainer = Trainer()
