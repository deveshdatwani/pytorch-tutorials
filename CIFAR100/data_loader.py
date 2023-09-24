import torch
import torchvision 

dataloader = torchvision.datasets.CIFAR100(root = "../datasets" , train = True, transform = None, target_transform = None)

print(len(dataloader))
