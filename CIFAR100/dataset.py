import torchvision

CIFAR100_dataloader = torchvision.datasets.CIFAR100(root="../datasets", train=True, transform=None, target_transform=None)
