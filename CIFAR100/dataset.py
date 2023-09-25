import torchvision
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

FIRE_ROOT = "/home/deveshdatwani/Tutorials/datasets/fire_dataset/labels.csv"

CIFAR100_dataloader = torchvision.datasets.CIFAR100(root="../datasets", train=True, transform=None, target_transform=None)

class FIRE_DATASET(Dataset):
    def __init__(self, root="./", 
                 data_csv=FIRE_ROOT, 
                 transforms=None):
        self.root = root
        self.data_df = pd.read_csv(data_csv)
        self.transforms = transforms
        self.len = len(self.data_df)

    
    def __getitem__(self, idx):
        img = read_image(self.data_df.iloc[idx]['img'])
        label = self.data_df.iloc[idx]['label']
        
        if self.transforms:
            img = self.transforms(img)

        data = {'img': img, 'label': label}

        return data
    

    def __len__(self):
        return self.len

