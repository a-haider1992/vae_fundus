import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os.path as osp
import numpy as np
import pdb

class ExplanationsPatchesDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file containing paths of all images.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.transform = transform

        # pdb.set_trace()
        
        df = pd.read_csv(osp.join(self.root_dir, txt_file), header=None, index_col=False, sep=',')
        self.data = df.values
        self.image_paths = np.array([osp.join(self.root_dir, x) for x in self.data[:, 0]])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image


class VAEDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file containing paths of all images.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.transform = transform

        # pdb.set_trace()
        
        df = pd.read_csv(osp.join(self.root_dir, txt_file), header=None, index_col=False, sep=',')
        self.data = df.values
        self.image_paths = np.array([osp.join(self.root_dir, x) for x in self.data[:, 0]])
        self.labels = np.array(self.data[:, 1])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):        
        # Load image
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
