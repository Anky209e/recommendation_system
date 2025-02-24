import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


class ReadingDataset(Dataset):
    def __init__(self,user_indices,item_indices,ratings):
        self.user_indices = torch.tensor(user_indices,dtype=torch.long)
        self.item_indices = torch.tensor(item_indices,dtype=torch.long)
        self.ratings = torch.tensor(ratings,dtype=torch.float)/100 
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return (
            self.user_indices[index],
            self.item_indices[index],
            self.ratings[index]
        )

