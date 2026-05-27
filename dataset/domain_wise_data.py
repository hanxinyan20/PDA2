from torch.utils.data import Dataset, TensorDataset
import torch
from .dataset_utils import *
from .base_data import *

class DomainDataset(Dataset):
    def __init__(self, x:torch.Tensor, y:torch.Tensor, domain_name:str):
        self.x = x
        self.y = y
        self.domain_name = domain_name
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def acquire_samples(self, n_samples:int, replace:bool):
        if replace:
            indices = torch.randint(0, len(self), (n_samples,))
            return self.x[indices], self.y[indices]
        
        assert n_samples <= len(self), f"n_samples{n_samples} should be less than or equal to the length of the dataset{len(self)}"
        x_acquired = self.x[-n_samples:]
        y_acquired = self.y[-n_samples:]
        self.x = self.x[:-n_samples]
        self.y = self.y[:-n_samples]
        return x_acquired, y_acquired
    
    def split(self, val_ratio: float):
        total_size = len(self)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_x, train_y = self.x[train_indices], self.y[train_indices]
        val_x, val_y = self.x[val_indices], self.y[val_indices]

        train_dataset = DomainDataset(train_x, train_y, self.domain_name)
        val_dataset = DomainDataset(val_x, val_y, self.domain_name)

        return train_dataset, val_dataset
    
class DomainWiseData(DataManager):
    def __init__(self, data_config):
        super().__init__(data_config)
        self.other_region_datasets = []
        for region in self.other_regions:
            x, y = self.load_domain_data(region)
            dataset = DomainDataset(x, y, region)
            self.other_region_datasets.append(dataset)
        
    def acquire_samples(self, region_idx:int, n_samples:int, replace:bool):
        self.region_acquired_times[region_idx] += 1
        return self.other_region_datasets[region_idx].acquire_samples(n_samples, replace)
    