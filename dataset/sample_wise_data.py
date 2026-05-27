import torch
from .dataset_utils import *
from torch.utils.data import TensorDataset
from .base_data import DataManager
class SampleWiseData(DataManager):
    def __init__(self, data_config):
        super().__init__(data_config)
        x_pool, y_pool,region_ids = [], [], []
        for i, region in enumerate(self.other_regions):
            x, y = self.load_domain_data(region)
            x_pool.append(x)
            y_pool.append(y)
            region_ids.append(torch.full((x.size(0),), fill_value=i))
        x_pool = torch.cat(x_pool, dim=0)
        y_pool = torch.cat(y_pool, dim=0)
        region_ids = torch.cat(region_ids, dim=0)
        indices = torch.randperm(x_pool.size(0))
        self.x_pool = x_pool[indices]
        self.y_pool = y_pool[indices]
        self.region_ids = region_ids[indices]
        
        self.src_train_size = self.x_train.size(0)
        
        
    def remove_from_pool_to_train(self, inds):
        inds = torch.tensor(inds, dtype=torch.long)
        
        x_ = self.x_pool[inds]
        y_ = self.y_pool[inds]
        id_ = self.region_ids[inds]
        
        for i in id_:
            self.region_acquired_times[i] += 1
        
        self.x_train = torch.cat([self.x_train, self.x_pool[inds]])
        self.y_train = torch.cat([self.y_train, self.y_pool[inds]])
        
        mask = torch.ones(len(self.x_pool), dtype=torch.bool)
        mask[inds] = False
        
        self.x_pool = self.x_pool[mask]
        self.y_pool = self.y_pool[mask]
        self.region_ids = self.region_ids[mask]
        
        return x_, y_, id_


    
    
    def get_acquired_samples(self):
        return self.x_train[self.src_train_size:], self.y_train[self.src_train_size:]