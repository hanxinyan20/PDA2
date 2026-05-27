from torch.utils.data import TensorDataset
import torch
from .dataset_utils import *
class DataManager:
    def __init__(self, data_config):
        self.task = data_config['task']
        self.src_region = data_config['src_region']
        self.tgt_region = data_config['tgt_region']
        self.other_regions = data_config['other_regions']
        self.replace = data_config['replace']
        self.tgt_valid_size = data_config['tgt_valid_size']
        
        self.get_task = {
            "income_noisy": get_ACSIncome,
            "income": get_ACSIncome,
            "pubcov": get_ACSPubCov,
            "mobility": get_ACSMobility,
            "accident":get_USAccident,
        }[data_config['task']]
        self.x_src, self.y_src = self.load_domain_data(self.src_region)
        self.x_tgt, self.y_tgt = self.load_domain_data(self.tgt_region)
        self.x_train, self.y_train, self.x_src_valid, self.y_src_valid = self.split(self.x_src, self.y_src, val_ratio=data_config['src_valid_ratio'])
        self.x_tgt_test, self.y_tgt_test, self.x_tgt_valid, self.y_tgt_valid = self.split(self.x_tgt, self.y_tgt, val_ratio=data_config['tgt_valid_size']/self.x_tgt.size(0))
        
        self.n_features = self.x_src.size(1)
        self.n_labels = len(torch.unique(self.y_src))
        self.region_acquired_times = [0] * len(self.other_regions)
    
    def get_region_acquired_times(self):
        return {self.other_regions[i]: self.region_acquired_times[i] for i in range(len(self.other_regions))}
    def get_train_dataset(self):
        return TensorDataset(self.x_train, self.y_train)
    def get_tgt_test_dataset(self):
        return TensorDataset(self.x_tgt_test, self.y_tgt_test)
    def get_tgt_valid_dataset(self):
        return TensorDataset(self.x_tgt_valid, self.y_tgt_valid)
    def get_src_valid_dataset(self):
        return TensorDataset(self.x_src_valid, self.y_src_valid)
        
    def load_domain_data(self, region):
        if self.task == 'accident':
            root_dir = 'data/accident/US_Accidents_Dec21_updated.csv'
        else:
            root_dir = 'data'
        x, y, _ = self.get_task(state=region, 
                                            root_dir=root_dir)
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()      
    
    def split(self, x, y, val_ratio:float):
        total_size = len(x)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_x, train_y = x[train_indices], y[train_indices]
        val_x, val_y = x[val_indices], y[val_indices]

        return train_x, train_y, val_x, val_y