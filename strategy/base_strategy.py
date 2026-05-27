from dataset import DomainWiseData, SampleWiseData, DomainDataset
import numpy as np
from torch import nn
from torch.optim import Adam
import torch
class AcquisitionStrategy:
    def __init__(self, acquisition_config):
        self.acquisition_config = acquisition_config
        for k, v in acquisition_config.items():
            setattr(self, k, v)
    
    def acquire(self, n_samples:int):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SampleWiseAcquisitionStrategy(AcquisitionStrategy):
    def __init__(self, acquisition_config, sample_wise_data:SampleWiseData):
        super().__init__(acquisition_config)
        self.sample_wise_data = sample_wise_data
    
    def acquire(self, n_samples:int):
        raise NotImplementedError("This method should be overridden by subclasses.")

class RandomAcquisitionStrategy(SampleWiseAcquisitionStrategy):
    def __init__(self, acquisition_config, sample_wise_data:SampleWiseData):
        super().__init__(acquisition_config, sample_wise_data)
    
    def acquire(self, n_samples:int):
        pool_size = self.sample_wise_data.x_pool.shape[0]
        if pool_size < n_samples:
            raise ValueError(f"Not enough samples in the pool. Pool size: {pool_size}, requested: {n_samples}")
        indices = np.random.choice(pool_size, n_samples, replace=False)
        x_acquired, y_acquired, id_ = self.sample_wise_data.remove_from_pool_to_train(indices)
        return x_acquired, y_acquired, id_

class DomainWiseAcquisitionStrategy(AcquisitionStrategy):
    def __init__(self, acquisition_config, domain_wise_data:DomainWiseData):
        super().__init__(acquisition_config)
        self.n_regions= len(domain_wise_data.other_regions)
        self.domain_wise_data = domain_wise_data
        
    def select_region(self):
        raise NotImplementedError("This method should be overridden by subclasses.")    
    
    def acquire(self, n_samples:int):
        region_idx = self.select_region()
        x_acquired, y_acquired =  self.domain_wise_data.acquire_samples(
            region_idx=region_idx,
            n_samples=n_samples,
            replace=self.acquisition_config['replace'],
        )
        return x_acquired, y_acquired, region_idx

class UniformAcquisitionStrategy(DomainWiseAcquisitionStrategy):
    def __init__(self, acquisition_config, domain_wise_data:DomainWiseData):
        super().__init__(acquisition_config, domain_wise_data)
    
    def select_region(self):
        return np.random.choice(self.n_regions)

class OracleAcquisitionStrategy(DomainWiseAcquisitionStrategy):
    def __init__(self, acquisition_config, domain_wise_data:DomainWiseData):
        super().__init__(acquisition_config, domain_wise_data)
        all_tgt_x = self.domain_wise_data.x_tgt
        all_tgt_y = self.domain_wise_data.y_tgt
        self.tgt_x, self.tgt_y, tgt_test_x, tgt_test_y = self.domain_wise_data.split(all_tgt_x, all_tgt_y, val_ratio=0.5)
        self.tgt_region_dataset = DomainDataset(self.tgt_x, self.tgt_y, "tgt_region")
        self.domain_wise_data.x_tgt_test = tgt_test_x
        self.domain_wise_data.y_tgt_test = tgt_test_y
    def acquire(self, n_samples):
        x, y =  self.tgt_region_dataset.acquire_samples(
            n_samples=n_samples,
            replace=self.acquisition_config['replace'],
        )
        return x, y, -1
        