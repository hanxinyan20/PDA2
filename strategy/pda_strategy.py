import numpy as np
from .base_strategy import DomainWiseAcquisitionStrategy
from dataset import DomainWiseData
class PDADomainWiseAcquisitionStrategy(DomainWiseAcquisitionStrategy):
    def __init__(self, acquisition_config, domain_wise_data:DomainWiseData, init_exploration:bool = True):

        super().__init__(acquisition_config, domain_wise_data)
        self.counts = np.zeros(self.n_regions) 
        self.values = np.zeros(self.n_regions)  
        self.alpha = acquisition_config['acquisition_function']['alpha'] 
        # self.init_exploration = init_exploration
        
    # def set_prior_values(self, values:np.ndarray):
    #     assert len(values) == self.n_regions, "Values array length must match the number of regions."
    #     self.values = values
    #     self.counts = np.zeros(self.n_regions)
        
    def select_region(self):
        # if self.init_exploration:
        for s in range(self.n_regions):
            if self.counts[s] == 0:
                return s
        ucb_values = [0.0 for _ in range(self.n_regions)]
        total_counts = np.sum(self.counts)
        
        for s in range(self.n_regions):
            bonus = self.alpha * np.sqrt(np.log(total_counts)/self.counts[s])
            ucb_values[s] = self.values[s] + bonus
        return np.argmax(ucb_values)
    
    def values_and_bouns(self):

        all_bouns = [0.0 for _ in range(self.n_regions)]
        total_counts = np.sum(self.counts)
        
        for s in range(self.n_regions):
            bonus = np.sqrt(self.alpha*np.log(total_counts)/self.counts[s])
            all_bouns[s] = bonus
        return self.values, all_bouns
    
    def update(self, region_idx:int, reward:float):
        
        self.counts[region_idx] += 1
        n = self.counts[region_idx]
        value = self.values[region_idx]
        new_value = ((n-1)*value + reward)/n
        self.values[region_idx] = new_value

    def mean_value(self):
        mean_value = np.sum(self.values * self.counts) / np.sum(self.counts)
        return mean_value
        
    
