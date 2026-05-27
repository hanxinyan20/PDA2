
from .base_strategy import DomainWiseAcquisitionStrategy
from dataset import DomainWiseData
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances, device):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth.to(device)

    def forward(self, X, device):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        L2_distances = (torch.cdist(X, X) ** 2).to(device)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances, device) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMD(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel
        
    def forward(self, X, Y, device, weight = None):
        self.kernel.to(device)
        Y = Y.reshape(-1, X.shape[1])
        K = self.kernel(torch.vstack([X, Y]), device)
        X_size = X.shape[0]
        # print(X.shape, Y.shape, K.shape)
        if weight is not None:
            xweight = torch.abs(weight)
            # xweight = xweight / xweight.sum()
            yweight = torch.ones(Y.shape[0]).to(device)
            # yweight = yweight / yweight.sum()

            XXweight = torch.outer(xweight, xweight)
            XYweight = torch.outer(xweight, yweight)
            YYweight = torch.outer(yweight, yweight)

            XX = (K[:X_size, :X_size] * XXweight).mean()
            XY = (K[:X_size, X_size:] * XYweight).mean()
            YY = (K[X_size:, X_size:] * YYweight).mean()
        else:
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY
def mmdTorch(X, y, num_steps, lr, lambdda,  device , selected_n_list = None,):
    n, p = X.shape
    if selected_n_list is None:
        weight = torch.ones(n, dtype=float)
    else:
        weight = torch.ones(len(selected_n_list), dtype=float)
    weight = weight.to(device)
    weight.requires_grad = True
    optimizer = optim.Adam([weight,], lr = lr)
    dis_list = []
    mmd = MMD()
    selected_n_list = torch.tensor(selected_n_list).to(device)

    for i in range(num_steps):
        optimizer.zero_grad()
        if selected_n_list is not None:
            sample_wise_weight = weight.repeat_interleave(selected_n_list)
            dis = mmd(X, y, device, weight = sample_wise_weight)
        else:
            dis = mmd(X, y, device, weight = weight)
        # lasso
        loss = dis + lambdda * abs(weight).sum()
        if i == 0 or (i + 1) % 500 == 0:
            # print(f"iter {i}  dis: {dis}  loss: {loss} weight: {weight.tolist()}")
            dis_list.append(dis.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

    return abs(weight).cpu().detach().numpy(), dis_list

class DAAAcquisitionStrategy(DomainWiseAcquisitionStrategy):
    def __init__(self, acquisition_config, domain_wise_data:DomainWiseData):
        super().__init__(acquisition_config, domain_wise_data)
        self.selected_x_list = []
        
        self.t = 0
    def select_region(self):
        if self.t < self.n_regions:
            region_idx = self.t
            self.t = self.t + 1
            return np.array([region_idx]*self.n_samples_per_round)
        else:
            selected_n_list = [selected_x.shape[0] for selected_x in self.selected_x_list]
            all_x = torch.cat([selected_x for selected_x in self.selected_x_list], dim=0)
            weight, _ = mmdTorch(all_x,self.domain_wise_data.get_tgt_valid_dataset()[:][0], num_steps=1000, lr = 0.01, lambdda = 0.001, selected_n_list = selected_n_list, device='cuda')
            normalized_weight = weight / weight.sum()
            region_idx = np.random.choice(len(weight), size = self.n_samples_per_round , p = normalized_weight)
            return region_idx
    def acquire(self, n_samples:int):
        region_idx = self.select_region()
        x_acquired = []
        y_acquired = []
        for idx in region_idx:
            x, y = self.domain_wise_data.acquire_samples(
                region_idx=idx,
                n_samples=1,
                replace=self.acquisition_config['replace'],
            )
            if len(self.selected_x_list) <= idx:
                self.selected_x_list.append(x)
            else:
                self.selected_x_list[idx] = torch.cat([self.selected_x_list[idx], x], dim=0)
            x_acquired.append(x)
            y_acquired.append(y)
        x_acquired = torch.cat(x_acquired, dim=0)
        y_acquired = torch.cat(y_acquired, dim=0)
        return x_acquired, y_acquired, region_idx