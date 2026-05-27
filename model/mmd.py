import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LassoCV

device = "cuda" if torch.cuda.is_available() else "cpu"
        
def cosine_loss(X, weight, y):
    weighted_center = torch.mm(X, abs(weight).unsqueeze(1)).reshape(-1)
    dis = 1 - F.cosine_similarity(weighted_center, y, dim = 0)
    return dis

def cosineTorch(X, y, num_steps, lr, lambdda):
    n, p = X.shape
    # print(X.shape, y.shape)
    weight = torch.ones(p)
    weight = weight.to(device)
    weight.requires_grad = True
    optimizer = optim.Adam([weight,], lr = lr)
    dis_list = []
    for i in range(num_steps):
        optimizer.zero_grad()
        dis = cosine_loss(X, weight, y)
        # lasso
        loss = dis + lambdda * abs(weight).sum()
        if i == 0 or (i + 1) % 500 == 0:
            print(f"iter {i}  dis: {dis}  loss: {loss}")
            dis_list.append(dis.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

    return abs(weight).cpu().detach().numpy(), dis_list

def mmdTorch(X, y, num_steps, lr, lambdda, selected_n_list = None):
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

def mseSklearn(X, y, alphas):
    X = np.array([item.detach().cpu().numpy() for item in X]).T
    # print(X.shape, y.shape)
    model = LassoCV(max_iter=3000, alphas=alphas)
    model.fit(X, y.flatten())
    return model.coef_

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
