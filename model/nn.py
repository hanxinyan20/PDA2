
import torch.nn as nn
import torch
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, num_classes):
        super(MultiClassClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        layers.append(nn.BatchNorm1d(layer_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)         # 第二个全连接层
        self.fc3 = nn.Linear(64, 1)           # 输出层，输出一个预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# class MultiClassClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(MultiClassClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         # add bathc normalization layer
#         self.bn1 = nn.BatchNorm1d(64)
#         self.fc2 = nn.Linear(64, 32)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.fc3 = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.bn1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.bn2(x)
#         x = self.fc3(x)
#         return x