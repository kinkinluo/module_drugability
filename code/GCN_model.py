# GCN_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool

class GCN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=128, pooling_ratio=0.5, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.sag1 = SAGPooling(hidden_dim, pooling_ratio)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.sag2 = SAGPooling(hidden_dim, pooling_ratio)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.sag3 = SAGPooling(hidden_dim, pooling_ratio)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.sag4 = SAGPooling(hidden_dim, pooling_ratio)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.sag1(x, edge_index, batch=batch)
        x = self.dropout(x)

        # 第二层
        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.sag2(x, edge_index, batch=batch)
        x = self.dropout(x)

        # 第三层
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x, edge_index, _, batch, _, _ = self.sag3(x, edge_index, batch=batch)
        x = self.dropout(x)

        # 第四层
        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x, edge_index, _, batch, _, _ = self.sag4(x, edge_index, batch=batch)
        x = self.dropout(x)

        # 全局池化
        x = global_mean_pool(x, batch)

        return x  # 返回图级别的特征