# protein_feature_extraction.py

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from GCN_model import GCN

def extract_features(model, data_list, device, batch_size=32):
    """
    使用GCN模型提取所有蛋白质的结构特征
    :param model: GCN模型实例
    :param data_list: list of Data
    :param device: torch.device
    :param batch_size: int
    :return: 2D NumPy array, shape (num_proteins, output_dim)
    """
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)  # 假设输出形状为 (batch_size, output_dim)
            features.append(out.cpu().numpy())
    
    # 将所有批次的特征垂直堆叠成一个二维数组
    features = np.vstack(features)
    return features