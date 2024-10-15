# hyperparameter_tuning.py

import itertools
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
from model import GCN
from evaluation import plot_silhouette_score

def grid_search(model_class, data_list, device, param_grid, n_clusters=10, batch_size=32):
    """
    使用网格搜索对GCN模型进行超参数调优
    :param model_class: GCN模型类
    :param data_list: list of Data
    :param device: torch.device
    :param param_grid: dict, {'hidden_dim': [64, 128], 'dropout_rate': [0.3, 0.5], 'pooling_ratio': [0.5, 0.7]}
    :param n_clusters: int
    :param batch_size: int
    :return: dict, best_params and best_score
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -1
    best_params = None
    
    for params in tqdm(param_combinations, desc="Grid Search"):
        # 初始化模型
        model = model_class(
            input_dim=7,  # 根据您的特征维度调整
            hidden_dim=params['hidden_dim'],
            output_dim=128,  # 可以固定或作为参数
            pooling_ratio=params['pooling_ratio'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        model.eval()
        
        # 创建DataLoader
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        
        # 提取特征
        extracted_features = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                features = model(batch.x, batch.edge_index, batch.batch).cpu().numpy()
                for i in range(batch.num_graphs):
                    extracted_features.append(features[i])
        extracted_features = np.array(extracted_features)
        
        # 评估特征质量
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(extracted_features)
            sil_score = silhouette_score(extracted_features, cluster_labels)
        except:
            sil_score = -1
        
        # 更新最佳参数
        if sil_score > best_score:
            best_score = sil_score
            best_params = params
    
    print(f"Best Silhouette Score: {best_score:.4f} with parameters: {best_params}")
    return best_params, best_score