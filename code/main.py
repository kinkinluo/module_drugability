# main.py

import argparse
import torch
import numpy as np
from data_processing import process_proteins
from feature_extraction import extract_features
from evaluation import visualize_features, plot_silhouette_score
from hyperparameter_tuning import grid_search
from model import GCN

def main(args):
    # 1. 数据处理
    print("Starting data processing...")
    data_list = process_proteins(
        csv_path=args.csv,
        pdb_folder=args.pdb_folder,
        all_assign_path=args.all_assign,
        distance_threshold=args.distance
    )
    print(f"Processed {len(data_list)} proteins.")
    
    # 2. 特征提取
    print("Extracting features with default GCN...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    default_gcn = GCN(
        input_dim=7,  # 根据您的特征维度调整
        hidden_dim=128,
        output_dim=128,
        pooling_ratio=0.5,
        dropout_rate=0.5
    ).to(device)
    default_gcn.eval()
    
    extracted_features = extract_features(default_gcn, data_list, device, batch_size=args.batch_size)
    
    # 将提取的特征保存
    np.save(args.output_features, extracted_features)
    print(f"Saved extracted features to {args.output_features}")
    
    # 3. 评估与可视化
    # 聚合所有特征（例如，取平均）
    protein_features = np.array([item['feature'] for item in extracted_features])
    print("Visualizing features with PCA...")
    visualize_features(protein_features, method='pca')
    print("Visualizing features with t-SNE...")
    visualize_features(protein_features, method='tsne')
    
    # 计算并打印轮廓系数
    sil_score = plot_silhouette_score(protein_features, n_clusters=10)
    
    # 4. 超参数调优（网格搜索）
    if args.grid_search:
        print("Starting hyperparameter tuning with grid search...")
        param_grid = {
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.3, 0.5, 0.7],
            'pooling_ratio': [0.3, 0.5, 0.7]
        }
        best_params, best_score = grid_search(
            model_class=GCN,
            data_list=data_list,
            device=device,
            param_grid=param_grid,
            n_clusters=10,
            batch_size=args.batch_size
        )
        print(f"Best Parameters: {best_params}, Best Silhouette Score: {best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein Feature Extraction using GCN.')
    parser.add_argument('--csv', required=True, type=str, help="Path to protein CSV file.")
    parser.add_argument('--pdb_folder', required=True, type=str, help="Path to folder containing PDB files.")
    parser.add_argument('--all_assign', required=True, type=str, help="Path to all_assign.txt file.")
    parser.add_argument('--output_features', default='protein_features.npy', type=str, help="Output file to save extracted features.")
    parser.add_argument('--distance', default=7.5, type=float, help="Distance threshold for contacts.")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size for feature extraction and grid search.")
    parser.add_argument('--grid_search', action='store_true', help="Enable hyperparameter grid search.")
    args = parser.parse_args()
    
    main(args)