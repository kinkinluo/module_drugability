# GCN_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def visualize_features(features, method='pca', n_components=2, perplexity=30, random_state=42):
    """
    可视化特征分布
    :param features: numpy array, shape (num_samples, feature_dim)
    :param method: 'pca' 或 'tsne'
    :param n_components: int
    :param perplexity: int (仅用于 t-SNE)
    :param random_state: int
    :return: None
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(features)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:,0], reduced[:,1], s=50, alpha=0.7)
        plt.title('PCA of Protein Features')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
    
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, n_iter=300)
        reduced = reducer.fit_transform(features)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:,0], reduced[:,1], s=50, alpha=0.7)
        plt.title('t-SNE of Protein Features')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
    
    else:
        raise ValueError("Unsupported method. Choose 'pca' or 'tsne'.")

def plot_silhouette_score(features, n_clusters=10):
    """
    使用K-Means聚类并绘制轮廓系数
    :param features: numpy array, shape (num_samples, feature_dim)
    :param n_clusters: int
    :return: float
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    sil_score = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score with {n_clusters} clusters: {sil_score:.4f}")
    return sil_score

if __name__ == "__main__":
    # 加载特征
    features = np.load('features.npy')  # 确保 features.npy 是一个二维数组
    print("Feature matrix shape:", features.shape)
    
    # 可视化特征
    visualize_features(features, method='pca')
    visualize_features(features, method='tsne')
    
    # 计算并打印轮廓系数
    plot_silhouette_score(features, n_clusters=10)