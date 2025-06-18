#!/usr/bin/env python3
"""
クラスタリングアルゴリズムの包括的なサンプルスクリプト
K-means、階層的クラスタリング、DBSCANを含む
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_clustering_datasets():
    """様々な形状のクラスタリング用データセットを生成"""
    np.random.seed(42)
    
    # 1. Blobデータ（明確に分離されたクラスタ）
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2,
                                  cluster_std=0.5, random_state=42)
    
    # 2. Moonsデータ（非凸形状）
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # 3. Circlesデータ（同心円）
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5,
                                       random_state=42)
    
    datasets = {
        'Blobs': (X_blobs, y_blobs),
        'Moons': (X_moons, y_moons),
        'Circles': (X_circles, y_circles)
    }
    
    return datasets

def demonstrate_kmeans():
    """K-meansクラスタリングの実演"""
    print("=== K-meansクラスタリング ===\n")
    
    # データの生成
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                          cluster_std=1.0, random_state=42)
    
    # エルボー法で最適なクラスタ数を決定
    inertias = []
    silhouette_scores = []
    K = range(2, 10)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # エルボー曲線
    axes[0, 0].plot(K, inertias, 'bo-')
    axes[0, 0].set_xlabel('Number of clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True)
    
    # シルエットスコア
    axes[0, 1].plot(K, silhouette_scores, 'ro-')
    axes[0, 1].set_xlabel('Number of clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score vs K')
    axes[0, 1].grid(True)
    
    # 最適なクラスタ数（k=4）でクラスタリング
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # クラスタリング結果
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    axes[1, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                      c='red', marker='x', s=200, linewidths=3)
    axes[1, 0].set_title('K-means Clustering Result (k=4)')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # 真のラベルとの比較
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    axes[1, 1].set_title('True Labels')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('kmeans_analysis.png')
    print("K-means分析を 'kmeans_analysis.png' として保存しました")
    
    # 評価指標
    print(f"\nシルエットスコア: {silhouette_score(X, y_pred):.3f}")
    print(f"Davies-Bouldin スコア: {davies_bouldin_score(X, y_pred):.3f}")

def demonstrate_hierarchical_clustering():
    """階層的クラスタリングの実演"""
    print("\n=== 階層的クラスタリング ===\n")
    
    # データの生成
    np.random.seed(42)
    X, _ = make_blobs(n_samples=50, centers=3, n_features=2,
                     cluster_std=0.5, random_state=42)
    
    # 樹形図の作成
    plt.figure(figsize=(12, 8))
    
    # リンケージの計算
    Z = linkage(X, method='ward')
    
    # 樹形図の描画
    plt.subplot(2, 2, 1)
    dendrogram(Z)
    plt.title('Dendrogram (Ward Linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # 異なるリンケージ方法の比較
    linkage_methods = ['ward', 'complete', 'average', 'single']
    
    for i, method in enumerate(linkage_methods):
        plt.subplot(2, 2, i+1)
        
        if i == 0:
            continue  # 最初のサブプロットは既に使用
        
        agg_clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
        y_pred = agg_clustering.fit_predict(X)
        
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
        plt.title(f'Hierarchical Clustering ({method})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('hierarchical_clustering.png')
    print("階層的クラスタリングを 'hierarchical_clustering.png' として保存しました")

def demonstrate_dbscan():
    """DBSCANクラスタリングの実演"""
    print("\n=== DBSCANクラスタリング ===\n")
    
    # ノイズを含むデータの生成
    np.random.seed(42)
    X1, _ = make_blobs(n_samples=200, centers=2, n_features=2,
                      cluster_std=0.5, random_state=42)
    X2, _ = make_blobs(n_samples=100, centers=1, n_features=2,
                      cluster_std=2.0, random_state=42)
    X2 = X2 + [6, 0]  # 位置をずらす
    
    # ノイズポイントの追加
    noise = np.random.uniform(-6, 6, (50, 2))
    X = np.vstack([X1, X2, noise])
    
    # パラメータの影響を調査
    eps_values = [0.3, 0.5, 1.0, 1.5]
    min_samples_values = [5, 10]
    
    fig, axes = plt.subplots(len(min_samples_values), len(eps_values), 
                            figsize=(15, 8))
    
    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(X)
            
            ax = axes[i, j] if len(min_samples_values) > 1 else axes[j]
            
            # ノイズポイントは-1として識別される
            unique_labels = set(y_pred)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # ノイズポイントは黒で表示
                    col = 'black'
                    marker = 'x'
                else:
                    marker = 'o'
                
                class_member_mask = (y_pred == k)
                xy = X[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                          s=50, alpha=0.6)
            
            ax.set_title(f'eps={eps}, min_samples={min_samples}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            # クラスタ数とノイズポイント数を表示
            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            n_noise = list(y_pred).count(-1)
            ax.text(0.02, 0.98, f'Clusters: {n_clusters}\nNoise: {n_noise}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dbscan_parameters.png')
    print("DBSCANパラメータの影響を 'dbscan_parameters.png' として保存しました")

def compare_clustering_algorithms():
    """異なるクラスタリングアルゴリズムの比較"""
    print("\n=== クラスタリングアルゴリズムの比較 ===\n")
    
    datasets = generate_clustering_datasets()
    
    fig, axes = plt.subplots(len(datasets), 4, figsize=(15, 12))
    
    for i, (name, (X, y_true)) in enumerate(datasets.items()):
        # データの標準化
        X = StandardScaler().fit_transform(X)
        
        # 元のデータ
        axes[i, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
        axes[i, 0].set_title(f'{name} - True Labels')
        if i == 0:
            axes[i, 0].set_ylabel('Original Data', fontsize=12)
        
        # K-means
        kmeans = KMeans(n_clusters=2, random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        axes[i, 1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
        axes[i, 1].set_title('K-means')
        
        # 階層的クラスタリング
        agg = AgglomerativeClustering(n_clusters=2)
        y_agg = agg.fit_predict(X)
        axes[i, 2].scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis', alpha=0.6)
        axes[i, 2].set_title('Hierarchical')
        
        # DBSCAN
        if name == 'Blobs':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
        elif name == 'Moons':
            dbscan = DBSCAN(eps=0.3, min_samples=5)
        else:  # Circles
            dbscan = DBSCAN(eps=0.2, min_samples=5)
        
        y_dbscan = dbscan.fit_predict(X)
        axes[i, 3].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)
        axes[i, 3].set_title('DBSCAN')
        
        # 評価指標の計算（DBSCANのノイズポイントを除く）
        mask = y_dbscan != -1
        if mask.sum() > 1:
            silhouette_kmeans = silhouette_score(X, y_kmeans)
            silhouette_agg = silhouette_score(X, y_agg)
            silhouette_dbscan = silhouette_score(X[mask], y_dbscan[mask]) if len(set(y_dbscan[mask])) > 1 else 0
            
            print(f"\n{name} データセット - シルエットスコア:")
            print(f"  K-means: {silhouette_kmeans:.3f}")
            print(f"  Hierarchical: {silhouette_agg:.3f}")
            print(f"  DBSCAN: {silhouette_dbscan:.3f}")
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png')
    print("\nクラスタリング手法の比較を 'clustering_comparison.png' として保存しました")

def demonstrate_high_dimensional_clustering():
    """高次元データのクラスタリング"""
    print("\n=== 高次元データのクラスタリング ===\n")
    
    # 高次元データの生成
    np.random.seed(42)
    n_samples = 300
    n_features = 50
    n_clusters = 3
    
    # 3つのクラスタを持つ高次元データ
    X_high = []
    y_true = []
    
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 5
        cluster_data = center + np.random.randn(n_samples // n_clusters, n_features)
        X_high.append(cluster_data)
        y_true.extend([i] * (n_samples // n_clusters))
    
    X_high = np.vstack(X_high)
    y_true = np.array(y_true)
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_high)
    
    print(f"元の次元数: {X_high.shape[1]}")
    print(f"PCA後の説明分散比: {pca.explained_variance_ratio_}")
    print(f"累積説明分散比: {pca.explained_variance_ratio_.sum():.3f}")
    
    # クラスタリング
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X_high)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真のラベル
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    axes[0].set_title('True Labels (PCA projection)')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    
    # クラスタリング結果
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    axes[1].set_title('K-means Clustering (PCA projection)')
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.savefig('high_dimensional_clustering.png')
    print("\n高次元クラスタリングを 'high_dimensional_clustering.png' として保存しました")

def main():
    print("=== scikit-learn クラスタリングサンプル ===\n")
    
    # 1. K-meansクラスタリング
    demonstrate_kmeans()
    
    # 2. 階層的クラスタリング
    demonstrate_hierarchical_clustering()
    
    # 3. DBSCANクラスタリング
    demonstrate_dbscan()
    
    # 4. アルゴリズムの比較
    compare_clustering_algorithms()
    
    # 5. 高次元データのクラスタリング
    demonstrate_high_dimensional_clustering()
    
    print("\n=== クラスタリング手法の選択指針 ===")
    print("1. K-means: クラスタ数が既知で、球状のクラスタを仮定できる場合")
    print("2. 階層的クラスタリング: クラスタの階層構造を理解したい場合")
    print("3. DBSCAN: クラスタ数が不明で、任意の形状のクラスタやノイズがある場合")
    print("4. 高次元データ: 次元削減（PCA等）と組み合わせて使用")

if __name__ == "__main__":
    main()