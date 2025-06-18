# 第5章：教師なし学習 - クラスタリング

## 5.1 クラスタリングの基礎

クラスタリングは、ラベルのないデータを類似性に基づいてグループ（クラスタ）に分ける教師なし学習の手法です。

### クラスタリングの応用例
- **顧客セグメンテーション**: 購買行動に基づく顧客の分類
- **画像圧縮**: 色の類似性に基づくピクセルのグループ化
- **異常検知**: 正常データのクラスタから外れたデータの検出
- **文書分類**: トピックに基づく文書のグループ化

## 5.2 K-means法

### サンプルコード1：K-meansの基本実装

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# データの生成
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                      cluster_std=0.60, random_state=0)

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-meansクラスタリング
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 元のラベル
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
axes[0].set_title('True Labels')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-meansの結果
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
axes[1].set_title('K-means Clustering')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# クラスタの統計情報
print("クラスタごとのサンプル数:")
unique, counts = np.unique(y_kmeans, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  クラスタ {cluster}: {count} サンプル")
```

### サンプルコード2：最適なクラスタ数の決定

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# エルボー法とシルエット分析
inertias = []
silhouette_scores = []
db_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# エルボー曲線
axes[0].plot(K, inertias, 'bo-')
axes[0].set_xlabel('Number of clusters')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

# シルエットスコア
axes[1].plot(K, silhouette_scores, 'go-')
axes[1].set_xlabel('Number of clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True)

# Davies-Bouldinスコア
axes[2].plot(K, db_scores, 'ro-')
axes[2].set_xlabel('Number of clusters')
axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Score (lower is better)')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# 最適なクラスタ数の提案
optimal_k = K[np.argmax(silhouette_scores)]
print(f"\nシルエットスコアに基づく最適なクラスタ数: {optimal_k}")
```

### サンプルコード3：K-means++とミニバッチK-means

```python
from sklearn.cluster import MiniBatchKMeans
import time

# 大規模データセットの作成
X_large, y_large = make_blobs(n_samples=10000, centers=5, n_features=20, 
                             random_state=42)

# 通常のK-means
start_time = time.time()
kmeans_normal = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans_normal.fit(X_large)
kmeans_time = time.time() - start_time

# ミニバッチK-means
start_time = time.time()
mbkmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
mbkmeans.fit(X_large)
mbkmeans_time = time.time() - start_time

print(f"通常のK-means実行時間: {kmeans_time:.3f}秒")
print(f"ミニバッチK-means実行時間: {mbkmeans_time:.3f}秒")
print(f"高速化: {kmeans_time/mbkmeans_time:.2f}倍")

# 結果の比較
from sklearn.metrics import adjusted_rand_score

score = adjusted_rand_score(kmeans_normal.labels_, mbkmeans.labels_)
print(f"\n2つの手法の一致度 (Adjusted Rand Score): {score:.3f}")

# 初期化方法の比較
init_methods = ['k-means++', 'random']
n_init_values = [1, 10, 20]

results = []
for init in init_methods:
    for n_init in n_init_values:
        scores = []
        for _ in range(5):
            kmeans = KMeans(n_clusters=5, init=init, n_init=n_init, random_state=None)
            kmeans.fit(X_large)
            scores.append(kmeans.inertia_)
        
        results.append({
            'init': init,
            'n_init': n_init,
            'mean_inertia': np.mean(scores),
            'std_inertia': np.std(scores)
        })

import pandas as pd
results_df = pd.DataFrame(results)
print("\n初期化方法の比較:")
print(results_df)
```

## 5.3 階層的クラスタリング

### サンプルコード4：階層的クラスタリングと樹形図

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_moons

# データの生成（少数サンプル）
np.random.seed(42)
X_small, _ = make_blobs(n_samples=50, centers=3, n_features=2, 
                       cluster_std=0.5, random_state=42)

# 階層的クラスタリングの実行
linkage_methods = ['ward', 'complete', 'average', 'single']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # リンケージの計算
    Z = linkage(X_small, method=method)
    
    # 樹形図の描画
    ax = axes[idx]
    dendrogram(Z, ax=ax)
    ax.set_title(f'Dendrogram ({method} linkage)')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Distance')

plt.tight_layout()
plt.show()

# 異なるリンケージ方法でのクラスタリング結果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X_small)
    
    ax = axes[idx]
    scatter = ax.scatter(X_small[:, 0], X_small[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'{method.capitalize()} Linkage')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### サンプルコード5：距離行列を使った階層的クラスタリング

```python
from sklearn.metrics import pairwise_distances

# カスタム距離行列の作成
# 例：マンハッタン距離
distance_matrix = pairwise_distances(X_small, metric='manhattan')

# 距離行列を使った階層的クラスタリング
agg_precomputed = AgglomerativeClustering(
    n_clusters=3, 
    affinity='precomputed', 
    linkage='average'
)
labels_precomputed = agg_precomputed.fit_predict(distance_matrix)

# 通常のユークリッド距離との比較
agg_euclidean = AgglomerativeClustering(
    n_clusters=3, 
    affinity='euclidean', 
    linkage='average'
)
labels_euclidean = agg_euclidean.fit_predict(X_small)

# 結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_small[:, 0], X_small[:, 1], c=labels_euclidean, cmap='viridis')
axes[0].set_title('Euclidean Distance')

axes[1].scatter(X_small[:, 0], X_small[:, 1], c=labels_precomputed, cmap='viridis')
axes[1].set_title('Manhattan Distance')

plt.tight_layout()
plt.show()

# クラスタの切断高さの決定
from scipy.cluster.hierarchy import fcluster

Z = linkage(X_small, method='ward')

# 異なる切断高さでのクラスタ数
heights = [5, 10, 15, 20, 25]
for h in heights:
    clusters = fcluster(Z, h, criterion='distance')
    n_clusters = len(np.unique(clusters))
    print(f"切断高さ {h}: {n_clusters} クラスタ")
```

## 5.4 DBSCAN（密度ベースクラスタリング）

### サンプルコード6：DBSCANの実装と特徴

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles

# 様々な形状のデータセット
datasets = []

# 1. 月形データ
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
datasets.append(('Moons', X_moons))

# 2. 円形データ
X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, 
                                   random_state=42)
datasets.append(('Circles', X_circles))

# 3. ノイズを含むブロブデータ
X_blobs, _ = make_blobs(n_samples=200, centers=3, random_state=42)
X_noise = np.random.uniform(-6, 6, (50, 2))
X_noisy = np.vstack([X_blobs, X_noise])
datasets.append(('Noisy Blobs', X_noisy))

# 各データセットでDBSCANとK-meansを比較
fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 12))

for i, (name, X) in enumerate(datasets):
    # データの標準化
    X_scaled = StandardScaler().fit_transform(X)
    
    # 元のデータ
    axes[i, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[i, 0].set_title(f'{name} - Original')
    
    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    axes[i, 1].set_title(f'{name} - K-means')
    
    # DBSCAN
    if name == 'Moons':
        dbscan = DBSCAN(eps=0.3, min_samples=5)
    elif name == 'Circles':
        dbscan = DBSCAN(eps=0.3, min_samples=5)
    else:  # Noisy Blobs
        dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # ノイズポイントは黒で表示
    unique_labels = set(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'black'
        
        class_member_mask = (dbscan_labels == k)
        xy = X[class_member_mask]
        axes[i, 2].scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6, 
                          label='Noise' if k == -1 else f'Cluster {k}')
    
    axes[i, 2].set_title(f'{name} - DBSCAN')
    
    # クラスタ数とノイズの情報
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    axes[i, 2].text(0.02, 0.98, f'Clusters: {n_clusters}\nNoise: {n_noise}',
                    transform=axes[i, 2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
```

### サンプルコード7：DBSCANのパラメータ調整

```python
# パラメータグリッドサーチ
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# k-距離グラフでepsの推定
X_sample = X_moons
k = 5  # min_samplesの値

# 最近傍距離の計算
nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
distances, indices = nbrs.kneighbors(X_sample)

# k番目の最近傍距離をソート
distances = np.sort(distances[:, k-1], axis=0)

# k-距離グラフ
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points sorted by k-distance')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('k-distance Graph for Epsilon Selection')
plt.grid(True)
plt.show()

# パラメータの組み合わせを試す
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_samples_values = [3, 5, 10]

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_sample)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # シルエットスコア（ノイズを除く）
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                score = silhouette_score(X_sample[mask], labels[mask])
            else:
                score = -1
        else:
            score = -1
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': score
        })

results_df = pd.DataFrame(results)
print("DBSCANパラメータ探索結果:")
print(results_df.sort_values('silhouette', ascending=False).head(10))

# ベストパラメータでの可視化
best_params = results_df.loc[results_df['silhouette'].idxmax()]
dbscan_best = DBSCAN(eps=best_params['eps'], min_samples=int(best_params['min_samples']))
labels_best = dbscan_best.fit_predict(X_sample)

plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 1], c=labels_best, cmap='viridis', alpha=0.6)
plt.title(f"Best DBSCAN: eps={best_params['eps']}, min_samples={int(best_params['min_samples'])}")
plt.colorbar()
plt.show()
```

## 5.5 その他のクラスタリング手法

### サンプルコード8：Mean Shift、Spectral Clustering、Gaussian Mixture

```python
from sklearn.cluster import MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.mixture import GaussianMixture

# データの準備
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

# 各アルゴリズムの実装
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'Mean Shift': MeanShift(bandwidth=estimate_bandwidth(X, quantile=0.2)),
    'Spectral': SpectralClustering(n_clusters=4, random_state=42),
    'Gaussian Mixture': GaussianMixture(n_components=4, random_state=42)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, algorithm) in enumerate(algorithms.items()):
    # クラスタリング実行
    if name == 'Gaussian Mixture':
        algorithm.fit(X)
        labels = algorithm.predict(X)
    else:
        labels = algorithm.fit_predict(X)
    
    # 可視化
    ax = axes[idx]
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title(name)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # クラスタ中心の表示（可能な場合）
    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
                  s=200, linewidths=3)
    elif name == 'Gaussian Mixture':
        centers = algorithm.means_
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
                  s=200, linewidths=3)

plt.tight_layout()
plt.show()

# 各手法の特徴
print("クラスタリング手法の比較:")
for name, algorithm in algorithms.items():
    if name == 'Gaussian Mixture':
        algorithm.fit(X)
        labels = algorithm.predict(X)
    else:
        labels = algorithm.fit_predict(X)
    
    n_clusters = len(np.unique(labels))
    silhouette = silhouette_score(X, labels) if n_clusters > 1 else 0
    
    print(f"\n{name}:")
    print(f"  クラスタ数: {n_clusters}")
    print(f"  シルエットスコア: {silhouette:.3f}")
```

## 練習問題

### 問題1：顧客セグメンテーション
1. 顧客の購買データ（購入金額、頻度、最終購入日）を生成
2. RFM分析のためのクラスタリング実装
3. 各クラスタの特徴を分析し、マーケティング戦略を提案

### 問題2：画像の色量子化
1. カラー画像を読み込み
2. K-meansを使って色数を削減（16色、8色、4色）
3. 圧縮率と画質の関係を分析

### 問題3：異常検知への応用
1. 正常データとわずかな異常データを含むデータセットを生成
2. DBSCAN、Isolation Forest、One-Class SVMで異常検知
3. 各手法の性能を比較

## 解答

### 解答1：顧客セグメンテーション

```python
# RFM分析のためのデータ生成
np.random.seed(42)
n_customers = 1000

# 顧客データの生成
customer_data = pd.DataFrame({
    'customer_id': range(n_customers),
    'recency': np.random.exponential(30, n_customers),  # 最終購入からの日数
    'frequency': np.random.poisson(5, n_customers),  # 購入回数
    'monetary': np.random.lognormal(4, 1.5, n_customers)  # 総購入金額
})

# 外れ値の処理
for col in ['recency', 'frequency', 'monetary']:
    Q1 = customer_data[col].quantile(0.25)
    Q3 = customer_data[col].quantile(0.75)
    IQR = Q3 - Q1
    customer_data[col] = customer_data[col].clip(
        lower=Q1 - 1.5 * IQR,
        upper=Q3 + 1.5 * IQR
    )

print("顧客データの統計:")
print(customer_data[['recency', 'frequency', 'monetary']].describe())

# データの標準化
scaler = StandardScaler()
X_rfm = scaler.fit_transform(customer_data[['recency', 'frequency', 'monetary']])

# 最適なクラスタ数の決定
silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_rfm)
    score = silhouette_score(X_rfm, labels)
    silhouette_scores.append(score)

optimal_k = range(2, 8)[np.argmax(silhouette_scores)]
print(f"\n最適なクラスタ数: {optimal_k}")

# クラスタリング実行
kmeans_rfm = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['cluster'] = kmeans_rfm.fit_predict(X_rfm)

# 各クラスタの特徴分析
cluster_summary = customer_data.groupby('cluster')[['recency', 'frequency', 'monetary']].agg({
    'recency': ['mean', 'std'],
    'frequency': ['mean', 'std'],
    'monetary': ['mean', 'std']
}).round(2)

print("\nクラスタ別の統計:")
print(cluster_summary)

# クラスタの解釈とラベル付け
cluster_labels = {}
for cluster in range(optimal_k):
    cluster_data = customer_data[customer_data['cluster'] == cluster]
    avg_recency = cluster_data['recency'].mean()
    avg_frequency = cluster_data['frequency'].mean()
    avg_monetary = cluster_data['monetary'].mean()
    
    # 簡単なルールベースのラベル付け
    if avg_monetary > customer_data['monetary'].median() and avg_frequency > customer_data['frequency'].median():
        label = "優良顧客"
    elif avg_recency > customer_data['recency'].median():
        label = "休眠顧客"
    elif avg_frequency < customer_data['frequency'].quantile(0.25):
        label = "新規顧客"
    else:
        label = "一般顧客"
    
    cluster_labels[cluster] = label

customer_data['segment'] = customer_data['cluster'].map(cluster_labels)

# 可視化
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(optimal_k):
    cluster_data = customer_data[customer_data['cluster'] == cluster]
    ax.scatter(cluster_data['recency'], 
              cluster_data['frequency'], 
              cluster_data['monetary'],
              label=f'{cluster_labels[cluster]} (n={len(cluster_data)})',
              alpha=0.6)

ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (times)')
ax.set_zlabel('Monetary (amount)')
ax.set_title('Customer Segmentation - RFM Analysis')
ax.legend()
plt.show()

# マーケティング戦略の提案
print("\nマーケティング戦略の提案:")
for cluster, label in cluster_labels.items():
    n_customers = len(customer_data[customer_data['cluster'] == cluster])
    pct = n_customers / len(customer_data) * 100
    
    print(f"\n{label} (クラスタ{cluster}, {pct:.1f}%):")
    if label == "優良顧客":
        print("  - VIPプログラムへの招待")
        print("  - 限定商品の優先案内")
        print("  - パーソナライズされた特別オファー")
    elif label == "休眠顧客":
        print("  - 再活性化キャンペーン")
        print("  - 期間限定の割引クーポン")
        print("  - 「お久しぶり」メールの送信")
    elif label == "新規顧客":
        print("  - ウェルカムキャンペーン")
        print("  - 初回購入特典")
        print("  - 商品レコメンデーション")
    else:
        print("  - 定期的なプロモーション")
        print("  - ロイヤリティプログラムへの誘導")
        print("  - クロスセル・アップセルの機会探索")
```

### 解答2：画像の色量子化

```python
from sklearn.utils import shuffle
from PIL import Image
import urllib.request
import io

# サンプル画像の読み込み（または任意の画像を使用）
# ここでは単純な合成画像を作成
def create_sample_image():
    img_array = np.zeros((200, 200, 3), dtype=np.uint8)
    # 赤い円
    center = (50, 50)
    radius = 30
    y, x = np.ogrid[:200, :200]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img_array[mask] = [255, 0, 0]
    
    # 緑の四角
    img_array[100:150, 50:100] = [0, 255, 0]
    
    # 青いグラデーション
    for i in range(200):
        img_array[i, 120:180] = [0, 0, int(255 * i / 200)]
    
    return img_array

# 画像の準備
original_image = create_sample_image()
print(f"元の画像サイズ: {original_image.shape}")

# 色量子化の実装
def quantize_colors(image, n_colors):
    # 画像を2次元配列に変換
    w, h, d = image.shape
    image_array = image.reshape((w * h, d))
    
    # ランダムサンプリング（計算効率化）
    sample_size = min(1000, len(image_array))
    image_sample = shuffle(image_array, random_state=0)[:sample_size]
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(image_sample)
    
    # 全ピクセルにラベルを割り当て
    labels = kmeans.predict(image_array)
    
    # 各ピクセルをクラスタ中心の色に置換
    quantized = kmeans.cluster_centers_[labels]
    
    # 元の形状に戻す
    return quantized.reshape((w, h, d)).astype(np.uint8)

# 異なる色数での量子化
color_counts = [32, 16, 8, 4]
quantized_images = {}

for n_colors in color_counts:
    quantized_images[n_colors] = quantize_colors(original_image, n_colors)

# 結果の可視化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# 元の画像
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 量子化画像
for idx, (n_colors, img) in enumerate(quantized_images.items(), 1):
    axes[idx].imshow(img)
    axes[idx].set_title(f'{n_colors} Colors')
    axes[idx].axis('off')

# 最後のサブプロットは圧縮率の比較
ax = axes[-1]
original_size = original_image.size * original_image.itemsize
compressed_sizes = []
compression_ratios = []

for n_colors in color_counts:
    # 簡単な圧縮率の推定（実際はより複雑）
    # カラーパレット + インデックス画像
    palette_size = n_colors * 3 * 8  # ビット
    index_bits = np.ceil(np.log2(n_colors))
    index_size = original_image.shape[0] * original_image.shape[1] * index_bits
    total_size = (palette_size + index_size) / 8  # バイト
    
    compressed_sizes.append(total_size)
    compression_ratios.append(original_size / total_size)

ax.bar(range(len(color_counts)), compression_ratios)
ax.set_xticks(range(len(color_counts)))
ax.set_xticklabels(color_counts)
ax.set_xlabel('Number of Colors')
ax.set_ylabel('Compression Ratio')
ax.set_title('Compression Efficiency')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 色の分布分析
print("\n各量子化レベルでの色分布:")
for n_colors in color_counts:
    img_flat = quantized_images[n_colors].reshape(-1, 3)
    unique_colors = np.unique(img_flat, axis=0)
    print(f"\n{n_colors}色量子化:")
    print(f"  実際の色数: {len(unique_colors)}")
    print(f"  圧縮率: {compression_ratios[color_counts.index(n_colors)]:.2f}x")
```

### 解答3：異常検知への応用

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score

# データの生成
np.random.seed(42)

# 正常データ（2つのクラスタ）
n_normal = 500
X_normal1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_normal // 2)
X_normal2 = np.random.multivariate_normal([5, 5], [[1.5, 0], [0, 1.5]], n_normal // 2)
X_normal = np.vstack([X_normal1, X_normal2])

# 異常データ
n_anomaly = 50
X_anomaly = np.random.uniform(-5, 10, (n_anomaly, 2))

# 全データの結合
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomaly)])

# データのシャッフル
shuffle_idx = np.random.permutation(len(X_all))
X_all = X_all[shuffle_idx]
y_true = y_true[shuffle_idx]

# 異常検知手法の実装
methods = {
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'One-Class SVM': OneClassSVM(gamma='scale', nu=0.1)
}

results = {}
predictions = {}

for name, method in methods.items():
    if name == 'DBSCAN':
        # DBSCANの場合、ノイズポイントを異常とみなす
        labels = method.fit_predict(X_all)
        y_pred = np.where(labels == -1, -1, 1)
    else:
        # Isolation ForestとOne-Class SVMは直接異常スコアを返す
        y_pred = method.fit_predict(X_all)
    
    predictions[name] = y_pred
    
    # 評価指標の計算
    tp = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == 1) & (y_pred == -1))
    tn = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == -1) & (y_pred == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Detected Anomalies': np.sum(y_pred == -1)
    }

# 結果の表示
results_df = pd.DataFrame(results).T
print("異常検知手法の比較:")
print(results_df.round(3))

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# 真のラベル
axes[0].scatter(X_all[y_true == 1, 0], X_all[y_true == 1, 1], 
                c='blue', alpha=0.6, label='Normal')
axes[0].scatter(X_all[y_true == -1, 0], X_all[y_true == -1, 1], 
                c='red', alpha=0.6, label='Anomaly')
axes[0].set_title('True Labels')
axes[0].legend()

# 各手法の結果
for idx, (name, y_pred) in enumerate(predictions.items(), 1):
    ax = axes[idx]
    
    # 正常と判定されたポイント
    normal_mask = y_pred == 1
    anomaly_mask = y_pred == -1
    
    ax.scatter(X_all[normal_mask, 0], X_all[normal_mask, 1], 
              c='blue', alpha=0.6, label='Normal')
    ax.scatter(X_all[anomaly_mask, 0], X_all[anomaly_mask, 1], 
              c='red', alpha=0.6, label='Anomaly')
    
    # 誤検知と見逃しをハイライト
    false_positive_mask = (y_true == 1) & (y_pred == -1)
    false_negative_mask = (y_true == -1) & (y_pred == 1)
    
    ax.scatter(X_all[false_positive_mask, 0], X_all[false_positive_mask, 1], 
              c='orange', marker='x', s=100, label='False Positive')
    ax.scatter(X_all[false_negative_mask, 0], X_all[false_negative_mask, 1], 
              c='purple', marker='x', s=100, label='False Negative')
    
    ax.set_title(f'{name}\nF1-Score: {results[name]["F1-Score"]:.3f}')
    ax.legend()

plt.tight_layout()
plt.show()

# 異常スコアの分布（Isolation Forestの場合）
if 'Isolation Forest' in methods:
    iso_forest = methods['Isolation Forest']
    anomaly_scores = iso_forest.decision_function(X_all)
    
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores[y_true == 1], bins=30, alpha=0.5, label='Normal', density=True)
    plt.hist(anomaly_scores[y_true == -1], bins=30, alpha=0.5, label='Anomaly', density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores (Isolation Forest)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## まとめ

この章では、教師なし学習のクラスタリングについて学習しました：

- **K-means法**: 最も基本的で高速なクラスタリング手法
- **階層的クラスタリング**: データの階層構造を理解できる
- **DBSCAN**: 任意の形状のクラスタと外れ値の検出が可能
- **その他の手法**: Mean Shift、Spectral Clustering、Gaussian Mixture
- **実用的な応用**: 顧客セグメンテーション、画像処理、異常検知

次章では、もう一つの重要な教師なし学習手法である次元削減について学習します。

[目次に戻る](README.md)