# 第6章：教師なし学習 - 次元削減

## 6.1 次元削減の必要性

高次元データは「次元の呪い」と呼ばれる問題を引き起こします。次元削減は以下の目的で使用されます：

- **可視化**: 高次元データを2次元・3次元で表現
- **ノイズ除去**: 重要でない変動を取り除く
- **計算効率の改善**: 特徴量を減らして処理速度を向上
- **過学習の防止**: パラメータ数を減らしてモデルを簡素化

## 6.2 主成分分析（PCA）

### サンプルコード1：PCAの基本実装

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
import seaborn as sns

# Irisデータセットでの例
iris = load_iris()
X = iris.data
y = iris.target

# データの標準化（PCAの前処理として重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 結果の可視化
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8,
                lw=2, label=target_name)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA of Iris Dataset')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, alpha=0.3)
plt.show()

# 主成分の解釈
components_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=iris.feature_names
)
print("主成分の構成:")
print(components_df)

# 各主成分の寄与度
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### サンプルコード2：最適な主成分数の選択

```python
# より高次元のデータセット（手書き数字）
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"元のデータ形状: {X_digits.shape}")

# 全主成分でPCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_digits)

# 説明分散比の累積和
cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)

# 95%の分散を説明するのに必要な主成分数
n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
print(f"95%の分散を説明するのに必要な主成分数: {n_components_95}")

# スクリープロット
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 個別の説明分散比
axes[0].plot(range(1, 21), pca_full.explained_variance_ratio_[:20], 'bo-')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot')
axes[0].grid(True, alpha=0.3)

# 累積説明分散比
axes[1].plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'ro-')
axes[1].axhline(y=0.95, color='k', linestyle='--', label='95% threshold')
axes[1].axvline(x=n_components_95, color='g', linestyle='--', 
                label=f'n_components={n_components_95}')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance Ratio')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 異なる主成分数での再構成誤差
n_components_list = [5, 10, 20, 30, 40, 50]
reconstruction_errors = []

for n_comp in n_components_list:
    pca_temp = PCA(n_components=n_comp)
    X_reduced = pca_temp.fit_transform(X_digits)
    X_reconstructed = pca_temp.inverse_transform(X_reduced)
    
    # 再構成誤差（MSE）
    mse = np.mean((X_digits - X_reconstructed) ** 2)
    reconstruction_errors.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(n_components_list, reconstruction_errors, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Reconstruction Error vs Number of Components')
plt.grid(True, alpha=0.3)
plt.show()
```

### サンプルコード3：画像データでのPCA（固有顔）

```python
from sklearn.datasets import fetch_olivetti_faces

# Olivetti顔データセットの読み込み
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data
y_faces = faces.target

print(f"データ形状: {X_faces.shape}")
print(f"画像サイズ: 64x64 = {64*64} ピクセル")

# PCAで次元削減
n_components = 150
pca_faces = PCA(n_components=n_components, whiten=True, random_state=42)
X_faces_pca = pca_faces.fit_transform(X_faces)

print(f"削減後の形状: {X_faces_pca.shape}")
print(f"圧縮率: {n_components}/{X_faces.shape[1]} = {n_components/X_faces.shape[1]:.2%}")

# 固有顔（主成分）の可視化
eigenfaces = pca_faces.components_.reshape((n_components, 64, 64))

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.ravel()

for i in range(15):
    axes[i].imshow(eigenfaces[i], cmap='gray')
    axes[i].set_title(f'Eigenface {i+1}')
    axes[i].axis('off')

plt.suptitle('Top 15 Eigenfaces')
plt.tight_layout()
plt.show()

# 元の画像と再構成画像の比較
n_samples_show = 5
sample_indices = np.random.choice(len(X_faces), n_samples_show, replace=False)

fig, axes = plt.subplots(3, n_samples_show, figsize=(15, 9))

for idx, sample_idx in enumerate(sample_indices):
    # 元の画像
    axes[0, idx].imshow(X_faces[sample_idx].reshape(64, 64), cmap='gray')
    axes[0, idx].set_title(f'Original #{sample_idx}')
    axes[0, idx].axis('off')
    
    # PCA適用後
    face_pca = X_faces_pca[sample_idx]
    
    # 再構成
    face_reconstructed = pca_faces.inverse_transform(face_pca.reshape(1, -1))
    axes[1, idx].imshow(face_reconstructed.reshape(64, 64), cmap='gray')
    axes[1, idx].set_title(f'Reconstructed')
    axes[1, idx].axis('off')
    
    # 差分
    diff = X_faces[sample_idx] - face_reconstructed.ravel()
    axes[2, idx].imshow(diff.reshape(64, 64), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[2, idx].set_title(f'Difference')
    axes[2, idx].axis('off')

plt.tight_layout()
plt.show()
```

## 6.3 t-SNE（t-distributed Stochastic Neighbor Embedding）

### サンプルコード4：t-SNEによる非線形次元削減

```python
from sklearn.manifold import TSNE

# MNISTのサブセットで実験
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# データのサブサンプリング（計算時間短縮のため）
n_samples = 1000
indices = np.random.choice(len(X_digits), n_samples, replace=False)
X_subset = X_digits[indices]
y_subset = y_digits[indices]

# t-SNEの実行（異なるperplexity値）
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)
    
    ax = axes[idx]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    ax.set_title(f't-SNE (perplexity={perplexity})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 各数字のラベルを中心に表示
    for i in range(10):
        mask = y_subset == i
        if mask.any():
            center = X_tsne[mask].mean(axis=0)
            ax.text(center[0], center[1], str(i), fontsize=12, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.colorbar(scatter, ax=axes, label='Digit')
plt.tight_layout()
plt.show()

# PCAとt-SNEの比較
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA
pca_comparison = PCA(n_components=2)
X_pca_comp = pca_comparison.fit_transform(X_subset)

axes[0].scatter(X_pca_comp[:, 0], X_pca_comp[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
axes[0].set_title('PCA Projection')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE
tsne_comparison = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_comp = tsne_comparison.fit_transform(X_subset)

axes[1].scatter(X_tsne_comp[:, 0], X_tsne_comp[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
axes[1].set_title('t-SNE Projection')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("PCAとt-SNEの特徴:")
print("- PCA: 線形変換、大域的構造を保持、高速")
print("- t-SNE: 非線形変換、局所的構造を保持、視覚化に優れる")
```

### サンプルコード5：t-SNEのパラメータ調整

```python
# t-SNEパラメータの影響を調査
from sklearn.metrics import pairwise_distances

# 小さなデータセットで実験
X_small = X_digits[:300]
y_small = y_digits[:300]

# 元の距離行列
original_distances = pairwise_distances(X_small)

# 異なるパラメータでt-SNE
params_grid = {
    'perplexity': [5, 15, 30, 50],
    'learning_rate': [10, 100, 200, 500]
}

# perplexityの影響
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, perp in enumerate(params_grid['perplexity']):
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate=200, 
                n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X_small)
    
    axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_small, 
                     cmap='tab10', alpha=0.6, s=30)
    axes[idx].set_title(f'Perplexity = {perp}')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')

plt.suptitle('Effect of Perplexity on t-SNE')
plt.tight_layout()
plt.show()

# 収束の監視
tsne_verbose = TSNE(n_components=2, perplexity=30, learning_rate=200,
                   n_iter=1000, verbose=1, random_state=42)
X_tsne_verbose = tsne_verbose.fit_transform(X_small)

# KL divergenceの推移（実際の実装では内部的に計算される）
print(f"\n最終的なKL divergence: {tsne_verbose.kl_divergence_:.4f}")
```

## 6.4 その他の次元削減手法

### サンプルコード6：UMAP、LLE、Isomap

```python
from sklearn.manifold import LocallyLinearEmbedding, Isomap
# UMAPは別途インストールが必要: pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAPを使用するには 'pip install umap-learn' を実行してください")

# Swiss Rollデータセットの作成
from sklearn.datasets import make_swiss_roll

n_samples = 1500
X_swiss, color = make_swiss_roll(n_samples, noise=0.05, random_state=42)

# 各手法での次元削減
methods = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2, perplexity=30, random_state=42),
    'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42),
    'Isomap': Isomap(n_components=2, n_neighbors=10)
}

if UMAP_AVAILABLE:
    methods['UMAP'] = umap.UMAP(n_components=2, random_state=42)

# 3D Swiss Rollの表示
fig = plt.figure(figsize=(18, 12))

# 元の3Dデータ
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original Swiss Roll")
ax.view_init(azim=-66, elev=12)

# 各手法の結果
for idx, (name, method) in enumerate(methods.items(), 2):
    ax = fig.add_subplot(2, 3, idx)
    
    try:
        X_transformed = method.fit_transform(X_swiss)
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, cmap=plt.cm.Spectral)
        ax.set_title(name)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    except Exception as e:
        ax.text(0.5, 0.5, f'{name}\nError: {str(e)[:30]}...', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name)

plt.tight_layout()
plt.show()

# 各手法の計算時間比較
import time

times = {}
for name, method in methods.items():
    start_time = time.time()
    try:
        method.fit_transform(X_swiss)
        elapsed = time.time() - start_time
        times[name] = elapsed
    except:
        times[name] = np.nan

print("\n計算時間の比較:")
for name, elapsed in times.items():
    if not np.isnan(elapsed):
        print(f"{name}: {elapsed:.3f}秒")
```

### サンプルコード7：特徴選択による次元削減

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 高次元データの生成
from sklearn.datasets import make_classification

X_high, y_high = make_classification(
    n_samples=500,
    n_features=100,
    n_informative=20,
    n_redundant=30,
    n_repeated=10,
    n_clusters_per_class=2,
    random_state=42
)

print(f"元のデータ形状: {X_high.shape}")

# 1. 分散による特徴選択
from sklearn.feature_selection import VarianceThreshold

var_selector = VarianceThreshold(threshold=0.1)
X_var = var_selector.fit_transform(X_high)
print(f"分散による選択後: {X_var.shape}")

# 2. 単変量統計による特徴選択
k_best = SelectKBest(f_classif, k=20)
X_kbest = k_best.fit_transform(X_high, y_high)
print(f"SelectKBest後: {X_kbest.shape}")

# 3. 相互情報量による特徴選択
mi_selector = SelectKBest(mutual_info_classif, k=20)
X_mi = mi_selector.fit_transform(X_high, y_high)
print(f"相互情報量による選択後: {X_mi.shape}")

# 4. 再帰的特徴除去（RFE）
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(rf, n_features_to_select=20)
X_rfe = rfe.fit_transform(X_high, y_high)
print(f"RFE後: {X_rfe.shape}")

# 選択された特徴の可視化
feature_scores = {
    'F-statistic': k_best.scores_,
    'Mutual Information': mi_selector.scores_,
    'RFE Ranking': -rfe.ranking_  # 負の値にして高いほど重要に
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (method, scores) in enumerate(feature_scores.items()):
    ax = axes[idx]
    
    # 上位20個の特徴をハイライト
    top_features = np.argsort(scores)[-20:]
    
    ax.bar(range(len(scores)), scores, color='lightblue')
    ax.bar(top_features, scores[top_features], color='darkblue')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Score')
    ax.set_title(f'{method}')
    ax.set_xlim(-1, len(scores))

plt.tight_layout()
plt.show()

# 次元削減手法の性能比較（分類タスク）
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

methods_comparison = {
    'Original': X_high,
    'PCA (20)': PCA(n_components=20).fit_transform(X_high),
    'SelectKBest': X_kbest,
    'RFE': X_rfe
}

results = []
for name, X_transformed in methods_comparison.items():
    # SVMで分類
    svm = SVC(kernel='rbf', random_state=42)
    scores = cross_val_score(svm, X_transformed, y_high, cv=5)
    
    results.append({
        'Method': name,
        'Features': X_transformed.shape[1],
        'Mean CV Score': scores.mean(),
        'Std CV Score': scores.std()
    })

results_df = pd.DataFrame(results)
print("\n次元削減手法の性能比較:")
print(results_df)
```

### サンプルコード8：オートエンコーダによる次元削減

```python
# 簡単なオートエンコーダの実装（scikit-learnのみ使用）
from sklearn.neural_network import MLPRegressor

class SimpleAutoencoder:
    def __init__(self, encoding_dim, hidden_layer_sizes=(100,), random_state=42):
        self.encoding_dim = encoding_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes + (encoding_dim,),
            activation='relu',
            solver='adam',
            random_state=random_state,
            max_iter=1000
        )
        
    def fit(self, X):
        # オートエンコーダは入力を出力として学習
        self.encoder.fit(X, X)
        return self
    
    def transform(self, X):
        # エンコーダ部分の出力を取得（簡略化のため最後の隠れ層の活性を使用）
        # 実際のオートエンコーダではエンコーダ部分のみを抽出
        _ = self.encoder.predict(X)  # フォワードパスを実行
        
        # 隠れ層の活性化を取得（これは簡略化された実装）
        # 実際の実装では中間層の出力を適切に取得する必要がある
        # ここではPCAで代用
        pca = PCA(n_components=self.encoding_dim)
        return pca.fit_transform(X)

# MNISTデータでの実験
X_digits_norm = X_digits / 16.0  # 正規化

# 簡単なオートエンコーダ
ae = SimpleAutoencoder(encoding_dim=32, hidden_layer_sizes=(128, 64))
ae.fit(X_digits_norm[:1000])  # 一部のデータで学習

# PCAとの比較
pca_32 = PCA(n_components=32)
X_pca_32 = pca_32.fit_transform(X_digits_norm)

print(f"元の次元数: {X_digits.shape[1]}")
print(f"圧縮後の次元数: 32")
print(f"圧縮率: {32/X_digits.shape[1]:.2%}")

# 再構成誤差の比較は省略（実際のオートエンコーダ実装が必要）
```

## 練習問題

### 問題1：顔認識のための次元削減
1. 顔画像データセットを使用
2. PCA、LDA（線形判別分析）を実装
3. 異なる次元数での認識精度を比較
4. 最適な次元数を決定

### 問題2：テキストデータの可視化
1. ニュース記事のTF-IDFベクトルを生成
2. PCA、t-SNE、UMAPで2次元に削減
3. カテゴリごとに色分けして可視化
4. 各手法の特徴を比較

### 問題3：異常検知のための次元削減
1. 高次元の正常データと異常データを生成
2. PCAで次元削減し、再構成誤差を計算
3. 再構成誤差を使った異常検知を実装
4. ROC曲線で性能評価

## 解答

### 解答1：顔認識のための次元削減

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Olivetti顔データセットを使用
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data
y_faces = faces.target

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_faces, y_faces, test_size=0.3, random_state=42, stratify=y_faces
)

# 異なる次元数での性能比較
n_components_list = [10, 20, 30, 50, 75, 100, 150]
results = []

for n_comp in n_components_list:
    # PCA
    pca = PCA(n_components=n_comp, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # PCA + KNN
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    acc_pca = knn_pca.score(X_test_pca, y_test)
    
    # LDA（最大でn_classes-1次元）
    if n_comp < len(np.unique(y_faces)):
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        
        # LDA + KNN
        knn_lda = KNeighborsClassifier(n_neighbors=5)
        knn_lda.fit(X_train_lda, y_train)
        acc_lda = knn_lda.score(X_test_lda, y_test)
    else:
        acc_lda = np.nan
    
    results.append({
        'n_components': n_comp,
        'PCA_accuracy': acc_pca,
        'LDA_accuracy': acc_lda,
        'compression_ratio': n_comp / X_faces.shape[1]
    })

results_df = pd.DataFrame(results)

# 結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 精度 vs 次元数
ax = axes[0]
ax.plot(results_df['n_components'], results_df['PCA_accuracy'], 'bo-', label='PCA')
ax.plot(results_df['n_components'][~results_df['LDA_accuracy'].isna()], 
        results_df['LDA_accuracy'][~results_df['LDA_accuracy'].isna()], 'ro-', label='LDA')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Accuracy')
ax.set_title('Face Recognition Accuracy vs Dimensionality')
ax.legend()
ax.grid(True, alpha=0.3)

# 精度 vs 圧縮率
ax = axes[1]
ax.plot(results_df['compression_ratio'] * 100, results_df['PCA_accuracy'], 'bo-', label='PCA')
ax.plot(results_df['compression_ratio'][~results_df['LDA_accuracy'].isna()] * 100, 
        results_df['LDA_accuracy'][~results_df['LDA_accuracy'].isna()], 'ro-', label='LDA')
ax.set_xlabel('Compression Ratio (%)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Compression Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("結果のサマリー:")
print(results_df)

# 最適な次元数
best_pca_idx = results_df['PCA_accuracy'].idxmax()
best_lda_idx = results_df['LDA_accuracy'].idxmax()

print(f"\nPCA最適次元数: {results_df.loc[best_pca_idx, 'n_components']} " +
      f"(精度: {results_df.loc[best_pca_idx, 'PCA_accuracy']:.3f})")
print(f"LDA最適次元数: {results_df.loc[best_lda_idx, 'n_components']} " +
      f"(精度: {results_df.loc[best_lda_idx, 'LDA_accuracy']:.3f})")

# 最適な設定での混同行列
n_comp_optimal = results_df.loc[best_pca_idx, 'n_components']
pca_optimal = PCA(n_components=n_comp_optimal, whiten=True)
X_train_opt = pca_optimal.fit_transform(X_train)
X_test_opt = pca_optimal.transform(X_test)

knn_optimal = KNeighborsClassifier(n_neighbors=5)
knn_optimal.fit(X_train_opt, y_train)
y_pred = knn_optimal.predict(X_test_opt)

# 一部の予測結果を可視化
n_samples_show = 10
sample_indices = np.random.choice(len(X_test), n_samples_show, replace=False)

fig, axes = plt.subplots(2, n_samples_show, figsize=(20, 4))

for idx, sample_idx in enumerate(sample_indices):
    # テスト画像
    axes[0, idx].imshow(X_test[sample_idx].reshape(64, 64), cmap='gray')
    axes[0, idx].set_title(f'True: {y_test[sample_idx]}')
    axes[0, idx].axis('off')
    
    # 予測結果
    pred_label = y_pred[sample_idx]
    color = 'green' if pred_label == y_test[sample_idx] else 'red'
    axes[1, idx].text(0.5, 0.5, f'Pred: {pred_label}', 
                      ha='center', va='center', fontsize=20, color=color)
    axes[1, idx].axis('off')

plt.suptitle(f'Face Recognition Results (PCA with {n_comp_optimal} components)')
plt.tight_layout()
plt.show()
```

### 解答2：テキストデータの可視化

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# ニュースグループデータの取得（4カテゴリのみ）
categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                               remove=('headers', 'footers', 'quotes'),
                               random_state=42)

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(newsgroups.data)
X_tfidf_dense = X_tfidf.toarray()

print(f"文書数: {X_tfidf_dense.shape[0]}")
print(f"特徴量数: {X_tfidf_dense.shape[1]}")

# 各手法での次元削減
reduction_methods = {
    'PCA': PCA(n_components=2, random_state=42),
    't-SNE': TSNE(n_components=2, perplexity=30, random_state=42)
}

if UMAP_AVAILABLE:
    reduction_methods['UMAP'] = umap.UMAP(n_components=2, random_state=42)

# カテゴリごとの色設定
colors = ['red', 'blue', 'green', 'orange']
category_colors = dict(zip(categories, colors))

fig, axes = plt.subplots(1, len(reduction_methods), figsize=(18, 6))
if len(reduction_methods) == 1:
    axes = [axes]

for idx, (method_name, method) in enumerate(reduction_methods.items()):
    print(f"\n{method_name}を実行中...")
    
    # 次元削減
    X_reduced = method.fit_transform(X_tfidf_dense)
    
    ax = axes[idx]
    
    # カテゴリごとにプロット
    for cat_idx, category in enumerate(categories):
        mask = newsgroups.target == cat_idx
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                  c=colors[cat_idx], label=category.split('.')[-1], 
                  alpha=0.6, s=30)
    
    ax.set_title(f'{method_name} Visualization')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    
plt.tight_layout()
plt.show()

# 各手法の特徴分析
print("\n各手法の特徴:")
print("- PCA: カテゴリ間の重なりが大きい（線形変換の限界）")
print("- t-SNE: カテゴリがより明確に分離（局所構造を保持）")
if UMAP_AVAILABLE:
    print("- UMAP: t-SNEに似た分離だが、大域的構造もある程度保持")

# 重要な単語の抽出（PCA）
pca_text = PCA(n_components=2)
X_pca_text = pca_text.fit_transform(X_tfidf_dense)

# 各主成分に最も寄与する単語
feature_names = vectorizer.get_feature_names_out()
for i in range(2):
    # 上位10単語
    top_indices = np.argsort(np.abs(pca_text.components_[i]))[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    top_weights = pca_text.components_[i][top_indices]
    
    print(f"\nPC{i+1}の上位10単語:")
    for word, weight in zip(top_words, top_weights):
        print(f"  {word}: {weight:.3f}")
```

### 解答3：異常検知のための次元削減

```python
from sklearn.metrics import roc_curve, auc

# 高次元データの生成
np.random.seed(42)

# 正常データ（ガウス分布）
n_normal = 1000
n_features = 50
X_normal = np.random.multivariate_normal(
    mean=np.zeros(n_features),
    cov=np.eye(n_features),
    size=n_normal
)

# 異常データ（別の分布）
n_anomaly = 50
# 一部の特徴量に大きな値を持つ
X_anomaly = np.random.multivariate_normal(
    mean=np.zeros(n_features),
    cov=np.eye(n_features),
    size=n_anomaly
)
# 異常パターンを追加
anomaly_features = np.random.choice(n_features, 10, replace=False)
X_anomaly[:, anomaly_features] += np.random.normal(5, 1, (n_anomaly, 10))

# データの結合
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

# データのシャッフル
shuffle_idx = np.random.permutation(len(X_all))
X_all = X_all[shuffle_idx]
y_true = y_true[shuffle_idx]

print(f"データ形状: {X_all.shape}")
print(f"正常データ: {n_normal}, 異常データ: {n_anomaly}")

# PCAによる次元削減と再構成誤差
def pca_anomaly_detection(X, n_components):
    """PCAを使った異常検知"""
    pca = PCA(n_components=n_components)
    
    # 正常データのみで学習（実際には異常データが混ざっている場合もある）
    # ここでは簡単のため全データで学習
    pca.fit(X)
    
    # 次元削減と再構成
    X_reduced = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    # 再構成誤差
    reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    return reconstruction_errors, pca

# 異なる主成分数での実験
n_components_list = [5, 10, 20, 30, 40]
auc_scores = []

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, n_comp in enumerate(n_components_list):
    errors, pca_model = pca_anomaly_detection(X_all, n_comp)
    
    # ROC曲線
    fpr, tpr, thresholds = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    
    ax = axes[idx]
    ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'PCA with {n_comp} components')
    ax.legend()
    ax.grid(True, alpha=0.3)

# AUCスコアの比較
ax = axes[-1]
ax.plot(n_components_list, auc_scores, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Components')
ax.set_ylabel('AUC Score')
ax.set_title('AUC vs Number of Components')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最適な主成分数
best_n_comp = n_components_list[np.argmax(auc_scores)]
print(f"\n最適な主成分数: {best_n_comp} (AUC: {max(auc_scores):.3f})")

# 最適な設定での詳細分析
errors_best, pca_best = pca_anomaly_detection(X_all, best_n_comp)

# 再構成誤差の分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(errors_best[y_true == 0], bins=30, alpha=0.5, label='Normal', density=True)
plt.hist(errors_best[y_true == 1], bins=30, alpha=0.5, label='Anomaly', density=True)
plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.title('Distribution of Reconstruction Errors')
plt.legend()

plt.subplot(1, 2, 2)
# 最適な閾値の決定（Youden's J statistic）
fpr, tpr, thresholds = roc_curve(y_true, errors_best)
j_scores = tpr - fpr
best_threshold_idx = np.argmax(j_scores)
best_threshold = thresholds[best_threshold_idx]

plt.scatter(errors_best[y_true == 0], np.random.uniform(-0.1, 0.1, sum(y_true == 0)), 
           alpha=0.5, label='Normal', s=20)
plt.scatter(errors_best[y_true == 1], np.random.uniform(-0.1, 0.1, sum(y_true == 1)), 
           alpha=0.5, label='Anomaly', s=20)
plt.axvline(x=best_threshold, color='red', linestyle='--', 
           label=f'Threshold = {best_threshold:.2f}')
plt.xlabel('Reconstruction Error')
plt.ylabel('Jittered y')
plt.title('Anomaly Detection Threshold')
plt.legend()
plt.ylim(-0.2, 0.2)

plt.tight_layout()
plt.show()

# 性能評価
y_pred = (errors_best > best_threshold).astype(int)
from sklearn.metrics import classification_report

print("\n異常検知の性能:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
```

## まとめ

この章では、教師なし学習の次元削減について学習しました：

- **PCA（主成分分析）**: 線形変換による分散最大化、高速で解釈しやすい
- **t-SNE**: 非線形変換による局所構造の保持、可視化に優れる
- **その他の手法**: UMAP、LLE、Isomap、オートエンコーダ
- **特徴選択**: 次元削減の代替手法
- **応用例**: 顔認識、テキスト可視化、異常検知

次章では、モデルの評価と改善について詳しく学習します。

[目次に戻る](README.md)