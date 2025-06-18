# 第1章：scikit-learn入門

## 1.1 scikit-learnとは

scikit-learn（サイキット・ラーン）は、Pythonで書かれたオープンソースの機械学習ライブラリです。シンプルで効率的なツールを提供し、データマイニングとデータ分析のための実用的なライブラリとして広く使用されています。

### 主な特徴

- **統一されたAPI**: 一貫性のあるインターフェースで様々なアルゴリズムを使用可能
- **豊富なアルゴリズム**: 分類、回帰、クラスタリング、次元削減など多様な手法を実装
- **優れたドキュメント**: 詳細な説明と豊富な例題
- **高速な実装**: NumPyとSciPyを基盤とした効率的な計算

## 1.2 基本的な使い方

scikit-learnの基本的なワークフローは以下の通りです：

1. データの準備
2. モデルの選択
3. モデルの学習（fit）
4. 予測（predict）
5. 評価

### サンプルコード1：最初の分類モデル

```python
# 必要なライブラリのインポート
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# データセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データの分割（訓練用70%、テスト用30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# モデルの作成と学習
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 予測
y_pred = knn.predict(X_test)

# 精度の評価
accuracy = accuracy_score(y_test, y_pred)
print(f"精度: {accuracy:.2f}")
```

## 1.3 データセットの読み込み

scikit-learnには練習用のデータセットが含まれています。

### サンプルコード2：様々なデータセット

```python
from sklearn.datasets import load_iris, load_digits, load_wine, make_classification
import matplotlib.pyplot as plt

# 1. Irisデータセット（アヤメの分類）
iris = load_iris()
print("Iris dataset:")
print(f"データの形状: {iris.data.shape}")
print(f"クラス数: {len(iris.target_names)}")
print(f"特徴量名: {iris.feature_names}")
print()

# 2. Digitsデータセット（手書き数字）
digits = load_digits()
print("Digits dataset:")
print(f"データの形状: {digits.data.shape}")
print(f"画像サイズ: 8x8")

# 手書き数字の表示
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, label in zip(axes.flat, digits.images, digits.target):
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 3. 人工データの生成
X, y = make_classification(
    n_samples=100,      # サンプル数
    n_features=2,       # 特徴量の数
    n_redundant=0,      # 冗長な特徴量の数
    n_informative=2,    # 情報を持つ特徴量の数
    n_clusters_per_class=1,
    random_state=42
)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('人工的に生成した分類データ')
plt.colorbar()
plt.show()
```

## 1.4 最初の機械学習モデル

### サンプルコード3：複数のアルゴリズムの比較

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# データの準備
wine = load_wine()
X = wine.data
y = wine.target

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# データの標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 複数のモデルを比較
models = {
    'ロジスティック回帰': LogisticRegression(random_state=42),
    '決定木': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    # モデルの学習
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    
    # 精度の計算
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}の結果:")
    print(f"精度: {accuracy:.3f}")
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

### サンプルコード4：学習曲線の可視化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

# 学習曲線を描画する関数
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.grid(True)
    
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.show()

# SVMモデルの学習曲線
svm = SVC(kernel='rbf', gamma=0.001)
plot_learning_curve(svm, X_train_scaled, y_train, "SVM Learning Curve")
```

## 1.5 モデルの保存と読み込み

### サンプルコード5：モデルの永続化

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# モデルの学習
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# モデルの保存
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("モデルを保存しました")

# モデルの読み込み
loaded_model = joblib.load('random_forest_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# 読み込んだモデルで予測
X_test_scaled_loaded = loaded_scaler.transform(X_test)
y_pred_loaded = loaded_model.predict(X_test)

# 結果の確認
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"読み込んだモデルの精度: {accuracy_loaded:.3f}")
```

## 練習問題

### 問題1：基本的な分類
ワインデータセットを使用して、以下の要件を満たすプログラムを作成してください：
1. データを訓練用80%、テスト用20%に分割
2. k近傍法（k=5）で学習
3. テストデータの精度を計算
4. 混同行列を表示

### 問題2：特徴量の重要度
1. ランダムフォレストを使用してアヤメデータセットを分類
2. 各特徴量の重要度を計算し、棒グラフで表示
3. 最も重要な2つの特徴量だけを使って再度学習し、精度を比較

### 問題3：交差検証
1. 手書き数字データセット（load_digits）を使用
2. SVMモデルで5分割交差検証を実行
3. 各分割の精度と平均精度を表示

## 解答

### 解答1：基本的な分類

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
wine = load_wine()
X = wine.data
y = wine.target

# データの分割（訓練用80%、テスト用20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# k近傍法（k=5）で学習
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 予測と精度計算
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"テストデータの精度: {accuracy:.3f}")

# 混同行列の表示
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, 
            yticklabels=wine.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
```

### 解答2：特徴量の重要度

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ランダムフォレストで学習
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 全特徴量での精度
accuracy_all = rf.score(X_test, y_test)
print(f"全特徴量を使用した精度: {accuracy_all:.3f}")

# 特徴量の重要度を取得
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 特徴量の重要度を表示
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# 最も重要な2つの特徴量だけを使用
top_2_features = indices[:2]
X_train_top2 = X_train[:, top_2_features]
X_test_top2 = X_test[:, top_2_features]

# 再度学習
rf_top2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top2.fit(X_train_top2, y_train)

# 精度の比較
accuracy_top2 = rf_top2.score(X_test_top2, y_test)
print(f"上位2特徴量のみの精度: {accuracy_top2:.3f}")
print(f"精度の差: {accuracy_all - accuracy_top2:.3f}")
```

### 解答3：交差検証

```python
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# データの読み込み
digits = load_digits()
X = digits.data
y = digits.target

# パイプラインの作成（スケーリング + SVM）
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', gamma=0.001))
])

# 5分割交差検証
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# 結果の表示
print("5分割交差検証の結果:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.3f}")

print(f"\n平均精度: {cv_scores.mean():.3f}")
print(f"標準偏差: {cv_scores.std():.3f}")

# 箱ひげ図で可視化
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation Results')
plt.ylim(0.9, 1.0)
plt.grid(True, alpha=0.3)
plt.show()
```

## まとめ

この章では、scikit-learnの基本的な使い方を学びました：

- scikit-learnの特徴と基本的なワークフロー
- 組み込みデータセットの使用方法
- 複数のアルゴリズムの比較方法
- モデルの保存と読み込み
- 学習曲線による性能の可視化

次章では、機械学習において重要なデータの前処理について詳しく学習します。

[目次に戻る](README.md)