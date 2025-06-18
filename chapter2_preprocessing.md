# 第2章：データの前処理

## 2.1 なぜ前処理が重要か

機械学習において、データの前処理は成功の鍵となります。実世界のデータは以下のような問題を抱えていることが多いです：

- **スケールの違い**: 特徴量間で値の範囲が大きく異なる
- **欠損値**: データの一部が欠けている
- **カテゴリカルデータ**: 数値でない文字列データ
- **外れ値**: 極端に大きい/小さい値
- **データの偏り**: 不均衡なクラス分布

## 2.2 データのスケーリング

### サンプルコード1：様々なスケーリング手法

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.datasets import load_wine

# データの読み込み
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

# 最初の4つの特徴量を使用
X_subset = X[:, :4]
feature_subset = feature_names[:4]

# 元のデータの統計量を表示
print("元のデータの統計量:")
df_original = pd.DataFrame(X_subset, columns=feature_subset)
print(df_original.describe())
print()

# 様々なスケーラーを適用
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer()
}

# 各スケーラーの結果を可視化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# 元のデータ
axes[0].boxplot(X_subset)
axes[0].set_title('Original Data')
axes[0].set_xticklabels(feature_subset, rotation=45)

# 各スケーラーを適用
for idx, (name, scaler) in enumerate(scalers.items(), 1):
    X_scaled = scaler.fit_transform(X_subset)
    axes[idx].boxplot(X_scaled)
    axes[idx].set_title(name)
    axes[idx].set_xticklabels(feature_subset, rotation=45)

plt.tight_layout()
plt.show()
```

### サンプルコード2：スケーリングの効果の比較

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# データの準備
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# スケーリングなしとありで比較
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', gamma='scale')
}

results = {}

for model_name, model in models.items():
    # スケーリングなし
    model_no_scale = model.__class__(**model.get_params())
    model_no_scale.fit(X_train, y_train)
    y_pred_no_scale = model_no_scale.predict(X_test)
    acc_no_scale = accuracy_score(y_test, y_pred_no_scale)
    
    # StandardScalerでスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_scaled = model.__class__(**model.get_params())
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)
    
    results[model_name] = {
        'no_scaling': acc_no_scale,
        'with_scaling': acc_scaled
    }

# 結果の表示
print("スケーリングの効果:")
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    print(f"  スケーリングなし: {scores['no_scaling']:.3f}")
    print(f"  スケーリングあり: {scores['with_scaling']:.3f}")
    print(f"  改善率: {(scores['with_scaling'] - scores['no_scaling']) * 100:.1f}%")
```

## 2.3 カテゴリカルデータの処理

### サンプルコード3：カテゴリカル変数のエンコーディング

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# サンプルデータの作成
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'price': [100, 150, 200, 250, 180]
})

print("元のデータ:")
print(data)
print()

# 1. Label Encoding
le = LabelEncoder()
data['color_label'] = le.fit_transform(data['color'])
print("Label Encoding (color):")
print(data[['color', 'color_label']])
print()

# 2. One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
color_encoded = ohe.fit_transform(data[['color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['color']))
print("One-Hot Encoding (color):")
print(pd.concat([data['color'], color_df], axis=1))
print()

# 3. Ordinal Encoding (順序がある場合)
size_order = ['S', 'M', 'L', 'XL']
oe = OrdinalEncoder(categories=[size_order])
data['size_ordinal'] = oe.fit_transform(data[['size']])
print("Ordinal Encoding (size):")
print(data[['size', 'size_ordinal']])
```

### サンプルコード4：pd.get_dummiesを使った実践的な例

```python
# より複雑なデータセットの作成
import numpy as np

np.random.seed(42)
n_samples = 1000

customer_data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'purchase': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

print("データの概要:")
print(customer_data.head())
print(f"\nデータ型:")
print(customer_data.dtypes)

# カテゴリカル変数をダミー変数に変換
customer_encoded = pd.get_dummies(customer_data, columns=['education', 'region'], drop_first=True)

print(f"\nエンコード後の列:")
print(customer_encoded.columns.tolist())

# 機械学習モデルでの使用例
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = customer_encoded.drop('purchase', axis=1)
y = customer_encoded['purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print(f"\nモデルの精度: {accuracy:.3f}")

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量の重要度:")
print(feature_importance)
```

## 2.4 欠損値の処理

### サンプルコード5：様々な欠損値処理手法

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 欠損値を含むデータの作成
np.random.seed(42)
n_samples = 100
n_features = 5

# 完全なデータを生成
X_complete = np.random.randn(n_samples, n_features)

# ランダムに欠損値を追加（20%の確率）
missing_mask = np.random.rand(n_samples, n_features) < 0.2
X_missing = X_complete.copy()
X_missing[missing_mask] = np.nan

# データフレームに変換
df_missing = pd.DataFrame(X_missing, columns=[f'feature_{i}' for i in range(n_features)])

print("欠損値の統計:")
print(df_missing.isnull().sum())
print(f"\n合計欠損値数: {df_missing.isnull().sum().sum()}")

# 1. 平均値で補完
imputer_mean = SimpleImputer(strategy='mean')
X_imputed_mean = imputer_mean.fit_transform(X_missing)

# 2. 中央値で補完
imputer_median = SimpleImputer(strategy='median')
X_imputed_median = imputer_median.fit_transform(X_missing)

# 3. 最頻値で補完（カテゴリカルデータ向け）
imputer_mode = SimpleImputer(strategy='most_frequent')
X_imputed_mode = imputer_mode.fit_transform(X_missing)

# 4. KNN Imputer
imputer_knn = KNNImputer(n_neighbors=5)
X_imputed_knn = imputer_knn.fit_transform(X_missing)

# 5. Iterative Imputer (MICE)
imputer_iter = IterativeImputer(random_state=42)
X_imputed_iter = imputer_iter.fit_transform(X_missing)

# 補完精度の比較（元の完全なデータとの差）
methods = {
    'Mean': X_imputed_mean,
    'Median': X_imputed_median,
    'Mode': X_imputed_mode,
    'KNN': X_imputed_knn,
    'Iterative': X_imputed_iter
}

print("\n補完精度の比較 (RMSE):")
for name, X_imputed in methods.items():
    rmse = np.sqrt(np.mean((X_complete[missing_mask] - X_imputed[missing_mask])**2))
    print(f"{name}: {rmse:.4f}")
```

## 2.5 特徴量エンジニアリング

### サンプルコード6：多項式特徴量の生成

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# シンプルな非線形データの生成
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = 2 * X.ravel() + X.ravel()**2 + np.random.randn(100) * 10

# 多項式特徴量の生成
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"元の特徴量数: {X.shape[1]}")
print(f"多項式特徴量数: {X_poly.shape[1]}")
print(f"生成された特徴量名: {poly_features.get_feature_names_out()}")

# 線形回帰で比較
from sklearn.linear_model import LinearRegression

# 元の特徴量での回帰
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# 多項式特徴量での回帰
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
y_pred_poly = lr_poly.predict(X_poly)

# 可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Features')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_pred_poly, color='green', linewidth=2, label='Polynomial')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Features')
plt.legend()

plt.tight_layout()
plt.show()

# R²スコアの比較
from sklearn.metrics import r2_score
print(f"\nR² スコア:")
print(f"線形回帰: {r2_score(y, y_pred):.3f}")
print(f"多項式回帰: {r2_score(y, y_pred_poly):.3f}")
```

### サンプルコード7：カスタム特徴量の作成

```python
from sklearn.base import BaseEstimator, TransformerMixin

# カスタムトランスフォーマーの作成
class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # 新しい特徴量を作成
        X_new = X.copy()
        
        # 比率特徴量
        if X.shape[1] >= 2:
            X_new = np.column_stack([X_new, X[:, 0] / (X[:, 1] + 1e-8)])
        
        # 対数変換
        X_new = np.column_stack([X_new, np.log1p(np.abs(X[:, 0]))])
        
        # 二乗
        X_new = np.column_stack([X_new, X[:, 0] ** 2])
        
        return X_new

# 使用例
X_sample = np.random.rand(100, 3) * 100
custom_transformer = CustomFeatureEngineering()
X_transformed = custom_transformer.fit_transform(X_sample)

print(f"元の特徴量数: {X_sample.shape[1]}")
print(f"変換後の特徴量数: {X_transformed.shape[1]}")
print(f"\n最初の5行:")
print(X_transformed[:5])
```

## 2.6 データの分割と層化

### サンプルコード8：層化抽出と時系列分割

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.datasets import make_classification

# 不均衡なデータセットの作成
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                         weights=[0.7, 0.2, 0.1], random_state=42)

# クラス分布の確認
unique, counts = np.unique(y, return_counts=True)
print("元のクラス分布:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} ({count/len(y)*100:.1f}%)")

# 1. 通常の分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\n通常の分割後のテストセットのクラス分布:")
unique, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} ({count/len(y_test)*100:.1f}%)")

# 2. 層化抽出
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print("\n層化抽出後のテストセットのクラス分布:")
unique, counts = np.unique(y_test_strat, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} ({count/len(y_test_strat)*100:.1f}%)")

# 3. 時系列データの分割
# 時系列データの模擬
time_series_data = np.arange(100).reshape(-1, 1)
time_series_target = np.sin(time_series_data.ravel() * 0.1) + np.random.randn(100) * 0.1

tscv = TimeSeriesSplit(n_splits=5)
print("\n時系列交差検証の分割:")
for i, (train_idx, test_idx) in enumerate(tscv.split(time_series_data)):
    print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    print(f"  Train range: {train_idx[0]}-{train_idx[-1]}")
    print(f"  Test range: {test_idx[0]}-{test_idx[-1]}")
```

## 練習問題

### 問題1：総合的な前処理パイプライン
以下の要件を満たす前処理パイプラインを作成してください：
1. 数値特徴量とカテゴリカル特徴量を含むデータセットを作成
2. 数値特徴量は標準化、カテゴリカル特徴量はOne-Hot Encoding
3. 欠損値は適切に処理
4. ColumnTransformerを使用して実装

### 問題2：外れ値の検出と処理
1. 外れ値を含むデータセットを生成
2. IQR法とIsolation Forestで外れ値を検出
3. 外れ値の処理前後でモデルの性能を比較

### 問題3：特徴量選択
1. 高次元データセット（特徴量100個以上）を生成
2. 相関係数、相互情報量、Recursive Feature Eliminationで特徴量選択
3. 選択された特徴量でモデルの性能を比較

## 解答

### 解答1：総合的な前処理パイプライン

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データセットの作成
np.random.seed(42)
n_samples = 1000

# 数値特徴量（一部に欠損値を含む）
age = np.random.randint(18, 80, n_samples).astype(float)
age[np.random.rand(n_samples) < 0.1] = np.nan

income = np.random.normal(50000, 20000, n_samples)
income[np.random.rand(n_samples) < 0.05] = np.nan

# カテゴリカル特徴量
education = np.random.choice(['HS', 'BS', 'MS', 'PhD'], n_samples)
city = np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples)

# ターゲット変数
target = np.random.choice([0, 1], n_samples)

# データフレームの作成
df = pd.DataFrame({
    'age': age,
    'income': income,
    'education': education,
    'city': city,
    'target': target
})

print("データの概要:")
print(df.info())
print("\n欠損値:")
print(df.isnull().sum())

# 特徴量とターゲットの分離
X = df.drop('target', axis=1)
y = df['target']

# 数値とカテゴリカルの列を特定
numeric_features = ['age', 'income']
categorical_features = ['education', 'city']

# 前処理パイプラインの作成
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformerで結合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 完全なパイプライン
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# モデルの訓練と評価
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nモデルの精度: {accuracy:.3f}")

# 前処理後のデータ形状を確認
X_transformed = preprocessor.fit_transform(X_train)
print(f"\n前処理後のデータ形状: {X_transformed.shape}")
```

### 解答2：外れ値の検出と処理

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 外れ値を含むデータセットの生成
np.random.seed(42)
n_samples = 300
n_outliers = 30

# 正常なデータ
X_normal = np.random.randn(n_samples, 2)
X_normal[:, 0] = X_normal[:, 0] * 2 + 5
X_normal[:, 1] = X_normal[:, 1] * 3 + 10

# 外れ値
X_outliers = np.random.uniform(low=-10, high=20, size=(n_outliers, 2))

# データの結合
X = np.vstack([X_normal, X_outliers])
y = np.hstack([np.ones(n_samples), np.zeros(n_outliers)])

# データの可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Normal', alpha=0.7)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Outlier', alpha=0.7, color='red')
plt.title('Original Data')
plt.legend()

# 1. IQR法による外れ値検出
def detect_outliers_iqr(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
    return outlier_mask

outliers_iqr = detect_outliers_iqr(X)

plt.subplot(1, 3, 2)
plt.scatter(X[~outliers_iqr, 0], X[~outliers_iqr, 1], label='Normal', alpha=0.7)
plt.scatter(X[outliers_iqr, 0], X[outliers_iqr, 1], label='Outlier (IQR)', alpha=0.7, color='orange')
plt.title('IQR Method')
plt.legend()

# 2. Isolation Forestによる外れ値検出
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(X) == -1

plt.subplot(1, 3, 3)
plt.scatter(X[~outliers_iso, 0], X[~outliers_iso, 1], label='Normal', alpha=0.7)
plt.scatter(X[outliers_iso, 0], X[outliers_iso, 1], label='Outlier (IF)', alpha=0.7, color='green')
plt.title('Isolation Forest')
plt.legend()

plt.tight_layout()
plt.show()

# 性能の比較
from sklearn.metrics import classification_report

print("外れ値検出の性能:")
print("\nIQR法:")
print(classification_report(y==0, outliers_iqr))
print("\nIsolation Forest:")
print(classification_report(y==0, outliers_iso))

# モデル性能への影響
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ダミーの分類タスク用データを作成
y_class = np.random.choice([0, 1], size=len(X))

# 全データでの性能
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
rf_all = RandomForestClassifier(random_state=42)
rf_all.fit(X_train, y_train)
score_all = rf_all.score(X_test, y_test)

# 外れ値除去後の性能
X_clean = X[~outliers_iso]
y_clean = y_class[~outliers_iso]
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42
)
rf_clean = RandomForestClassifier(random_state=42)
rf_clean.fit(X_train_clean, y_train_clean)
score_clean = rf_clean.score(X_test_clean, y_test_clean)

print(f"\nモデル性能の比較:")
print(f"全データ使用: {score_all:.3f}")
print(f"外れ値除去後: {score_clean:.3f}")
```

### 解答3：特徴量選択

```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 高次元データセットの生成
X, y = make_classification(
    n_samples=500,
    n_features=100,
    n_informative=20,
    n_redundant=30,
    n_repeated=10,
    n_classes=2,
    random_state=42
)

print(f"データセットの形状: {X.shape}")
print(f"有益な特徴量: 20個")
print(f"冗長な特徴量: 30個")
print(f"繰り返し特徴量: 10個")
print(f"ノイズ特徴量: 40個")

# ベースラインモデルの性能
rf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_scores = cross_val_score(rf, X, y, cv=5)
print(f"\nベースライン性能 (全特徴量): {baseline_scores.mean():.3f} (+/- {baseline_scores.std():.3f})")

# 1. 相関係数による特徴量選択
selector_corr = SelectKBest(score_func=f_classif, k=20)
X_corr = selector_corr.fit_transform(X, y)
scores_corr = cross_val_score(rf, X_corr, y, cv=5)
print(f"\n相関係数 (上位20特徴量): {scores_corr.mean():.3f} (+/- {scores_corr.std():.3f})")

# 2. 相互情報量による特徴量選択
selector_mi = SelectKBest(score_func=mutual_info_classif, k=20)
X_mi = selector_mi.fit_transform(X, y)
scores_mi = cross_val_score(rf, X_mi, y, cv=5)
print(f"相互情報量 (上位20特徴量): {scores_mi.mean():.3f} (+/- {scores_mi.std():.3f})")

# 3. Recursive Feature Elimination (RFE)
rfe = RFE(estimator=rf, n_features_to_select=20)
X_rfe = rfe.fit_transform(X, y)
scores_rfe = cross_val_score(rf, X_rfe, y, cv=5)
print(f"RFE (上位20特徴量): {scores_rfe.mean():.3f} (+/- {scores_rfe.std():.3f})")

# 特徴量数と性能の関係を可視化
k_features = [5, 10, 20, 30, 40, 50, 75, 100]
scores_by_k = []

for k in k_features:
    if k <= X.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        scores = cross_val_score(rf, X_selected, y, cv=5)
        scores_by_k.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_features[:len(scores_by_k)], scores_by_k, 'o-')
plt.xlabel('Number of Features')
plt.ylabel('Cross-validation Score')
plt.title('Performance vs Number of Features')
plt.grid(True, alpha=0.3)
plt.show()

# 選択された特徴量の重要度
rf.fit(X, y)
feature_importance = rf.feature_importances_

# 各手法で選択された特徴量のインデックス
selected_features_corr = selector_corr.get_support(indices=True)
selected_features_mi = selector_mi.get_support(indices=True)
selected_features_rfe = rfe.get_support(indices=True)

# 選択された特徴量の平均重要度
print(f"\n選択された特徴量の平均重要度:")
print(f"相関係数: {feature_importance[selected_features_corr].mean():.4f}")
print(f"相互情報量: {feature_importance[selected_features_mi].mean():.4f}")
print(f"RFE: {feature_importance[selected_features_rfe].mean():.4f}")
```

## まとめ

この章では、機械学習における重要な前処理技術を学びました：

- データのスケーリング（StandardScaler、MinMaxScaler、RobustScaler）
- カテゴリカルデータの処理（LabelEncoder、OneHotEncoder）
- 欠損値の処理（SimpleImputer、KNNImputer、IterativeImputer）
- 特徴量エンジニアリング（多項式特徴量、カスタム変換）
- 外れ値の検出と処理
- 特徴量選択

次章では、教師あり学習の分類問題について詳しく学習します。