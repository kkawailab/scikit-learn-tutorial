# 第4章：教師あり学習 - 回帰

## 4.1 回帰問題の基礎

回帰問題は、連続値を予測する機械学習タスクです。分類が離散的なクラスを予測するのに対し、回帰は数値を予測します。

### 回帰問題の例
- 住宅価格の予測
- 売上予測
- 気温予測
- 株価予測

## 4.2 線形回帰

### サンプルコード1：単回帰と重回帰

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression

# 1. 単回帰の例
np.random.seed(42)
X_simple = 2 * np.random.rand(100, 1)
y_simple = 4 + 3 * X_simple + np.random.randn(100, 1)

# モデルの学習
lr_simple = LinearRegression()
lr_simple.fit(X_simple, y_simple)

# 予測
y_pred_simple = lr_simple.predict(X_simple)

# 可視化
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, alpha=0.5, label='Data points')
plt.plot(X_simple, y_pred_simple, 'r-', linewidth=2, label='Linear regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Simple Linear Regression\ny = {lr_simple.intercept_[0]:.2f} + {lr_simple.coef_[0][0]:.2f}x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"切片: {lr_simple.intercept_[0]:.2f}")
print(f"係数: {lr_simple.coef_[0][0]:.2f}")
print(f"R²スコア: {r2_score(y_simple, y_pred_simple):.3f}")

# 2. 重回帰の例
X_multi, y_multi = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']

# データフレームに変換
df = pd.DataFrame(X_multi, columns=feature_names)
df['target'] = y_multi

# 相関行列の可視化
import seaborn as sns

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)

# 重回帰モデル
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)

# 予測
y_pred_train = lr_multi.predict(X_train)
y_pred_test = lr_multi.predict(X_test)

# 係数の可視化
plt.figure(figsize=(10, 6))
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_multi.coef_
})
plt.bar(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Linear Regression Coefficients')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

print("\n重回帰モデルの評価:")
print(f"訓練データ R²: {r2_score(y_train, y_pred_train):.3f}")
print(f"テストデータ R²: {r2_score(y_test, y_pred_test):.3f}")
print(f"平均二乗誤差 (MSE): {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"平均絶対誤差 (MAE): {mean_absolute_error(y_test, y_pred_test):.2f}")
```

### サンプルコード2：残差分析

```python
# 残差の計算
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 予測値 vs 実際の値
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs Actual')
ax.grid(True, alpha=0.3)

# 2. 残差プロット
ax = axes[0, 1]
ax.scatter(y_pred_test, residuals_test, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# 3. 残差のヒストグラム
ax = axes[1, 0]
ax.hist(residuals_test, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Residuals')
ax.grid(True, alpha=0.3)

# 4. Q-Qプロット
from scipy import stats
ax = axes[1, 1]
stats.probplot(residuals_test, dist="norm", plot=ax)
ax.set_title('Q-Q Plot')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 残差の統計的検定
from scipy.stats import normaltest
statistic, p_value = normaltest(residuals_test)
print(f"\n残差の正規性検定:")
print(f"統計量: {statistic:.4f}")
print(f"p値: {p_value:.4f}")
print(f"結論: {'残差は正規分布に従う' if p_value > 0.05 else '残差は正規分布に従わない'}")
```

## 4.3 多項式回帰

### サンプルコード3：多項式回帰と過学習

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 非線形データの生成
np.random.seed(42)
X_nonlinear = np.sort(np.random.rand(100, 1) * 6, axis=0)
y_nonlinear = np.sin(X_nonlinear).ravel() + np.random.normal(0, 0.1, X_nonlinear.shape[0])

# 異なる次数の多項式回帰
degrees = [1, 3, 6, 9]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees, 1):
    plt.subplot(2, 2, i)
    
    # 多項式特徴量の生成と回帰
    poly_model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    poly_model.fit(X_nonlinear, y_nonlinear)
    
    # 予測用のデータ
    X_plot = np.linspace(0, 6, 300).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    # プロット
    plt.scatter(X_nonlinear, y_nonlinear, alpha=0.5, label='Data')
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {degree}')
    plt.plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, label='True function')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # スコアの計算
    train_score = poly_model.score(X_nonlinear, y_nonlinear)
    print(f"Degree {degree} - Training R²: {train_score:.3f}")

plt.tight_layout()
plt.show()

# 学習曲線で過学習を確認
from sklearn.model_selection import learning_curve

degrees_to_compare = [1, 3, 6, 9, 15]
plt.figure(figsize=(12, 8))

for degree in degrees_to_compare:
    poly_model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    
    train_sizes, train_scores, val_scores = learning_curve(
        poly_model, X_nonlinear, y_nonlinear,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2'
    )
    
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', 
             label=f'Degree {degree}', alpha=0.7)

plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curves for Different Polynomial Degrees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 4.4 正則化（Ridge、Lasso、Elastic Net）

### サンプルコード4：正則化手法の比較

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# 高次元データの生成（特徴量が多い）
X_high_dim, y_high_dim = make_regression(
    n_samples=100, n_features=20, n_informative=10,
    noise=5, random_state=42
)

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_high_dim)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_high_dim, test_size=0.3, random_state=42
)

# 異なるアルファ値での比較
alphas = np.logspace(-4, 4, 50)
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(max_iter=10000),
    'ElasticNet': ElasticNet(max_iter=10000)
}

# 各モデルの係数の変化を追跡
plt.figure(figsize=(15, 10))

for idx, (name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 2, idx)
    
    if name == 'Linear':
        # 線形回帰は正則化なし
        model.fit(X_train, y_train)
        coefs = model.coef_
        plt.hlines(coefs, 0, 1, alpha=0.7)
        plt.xlim(0, 1)
    else:
        # 正則化モデル
        coefs = []
        for alpha in alphas:
            if name == 'Ridge':
                model = Ridge(alpha=alpha)
            elif name == 'Lasso':
                model = Lasso(alpha=alpha, max_iter=10000)
            else:
                model = ElasticNet(alpha=alpha, max_iter=10000)
            
            model.fit(X_train, y_train)
            coefs.append(model.coef_)
        
        coefs = np.array(coefs)
        for feature_idx in range(X_train.shape[1]):
            plt.plot(np.log10(alphas), coefs[:, feature_idx], alpha=0.7)
        
        plt.xlabel('log(alpha)')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.ylabel('Coefficients')
    plt.title(f'{name} Regression Coefficients')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最適なアルファ値の選択（交差検証）
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# 交差検証付きモデル
ridge_cv = RidgeCV(alphas=alphas, cv=5)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
elastic_cv = ElasticNetCV(alphas=alphas, cv=5, max_iter=10000)

models_cv = {
    'Linear Regression': LinearRegression(),
    'Ridge (CV)': ridge_cv,
    'Lasso (CV)': lasso_cv,
    'ElasticNet (CV)': elastic_cv
}

results = []

for name, model in models_cv.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    result = {
        'Model': name,
        'Train R²': model.score(X_train, y_train),
        'Test R²': model.score(X_test, y_test),
        'MSE': mean_squared_error(y_test, y_pred),
        'Non-zero Coefs': np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else 'N/A'
    }
    
    if hasattr(model, 'alpha_'):
        result['Best Alpha'] = model.alpha_
    
    results.append(result)

results_df = pd.DataFrame(results)
print("正則化モデルの比較:")
print(results_df.round(3))

# 特徴量選択の効果（Lasso）
selected_features = np.where(lasso_cv.coef_ != 0)[0]
print(f"\nLassoで選択された特徴量: {len(selected_features)}/{X_train.shape[1]}")
print(f"選択された特徴量のインデックス: {selected_features}")
```

## 4.5 その他の回帰手法

### サンプルコード5：ランダムフォレスト回帰とGradient Boosting

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 非線形データの生成
X_complex, y_complex = make_regression(
    n_samples=500, n_features=10, n_informative=5,
    noise=10, random_state=42
)

# データの分割と標準化
X_train, X_test, y_train, y_test = train_test_split(
    X_complex, y_complex, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 様々な回帰モデル
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', gamma='scale'),
    'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# 各モデルの評価
results = []

for name, regressor in regressors.items():
    # SVRとKNNは標準化されたデータを使用
    if name in ['SVR (RBF)', 'K-Neighbors']:
        regressor.fit(X_train_scaled, y_train)
        y_pred_train = regressor.predict(X_train_scaled)
        y_pred_test = regressor.predict(X_test_scaled)
    else:
        regressor.fit(X_train, y_train)
        y_pred_train = regressor.predict(X_train)
        y_pred_test = regressor.predict(X_test)
    
    results.append({
        'Model': name,
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Train MSE': mean_squared_error(y_train, y_pred_train),
        'Test MSE': mean_squared_error(y_test, y_pred_test),
        'MAE': mean_absolute_error(y_test, y_pred_test)
    })

results_df = pd.DataFrame(results).round(3)
print("回帰モデルの比較:")
print(results_df)

# 予測値の散布図行列
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, regressor) in enumerate(regressors.items()):
    if name in ['SVR (RBF)', 'K-Neighbors']:
        y_pred = regressor.predict(X_test_scaled)
    else:
        y_pred = regressor.predict(X_test)
    
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### サンプルコード6：時系列回帰

```python
# 時系列データの生成
np.random.seed(42)
n_samples = 365  # 1年分のデータ
time = np.arange(n_samples)

# トレンド + 季節性 + ノイズ
trend = 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 365 * 4)  # 四季
noise = np.random.normal(0, 5, n_samples)
y_time = trend + seasonality + noise

# 特徴量エンジニアリング
X_time = pd.DataFrame({
    'time': time,
    'day_of_year': time % 365,
    'sin_seasonal': np.sin(2 * np.pi * time / 365),
    'cos_seasonal': np.cos(2 * np.pi * time / 365),
    'sin_quarterly': np.sin(2 * np.pi * time / 365 * 4),
    'cos_quarterly': np.cos(2 * np.pi * time / 365 * 4)
})

# 訓練データとテストデータの分割（時系列なので順序を保つ）
split_point = int(0.8 * n_samples)
X_train_time = X_time[:split_point]
X_test_time = X_time[split_point:]
y_train_time = y_time[:split_point]
y_test_time = y_time[split_point:]

# モデルの学習
rf_time = RandomForestRegressor(n_estimators=100, random_state=42)
rf_time.fit(X_train_time, y_train_time)

# 予測
y_pred_time = rf_time.predict(X_test_time)

# 可視化
plt.figure(figsize=(15, 8))

# 全体のプロット
plt.subplot(2, 1, 1)
plt.plot(time[:split_point], y_train_time, label='Training Data', alpha=0.7)
plt.plot(time[split_point:], y_test_time, label='Test Data', alpha=0.7)
plt.plot(time[split_point:], y_pred_time, 'r-', label='Predictions', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Time Series Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# 残差のプロット
plt.subplot(2, 1, 2)
residuals_time = y_test_time - y_pred_time
plt.plot(time[split_point:], residuals_time)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Day')
plt.ylabel('Residuals')
plt.title('Prediction Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"時系列回帰の評価:")
print(f"R²スコア: {r2_score(y_test_time, y_pred_time):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_time, y_pred_time)):.2f}")
print(f"MAE: {mean_absolute_error(y_test_time, y_pred_time):.2f}")

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': X_time.columns,
    'importance': rf_time.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量の重要度:")
print(feature_importance)
```

## 練習問題

### 問題1：住宅価格予測
1. 住宅の特徴量（広さ、部屋数、築年数など）を含むデータセットを生成
2. 線形回帰、Ridge回帰、ランダムフォレストで予測モデルを構築
3. 交差検証でモデルを評価
4. 最も重要な特徴量を特定

### 問題2：非線形回帰
1. y = x² + sin(x) + ノイズ のような非線形データを生成
2. 多項式回帰、SVR、ニューラルネットワークで予測
3. 各モデルの汎化性能を比較

### 問題3：正則化の効果
1. 多重共線性のあるデータセットを作成
2. 通常の線形回帰とRidge回帰の係数を比較
3. 正則化パラメータを変化させて最適値を見つける

## 解答

### 解答1：住宅価格予測

```python
# 住宅データセットの生成
np.random.seed(42)
n_houses = 1000

# 特徴量の生成
house_data = pd.DataFrame({
    'area': np.random.normal(150, 50, n_houses),  # 面積（㎡）
    'rooms': np.random.randint(1, 6, n_houses),   # 部屋数
    'age': np.random.randint(0, 50, n_houses),    # 築年数
    'distance_station': np.random.exponential(2, n_houses),  # 駅からの距離（km）
    'floor': np.random.randint(1, 15, n_houses),  # 階数
    'south_facing': np.random.choice([0, 1], n_houses, p=[0.6, 0.4])  # 南向きか
})

# 価格の生成（現実的な関係性を持たせる）
house_data['price'] = (
    300 * house_data['area'] +  # 面積の影響が大きい
    50000 * house_data['rooms'] +  # 部屋数
    -10000 * house_data['age'] +  # 築年数（古いほど安い）
    -30000 * house_data['distance_station'] +  # 駅から遠いほど安い
    5000 * house_data['floor'] +  # 階数
    100000 * house_data['south_facing'] +  # 南向きは高い
    np.random.normal(0, 100000, n_houses)  # ノイズ
) / 10000  # 万円単位に変換

# 負の価格を除去
house_data = house_data[house_data['price'] > 0]

print("住宅データの概要:")
print(house_data.describe())

# 特徴量とターゲットの分離
X_house = house_data.drop('price', axis=1)
y_house = house_data['price']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# モデルの構築と評価
from sklearn.model_selection import cross_val_score

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# 交差検証
cv_results = []

for name, model in models.items():
    # 5分割交差検証
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    
    # テストセットでの評価
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    cv_results.append({
        'Model': name,
        'CV RMSE Mean': rmse_scores.mean(),
        'CV RMSE Std': rmse_scores.std(),
        'Test RMSE': test_rmse,
        'Test R²': test_r2
    })

cv_results_df = pd.DataFrame(cv_results)
print("\n交差検証結果:")
print(cv_results_df.round(3))

# 特徴量の重要度（Random Forest）
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_house.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - House Price Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n特徴量の重要度:")
print(feature_importance)

# 予測値の可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (万円)')
plt.ylabel('Predicted Price (万円)')
plt.title('Random Forest Predictions')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - rf_model.predict(X_test)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (万円)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 解答2：非線形回帰

```python
from sklearn.neural_network import MLPRegressor

# 非線形データの生成
np.random.seed(42)
X_nonlinear = np.sort(np.random.uniform(-5, 5, 300)).reshape(-1, 1)
y_nonlinear = X_nonlinear.ravel()**2 + 5*np.sin(X_nonlinear.ravel()) + np.random.normal(0, 2, 300)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_nonlinear, y_nonlinear, test_size=0.3, random_state=42
)

# データの標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデルの定義
models_nonlinear = {
    'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
    'Polynomial (degree 5)': make_pipeline(PolynomialFeatures(5), LinearRegression()),
    'SVR (RBF)': SVR(kernel='rbf', gamma='scale', C=10),
    'Neural Network': MLPRegressor(hidden_layers=(50, 50), max_iter=1000, random_state=42)
}

# 各モデルの学習と評価
results_nonlinear = []
predictions = {}

for name, model in models_nonlinear.items():
    if name in ['SVR (RBF)', 'Neural Network']:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # プロット用の予測
        X_plot_scaled = scaler.transform(np.linspace(-5, 5, 300).reshape(-1, 1))
        y_plot = model.predict(X_plot_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # プロット用の予測
        X_plot = np.linspace(-5, 5, 300).reshape(-1, 1)
        y_plot = model.predict(X_plot)
    
    predictions[name] = (np.linspace(-5, 5, 300), y_plot)
    
    results_nonlinear.append({
        'Model': name,
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
    })

results_nonlinear_df = pd.DataFrame(results_nonlinear)
print("非線形回帰モデルの比較:")
print(results_nonlinear_df.round(3))

# 予測の可視化
plt.figure(figsize=(15, 10))

for idx, (name, (x_plot, y_plot)) in enumerate(predictions.items(), 1):
    plt.subplot(2, 2, idx)
    plt.scatter(X_nonlinear, y_nonlinear, alpha=0.3, label='Data')
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label=name)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'{name}\nTest R² = {results_nonlinear_df[results_nonlinear_df["Model"]==name]["Test R²"].values[0]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 解答3：正則化の効果

```python
# 多重共線性のあるデータセットの作成
np.random.seed(42)
n_samples = 100
n_features = 20

# 相関の高い特徴量を生成
X_base = np.random.randn(n_samples, 5)
X_collinear = np.zeros((n_samples, n_features))

# 基本特徴量をコピーして相関を作る
for i in range(5):
    X_collinear[:, i] = X_base[:, i]
    X_collinear[:, i+5] = X_base[:, i] + np.random.normal(0, 0.1, n_samples)
    X_collinear[:, i+10] = X_base[:, i] + np.random.normal(0, 0.2, n_samples)
    X_collinear[:, i+15] = X_base[:, i] + np.random.normal(0, 0.3, n_samples)

# ターゲット変数
true_coef = np.zeros(n_features)
true_coef[:5] = [1, -2, 3, -1, 2]
y_collinear = X_collinear @ true_coef + np.random.normal(0, 0.5, n_samples)

# 相関行列の可視化
plt.figure(figsize=(10, 8))
correlation_matrix = np.corrcoef(X_collinear.T)
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            xticklabels=range(n_features), yticklabels=range(n_features))
plt.title('Feature Correlation Matrix (Multicollinearity)')
plt.show()

# データの分割と標準化
X_train, X_test, y_train, y_test = train_test_split(
    X_collinear, y_collinear, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 通常の線形回帰
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Ridge回帰（異なるアルファ値）
alphas = np.logspace(-3, 3, 100)
ridge_coefs = []
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    ridge_scores.append(ridge.score(X_test_scaled, y_test))

ridge_coefs = np.array(ridge_coefs)

# 係数の比較
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.bar(range(n_features), lr.coef_, alpha=0.7, label='Linear Regression')
plt.bar(range(n_features), true_coef, alpha=0.7, label='True Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Linear Regression Coefficients (with multicollinearity)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i in range(n_features):
    plt.plot(np.log10(alphas), ridge_coefs[:, i], alpha=0.5)
plt.xlabel('log10(alpha)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients vs Alpha')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最適なアルファ値の選択
from sklearn.model_selection import cross_val_score

cv_scores = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

best_alpha_idx = np.argmax(cv_scores)
best_alpha = alphas[best_alpha_idx]

plt.figure(figsize=(10, 6))
plt.plot(np.log10(alphas), cv_scores, 'b-', linewidth=2)
plt.plot(np.log10(best_alpha), cv_scores[best_alpha_idx], 'ro', markersize=10)
plt.xlabel('log10(alpha)')
plt.ylabel('Cross-validation R² Score')
plt.title(f'Ridge Regression: Alpha Selection (Best alpha = {best_alpha:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

# 最適なRidgeモデルの評価
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train_scaled, y_train)

print("モデルの比較:")
print(f"線形回帰 - Test R²: {lr.score(X_test_scaled, y_test):.3f}")
print(f"Ridge回帰 (最適α) - Test R²: {ridge_best.score(X_test_scaled, y_test):.3f}")

# 係数の安定性
print(f"\n係数の標準偏差:")
print(f"線形回帰: {np.std(lr.coef_):.3f}")
print(f"Ridge回帰: {np.std(ridge_best.coef_):.3f}")
```

## まとめ

この章では、教師あり学習の回帰問題について学習しました：

- 線形回帰の基本と重回帰分析
- 残差分析による回帰診断
- 多項式回帰と過学習
- 正則化手法（Ridge、Lasso、Elastic Net）
- ランダムフォレスト、Gradient Boostingなどの非線形回帰
- 時系列データの回帰分析
- 多重共線性と正則化の効果

次章では、教師なし学習のクラスタリングについて詳しく学習します。