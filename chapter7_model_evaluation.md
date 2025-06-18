# 第7章：モデルの評価と改善

## 7.1 モデル評価の重要性

機械学習モデルの成功は、適切な評価と継続的な改善にかかっています。この章では以下を学習します：

- **交差検証**: より信頼性の高いモデル評価
- **評価指標**: タスクに応じた適切な指標の選択
- **ハイパーパラメータチューニング**: 最適なモデル設定の探索
- **過学習の診断と対策**: 汎化性能の向上

## 7.2 交差検証

### サンプルコード1：様々な交差検証手法

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, 
    TimeSeriesSplit, GroupKFold, cross_val_score
)
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# データの生成
X, y = make_classification(n_samples=200, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=3, random_state=42)

# 基本モデル
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 1. K-Fold交差検証
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("K-Fold交差検証:")
print(f"各分割のスコア: {scores_kfold}")
print(f"平均スコア: {scores_kfold.mean():.3f} (+/- {scores_kfold.std() * 2:.3f})")

# 2. Stratified K-Fold（クラスバランスを保持）
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stratified = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')

print("\nStratified K-Fold交差検証:")
print(f"各分割のスコア: {scores_stratified}")
print(f"平均スコア: {scores_stratified.mean():.3f} (+/- {scores_stratified.std() * 2:.3f})")

# 交差検証の可視化
def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """交差検証の分割を可視化"""
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    
    # インデックスの生成
    indices = np.arange(len(y))
    
    # 各分割での訓練/テストインデックス
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # 訓練とテストのインデックスを可視化
        indices_train = np.zeros(len(indices), dtype=bool)
        indices_test = np.zeros(len(indices), dtype=bool)
        indices_train[tr] = True
        indices_test[tt] = True
        
        # プロット
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                  c=indices_train, marker='_', lw=lw, cmap=cmap_cv,
                  vmin=0, vmax=1)
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                  c=indices_test, marker='_', lw=lw, cmap=cmap_cv,
                  vmin=0.2, vmax=0.8)
    
    ax.set_ylim(0, n_splits + 1)
    ax.set_xlim(0, len(y))
    ax.set_ylabel('CV Fold', fontsize=15)
    ax.set_xlabel('Sample Index', fontsize=15)
    ax.set_title(f'{cv.__class__.__name__}', fontsize=15)

# 異なる交差検証手法の可視化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

cvs = [
    (KFold(n_splits=5), axes[0, 0]),
    (StratifiedKFold(n_splits=5), axes[0, 1]),
    (TimeSeriesSplit(n_splits=5), axes[1, 0]),
    (KFold(n_splits=10), axes[1, 1])
]

for cv, ax in cvs:
    plot_cv_indices(cv, X[:100], y[:100], ax, cv.n_splits)

plt.tight_layout()
plt.show()
```

### サンプルコード2：カスタム交差検証とネストした交差検証

```python
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

# カスタム評価指標
def custom_score(y_true, y_pred):
    """カスタム評価指標の例（F2スコア）"""
    from sklearn.metrics import fbeta_score
    return fbeta_score(y_true, y_pred, beta=2, average='weighted')

# 複数の評価指標で交差検証
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'custom_f2': make_scorer(custom_score)
}

# 交差検証の実行
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, 
                          return_train_score=True)

# 結果の整理
results_df = pd.DataFrame(cv_results)
print("交差検証の詳細結果:")
print(results_df.describe())

# 訓練スコアとテストスコアの比較
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['accuracy', 'precision', 'recall', 'custom_f2']
train_means = [results_df[f'train_{metric}'].mean() for metric in metrics]
test_means = [results_df[f'test_{metric}'].mean() for metric in metrics]
train_stds = [results_df[f'train_{metric}'].std() for metric in metrics]
test_stds = [results_df[f'test_{metric}'].std() for metric in metrics]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds, 
                label='Train', capsize=5)
bars2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, 
                label='Test', capsize=5)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Train vs Test Scores Across Different Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ネストした交差検証
from sklearn.svm import SVC

# 内側のCV：ハイパーパラメータチューニング
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 外側のCV：モデル評価
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ネストした交差検証の実行
nested_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 内側のCV（グリッドサーチ）
    grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 外側のテストセットで評価
    score = grid_search.score(X_test, y_test)
    nested_scores.append(score)

print(f"\nネストした交差検証スコア: {nested_scores}")
print(f"平均スコア: {np.mean(nested_scores):.3f} (+/- {np.std(nested_scores) * 2:.3f})")
```

## 7.3 評価指標

### サンプルコード3：分類の評価指標

```python
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# 不均衡なデータセットの生成
X_imb, y_imb = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                                   n_features=20, n_informative=15, 
                                   random_state=42)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# モデルの学習
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 予測
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 1. 混同行列
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# 2. 分類レポート
print("分類レポート:")
print(classification_report(y_test, y_pred))

# 3. ROC曲線とPR曲線
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC曲線
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# PR曲線
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

ax2.plot(recall, precision, color='darkorange', lw=2,
         label=f'PR curve (AP = {avg_precision:.3f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. 閾値の影響
thresholds = np.linspace(0, 1, 50)
metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    if len(np.unique(y_pred_thresh)) > 1:  # 両方のクラスが予測される場合
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_thresh, average='binary'
        )
        acc = accuracy_score(y_test, y_pred_thresh)
        
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['accuracy'].append(acc)
    else:
        metrics['precision'].append(np.nan)
        metrics['recall'].append(np.nan)
        metrics['f1'].append(np.nan)
        metrics['accuracy'].append(np.nan)

# プロット
plt.figure(figsize=(10, 6))
for metric, values in metrics.items():
    plt.plot(thresholds, values, label=metric.capitalize(), linewidth=2)

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Decision Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.show()

# 最適な閾値（F1スコア最大化）
best_threshold_idx = np.nanargmax(metrics['f1'])
best_threshold = thresholds[best_threshold_idx]
print(f"\n最適な閾値（F1スコア基準）: {best_threshold:.3f}")
print(f"そのときのF1スコア: {metrics['f1'][best_threshold_idx]:.3f}")
```

### サンプルコード4：回帰の評価指標

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_squared_log_error
)
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# 回帰データの生成
X_reg, y_reg, coef = make_regression(n_samples=200, n_features=20, 
                                     n_informative=10, noise=10,
                                     coef=True, random_state=42)

# データの分割
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 複数のモデルで比較
models_reg = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0)
}

results_reg = []

for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    
    results_reg.append({
        'Model': name,
        'MSE': mean_squared_error(y_test_reg, y_pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg),
        'R²': r2_score(y_test_reg, y_pred_reg),
        'Explained Variance': explained_variance_score(y_test_reg, y_pred_reg)
    })

results_reg_df = pd.DataFrame(results_reg)
print("回帰モデルの評価指標:")
print(results_reg_df.round(3))

# 予測値と残差のプロット
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models_reg.items()):
    y_pred = model.predict(X_test_reg)
    residuals = y_test_reg - y_pred
    
    # 予測値 vs 実際の値
    ax1 = axes[idx]
    ax1.scatter(y_test_reg, y_pred, alpha=0.5)
    ax1.plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'{name}: Predictions vs True')
    ax1.grid(True, alpha=0.3)
    
    # 残差プロット
    ax2 = axes[idx + 3]
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{name}: Residual Plot')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# カスタム評価指標：MAPE（Mean Absolute Percentage Error）
def mean_absolute_percentage_error(y_true, y_pred):
    """MAPEの計算（0除算を避ける）"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 各モデルのMAPE
print("\nMAPE (Mean Absolute Percentage Error):")
for name, model in models_reg.items():
    y_pred = model.predict(X_test_reg)
    mape = mean_absolute_percentage_error(y_test_reg, y_pred)
    print(f"{name}: {mape:.2f}%")
```

## 7.4 ハイパーパラメータチューニング

### サンプルコード5：Grid Search vs Random Search vs Bayesian Optimization

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import time

# より複雑なデータセット
X_complex, y_complex = make_classification(
    n_samples=1000, n_features=30, n_informative=20,
    n_redundant=10, n_classes=5, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_complex, y_complex, test_size=0.3, random_state=42
)

# 1. Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Grid Search実行中...")
start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_c, y_train_c)

grid_time = time.time() - start_time
print(f"Grid Search完了: {grid_time:.2f}秒")
print(f"最適パラメータ: {grid_search.best_params_}")
print(f"最適スコア: {grid_search.best_score_:.3f}")

# 2. Random Search
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.7)
}

print("\nRandom Search実行中...")
start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train_c, y_train_c)

random_time = time.time() - start_time
print(f"Random Search完了: {random_time:.2f}秒")
print(f"最適パラメータ: {random_search.best_params_}")
print(f"最適スコア: {random_search.best_score_:.3f}")

# 結果の比較
print(f"\n実行時間の比較:")
print(f"Grid Search: {grid_time:.2f}秒")
print(f"Random Search: {random_time:.2f}秒")
print(f"高速化: {grid_time/random_time:.2f}倍")

# パラメータ空間の可視化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Grid Searchの結果
grid_results = pd.DataFrame(grid_search.cv_results_)
pivot_grid = grid_results.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators'
)

sns.heatmap(pivot_grid, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
axes[0].set_title('Grid Search: Score Heatmap')

# Random Searchの結果（散布図）
random_results = pd.DataFrame(random_search.cv_results_)
scatter = axes[1].scatter(
    random_results['param_n_estimators'],
    random_results['param_max_depth'].fillna(30),  # Noneを30として表示
    c=random_results['mean_test_score'],
    cmap='viridis',
    alpha=0.6
)
axes[1].set_xlabel('n_estimators')
axes[1].set_ylabel('max_depth')
axes[1].set_title('Random Search: Parameter Space Exploration')
plt.colorbar(scatter, ax=axes[1], label='Score')

plt.tight_layout()
plt.show()
```

### サンプルコード6：ベイズ最適化（簡易実装）

```python
# Scikit-optimizeを使用したベイズ最適化の例
# 注：実際にはpip install scikit-optimizeが必要
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class SimpleBayesianOptimization:
    """簡易的なベイズ最適化の実装"""
    
    def __init__(self, func, bounds, n_calls=20):
        self.func = func
        self.bounds = bounds
        self.n_calls = n_calls
        self.X_samples = []
        self.y_samples = []
        
    def acquisition_function(self, X, gp, y_max):
        """獲得関数（Expected Improvement）"""
        mu, sigma = gp.predict(X.reshape(-1, len(self.bounds)), return_std=True)
        
        with np.errstate(divide='warn'):
            Z = (mu - y_max) / sigma
            ei = (mu - y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def optimize(self):
        """ベイズ最適化の実行"""
        from scipy.stats import norm
        
        # 初期サンプリング
        for _ in range(5):
            x = [np.random.uniform(low, high) for low, high in self.bounds]
            y = self.func(x)
            self.X_samples.append(x)
            self.y_samples.append(y)
        
        # ガウス過程の初期化
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # 最適化ループ
        for i in range(self.n_calls - 5):
            # ガウス過程の学習
            gp.fit(np.array(self.X_samples), np.array(self.y_samples))
            
            # 次のサンプル点を獲得関数で決定
            y_max = max(self.y_samples)
            
            # グリッドサーチで獲得関数を最大化
            x_tries = np.random.uniform(
                [b[0] for b in self.bounds],
                [b[1] for b in self.bounds],
                size=(1000, len(self.bounds))
            )
            
            ei_values = [self.acquisition_function(x, gp, y_max) for x in x_tries]
            x_next = x_tries[np.argmax(ei_values)]
            
            # 新しい点での評価
            y_next = self.func(x_next)
            self.X_samples.append(x_next.tolist())
            self.y_samples.append(y_next)
        
        # 最適解
        best_idx = np.argmax(self.y_samples)
        return self.X_samples[best_idx], self.y_samples[best_idx]

# ベイズ最適化の使用例
def objective_function(params):
    """最適化する目的関数（RandomForestの交差検証スコア）"""
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    scores = cross_val_score(rf, X_train_c, y_train_c, cv=3)
    return scores.mean()

# パラメータの範囲
bounds = [(50, 300), (5, 30), (2, 20)]

# ベイズ最適化の実行
print("ベイズ最適化実行中...")
start_time = time.time()

bayes_opt = SimpleBayesianOptimization(objective_function, bounds, n_calls=30)
best_params, best_score = bayes_opt.optimize()

bayes_time = time.time() - start_time
print(f"ベイズ最適化完了: {bayes_time:.2f}秒")
print(f"最適パラメータ: n_estimators={int(best_params[0])}, " +
      f"max_depth={int(best_params[1])}, min_samples_split={int(best_params[2])}")
print(f"最適スコア: {best_score:.3f}")

# 最適化過程の可視化
plt.figure(figsize=(10, 6))
plt.plot(range(len(bayes_opt.y_samples)), bayes_opt.y_samples, 'bo-')
plt.xlabel('Iteration')
plt.ylabel('Cross-validation Score')
plt.title('Bayesian Optimization Progress')
plt.grid(True, alpha=0.3)
plt.show()
```

## 7.5 学習曲線と検証曲線

### サンプルコード7：過学習の診断

```python
from sklearn.model_selection import learning_curve, validation_curve

# 学習曲線
def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5):
    """学習曲線のプロット"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='orange')
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue',
             label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='orange',
             label='Cross-validation score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.01)
    plt.show()
    
    return train_sizes, train_scores, val_scores

# 異なる複雑さのモデルで比較
models_complexity = {
    'Low Complexity': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
    'Medium Complexity': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    'High Complexity': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
}

for name, model in models_complexity.items():
    print(f"\n{name}:")
    plot_learning_curve(model, X_complex, y_complex, title=f"Learning Curve - {name}")

# 検証曲線
def plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve"):
    """検証曲線のプロット"""
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name,
        param_range=param_range, cv=5, scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(param_range, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(param_range, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='orange')
    
    plt.plot(param_range, train_mean, 'o-', color='blue',
             label='Training score')
    plt.plot(param_range, val_mean, 'o-', color='orange',
             label='Cross-validation score')
    
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 最適値を表示
    best_idx = np.argmax(val_mean)
    plt.axvline(x=param_range[best_idx], color='green', linestyle='--',
                label=f'Best {param_name}={param_range[best_idx]}')
    plt.legend()
    plt.show()

# max_depthの検証曲線
param_range = np.arange(1, 21)
plot_validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_complex, y_complex,
    param_name='max_depth',
    param_range=param_range,
    title='Validation Curve - Max Depth'
)

# n_estimatorsの検証曲線
param_range = np.arange(10, 210, 20)
plot_validation_curve(
    RandomForestClassifier(max_depth=10, random_state=42),
    X_complex, y_complex,
    param_name='n_estimators',
    param_range=param_range,
    title='Validation Curve - Number of Trees'
)
```

### サンプルコード8：アンサンブル手法による改善

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 基本分類器の定義
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'svc': SVC(probability=True, random_state=42),
    'nb': GaussianNB()
}

# 個々のモデルの性能
individual_scores = {}

for name, model in base_models.items():
    scores = cross_val_score(model, X_train_c, y_train_c, cv=5)
    individual_scores[name] = scores
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Voting Classifier
voting_clf_hard = VotingClassifier(
    estimators=list(base_models.items()),
    voting='hard'
)

voting_clf_soft = VotingClassifier(
    estimators=list(base_models.items()),
    voting='soft'
)

# アンサンブル手法の比較
ensemble_models = {
    'Voting (Hard)': voting_clf_hard,
    'Voting (Soft)': voting_clf_soft,
    'Bagging': BaggingClassifier(base_estimator=base_models['rf'], 
                                 n_estimators=10, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

ensemble_scores = {}

print("\nアンサンブル手法:")
for name, model in ensemble_models.items():
    scores = cross_val_score(model, X_train_c, y_train_c, cv=5)
    ensemble_scores[name] = scores
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# 結果の可視化
all_scores = {**individual_scores, **ensemble_scores}

plt.figure(figsize=(12, 8))
positions = np.arange(len(all_scores))

for i, (name, scores) in enumerate(all_scores.items()):
    plt.scatter([i] * len(scores), scores, alpha=0.4, s=100)
    plt.boxplot(scores, positions=[i], widths=0.5)

plt.xticks(positions, all_scores.keys(), rotation=45, ha='right')
plt.ylabel('Cross-validation Score')
plt.title('Model Performance Comparison')
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=np.mean([s.mean() for s in individual_scores.values()]), 
            color='red', linestyle='--', label='Average Individual Score')
plt.legend()
plt.tight_layout()
plt.show()

# 特徴量の重要度（アンサンブルから）
voting_clf_soft.fit(X_train_c, y_train_c)

# Random Forestコンポーネントから特徴量重要度を取得
rf_in_voting = voting_clf_soft.named_estimators_['rf']
feature_importance = rf_in_voting.feature_importances_

# 上位10個の重要な特徴量
top_features_idx = np.argsort(feature_importance)[-10:][::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(10), feature_importance[top_features_idx])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importances from Ensemble')
plt.xticks(range(10), [f'Feature {i}' for i in top_features_idx])
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

## 練習問題

### 問題1：カスタム評価指標の実装
1. ビジネス要件に基づくカスタム評価指標を定義（例：誤分類のコスト考慮）
2. GridSearchCVでカスタム指標を使用
3. 標準的な指標との比較

### 問題2：時系列データの交差検証
1. 時系列データを生成
2. TimeSeriesSplitを使った交差検証
3. 通常のK-Foldとの結果比較

### 問題3：モデルの公平性評価
1. バイアスのあるデータセットを作成
2. 異なるグループ間での性能差を評価
3. 公平性を改善する手法の実装

## 解答

### 解答1：カスタム評価指標の実装

```python
# ビジネスシナリオ：医療診断
# False Negative（病気を見逃す）のコストがFalse Positiveより10倍高い

def custom_medical_score(y_true, y_pred):
    """医療診断用のカスタムスコア"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # コスト行列
    cost_fp = 1  # 健康な人を病気と誤診
    cost_fn = 10  # 病気の人を健康と誤診
    
    # 総コスト
    total_cost = fp * cost_fp + fn * cost_fn
    
    # スコアに変換（コストが低いほど良い）
    # 最大コストで正規化して0-1の範囲に
    max_cost = len(y_true) * cost_fn
    score = 1 - (total_cost / max_cost)
    
    return score

# 医療データのシミュレーション
X_medical, y_medical = make_classification(
    n_samples=1000, n_features=20, n_classes=2,
    weights=[0.9, 0.1],  # 10%が病気
    random_state=42
)

X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_medical, y_medical, test_size=0.3, random_state=42, stratify=y_medical
)

# カスタムスコアラーの作成
custom_scorer = make_scorer(custom_medical_score, greater_is_better=True)

# 複数の評価指標での比較
scoring_comparison = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'custom_medical': custom_scorer
}

# GridSearchCVでの使用
param_grid_medical = {
    'C': [0.1, 1, 10],
    'class_weight': [None, 'balanced', {0: 1, 1: 5}, {0: 1, 1: 10}]
}

from sklearn.linear_model import LogisticRegression

results_by_metric = {}

for metric_name, metric in scoring_comparison.items():
    print(f"\n{metric_name}で最適化:")
    
    grid = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid_medical,
        cv=5,
        scoring=metric,
        n_jobs=-1
    )
    
    grid.fit(X_train_med, y_train_med)
    
    # 最適モデルでの予測
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_med)
    
    # 各種指標での評価
    tn, fp, fn, tp = confusion_matrix(y_test_med, y_pred).ravel()
    
    results_by_metric[metric_name] = {
        'best_params': grid.best_params_,
        'accuracy': accuracy_score(y_test_med, y_pred),
        'f1': f1_score(y_test_med, y_pred),
        'custom': custom_medical_score(y_test_med, y_pred),
        'false_negatives': fn,
        'false_positives': fp
    }
    
    print(f"最適パラメータ: {grid.best_params_}")
    print(f"False Negatives: {fn}, False Positives: {fp}")

# 結果の比較
comparison_df = pd.DataFrame(results_by_metric).T
print("\n各指標で最適化した結果の比較:")
print(comparison_df)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# False Negatives vs False Positives
ax1 = axes[0]
for metric in results_by_metric:
    result = results_by_metric[metric]
    ax1.scatter(result['false_positives'], result['false_negatives'],
               s=200, label=metric)
    ax1.annotate(metric, (result['false_positives'], result['false_negatives']))

ax1.set_xlabel('False Positives')
ax1.set_ylabel('False Negatives')
ax1.set_title('Trade-off: FP vs FN')
ax1.legend()
ax1.grid(True, alpha=0.3)

# カスタムスコアの比較
ax2 = axes[1]
metrics = list(results_by_metric.keys())
custom_scores = [results_by_metric[m]['custom'] for m in metrics]
colors = ['red' if m == 'custom_medical' else 'blue' for m in metrics]

bars = ax2.bar(metrics, custom_scores, color=colors)
ax2.set_ylabel('Custom Medical Score')
ax2.set_title('Custom Score Comparison')
ax2.grid(True, alpha=0.3, axis='y')

# 値をバーの上に表示
for bar, score in zip(bars, custom_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### 解答2：時系列データの交差検証

```python
from sklearn.model_selection import TimeSeriesSplit

# 時系列データの生成
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

# トレンド + 季節性 + ノイズ
trend = 0.002 * time
seasonality = np.sin(2 * np.pi * time / 50)
noise = 0.5 * np.random.randn(n_samples)

# 特徴量：過去の値
X_ts = np.column_stack([
    np.roll(trend + seasonality + noise, i) for i in range(1, 11)
])[10:]  # 最初の10個は除外

# ターゲット：現在の値
y_ts = (trend + seasonality + noise)[10:]

# 時系列分割
tscv = TimeSeriesSplit(n_splits=5)

# 通常のK-Foldとの比較
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 両方の手法でモデルを評価
from sklearn.ensemble import GradientBoostingRegressor

model_ts = GradientBoostingRegressor(n_estimators=100, random_state=42)

# TimeSeriesSplitでの評価
ts_scores = []
ts_train_sizes = []
ts_test_sizes = []

for train_idx, test_idx in tscv.split(X_ts):
    X_train_fold, X_test_fold = X_ts[train_idx], X_ts[test_idx]
    y_train_fold, y_test_fold = y_ts[train_idx], y_ts[test_idx]
    
    model_ts.fit(X_train_fold, y_train_fold)
    score = model_ts.score(X_test_fold, y_test_fold)
    ts_scores.append(score)
    ts_train_sizes.append(len(train_idx))
    ts_test_sizes.append(len(test_idx))

# 通常のK-Foldでの評価
kf_scores = cross_val_score(model_ts, X_ts, y_ts, cv=kfold, scoring='r2')

print("時系列交差検証:")
for i, (score, train_size, test_size) in enumerate(zip(ts_scores, ts_train_sizes, ts_test_sizes)):
    print(f"  Fold {i+1}: Score={score:.3f}, Train={train_size}, Test={test_size}")
print(f"  平均スコア: {np.mean(ts_scores):.3f} (+/- {np.std(ts_scores) * 2:.3f})")

print(f"\n通常のK-Fold交差検証:")
print(f"  スコア: {kf_scores}")
print(f"  平均スコア: {kf_scores.mean():.3f} (+/- {kf_scores.std() * 2:.3f})")

# 分割の可視化
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# TimeSeriesSplit
ax1 = axes[0]
for i, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    # 訓練データ
    ax1.scatter(train_idx, [i] * len(train_idx), c='blue', marker='s', s=10)
    # テストデータ
    ax1.scatter(test_idx, [i] * len(test_idx), c='red', marker='s', s=10)

ax1.set_ylim(-0.5, 4.5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('CV Fold')
ax1.set_title('TimeSeriesSplit')
ax1.legend(['Train', 'Test'])

# 通常のK-Fold
ax2 = axes[1]
for i, (train_idx, test_idx) in enumerate(kfold.split(X_ts)):
    ax2.scatter(train_idx, [i] * len(train_idx), c='blue', marker='s', s=10)
    ax2.scatter(test_idx, [i] * len(test_idx), c='red', marker='s', s=10)

ax2.set_ylim(-0.5, 4.5)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('CV Fold')
ax2.set_title('K-Fold (Shuffled)')
ax2.legend(['Train', 'Test'])

plt.tight_layout()
plt.show()

# 予測の時系列プロット
# 最後の分割での予測
train_idx, test_idx = list(tscv.split(X_ts))[-1]
X_train_last, X_test_last = X_ts[train_idx], X_ts[test_idx]
y_train_last, y_test_last = y_ts[train_idx], y_ts[test_idx]

model_ts.fit(X_train_last, y_train_last)
y_pred_ts = model_ts.predict(X_test_last)

plt.figure(figsize=(15, 6))
plt.plot(train_idx, y_ts[train_idx], 'b-', label='Training Data', alpha=0.6)
plt.plot(test_idx, y_ts[test_idx], 'g-', label='Test Data (Actual)', alpha=0.6)
plt.plot(test_idx, y_pred_ts, 'r--', label='Test Data (Predicted)', alpha=0.8)
plt.xlabel('Time Index')
plt.ylabel('Value')
plt.title('Time Series Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 解答3：モデルの公平性評価

```python
# バイアスのあるデータセットの作成
np.random.seed(42)

# グループA（多数派）
n_group_a = 800
X_group_a = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_group_a)
# グループAは承認されやすい
y_group_a = (X_group_a[:, 0] + X_group_a[:, 1] + np.random.normal(0, 0.5, n_group_a) > 0).astype(int)

# グループB（少数派）
n_group_b = 200
X_group_b = np.random.multivariate_normal([0.5, 0.5], [[1, 0.5], [0.5, 1]], n_group_b)
# グループBは承認されにくい（バイアス）
y_group_b = (X_group_b[:, 0] + X_group_b[:, 1] + np.random.normal(-1, 0.5, n_group_b) > 1).astype(int)

# データの結合
X_fair = np.vstack([X_group_a, X_group_b])
y_fair = np.hstack([y_group_a, y_group_b])
group_fair = np.hstack([np.zeros(n_group_a), np.ones(n_group_b)])

# データのシャッフル
shuffle_idx = np.random.permutation(len(X_fair))
X_fair = X_fair[shuffle_idx]
y_fair = y_fair[shuffle_idx]
group_fair = group_fair[shuffle_idx]

print("データの統計:")
print(f"グループA: {n_group_a}人, 承認率: {y_group_a.mean():.2%}")
print(f"グループB: {n_group_b}人, 承認率: {y_group_b.mean():.2%}")

# データの分割
X_train_fair, X_test_fair, y_train_fair, y_test_fair, group_train, group_test = train_test_split(
    X_fair, y_fair, group_fair, test_size=0.3, random_state=42, stratify=group_fair
)

# 通常のモデル学習
model_biased = LogisticRegression(random_state=42)
model_biased.fit(X_train_fair, y_train_fair)

# 予測
y_pred_fair = model_biased.predict(X_test_fair)
y_proba_fair = model_biased.predict_proba(X_test_fair)[:, 1]

# 公平性の評価
def evaluate_fairness(y_true, y_pred, y_proba, groups):
    """グループ間の公平性を評価"""
    results = {}
    
    for group in np.unique(groups):
        mask = groups == group
        
        # 各グループの指標
        accuracy = accuracy_score(y_true[mask], y_pred[mask])
        
        # 承認率
        positive_rate = y_pred[mask].mean()
        
        # True Positive Rate (Recall)
        if y_true[mask].sum() > 0:
            tpr = recall_score(y_true[mask], y_pred[mask])
        else:
            tpr = np.nan
        
        # False Positive Rate
        if (y_true[mask] == 0).sum() > 0:
            fpr = (y_pred[mask][y_true[mask] == 0] == 1).mean()
        else:
            fpr = np.nan
        
        results[f'Group {int(group)}'] = {
            'n_samples': mask.sum(),
            'accuracy': accuracy,
            'positive_rate': positive_rate,
            'tpr': tpr,
            'fpr': fpr,
            'actual_positive_rate': y_true[mask].mean()
        }
    
    return pd.DataFrame(results).T

# 公平性評価
fairness_results = evaluate_fairness(y_test_fair, y_pred_fair, y_proba_fair, group_test)
print("\n公平性評価（バイアスありモデル）:")
print(fairness_results.round(3))

# 公平性を改善する手法
# 1. 再重み付け
from sklearn.utils.class_weight import compute_sample_weight

# グループとクラスの組み合わせで重み付け
combined_labels = group_train * 2 + y_train_fair
sample_weights = compute_sample_weight('balanced', combined_labels)

# 重み付きモデル
model_weighted = LogisticRegression(random_state=42)
model_weighted.fit(X_train_fair, y_train_fair, sample_weight=sample_weights)

y_pred_weighted = model_weighted.predict(X_test_fair)

# 2. 閾値の調整
# グループごとに異なる閾値を使用
thresholds = {}
for group in [0, 1]:
    mask = group_train == group
    # ROC曲線から最適な閾値を決定
    fpr, tpr, threshs = roc_curve(y_train_fair[mask], 
                                  model_biased.predict_proba(X_train_fair[mask])[:, 1])
    # Youden's J統計量
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    thresholds[group] = threshs[best_thresh_idx]

# グループごとの閾値で予測
y_pred_adjusted = np.zeros_like(y_test_fair)
for group in [0, 1]:
    mask = group_test == group
    y_pred_adjusted[mask] = (y_proba_fair[mask] >= thresholds[group]).astype(int)

# 改善後の評価
print("\n公平性評価（重み付きモデル）:")
fairness_weighted = evaluate_fairness(y_test_fair, y_pred_weighted, 
                                    model_weighted.predict_proba(X_test_fair)[:, 1], 
                                    group_test)
print(fairness_weighted.round(3))

print("\n公平性評価（閾値調整）:")
fairness_adjusted = evaluate_fairness(y_test_fair, y_pred_adjusted, 
                                    y_proba_fair, group_test)
print(fairness_adjusted.round(3))

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 承認率の比較
ax = axes[0, 0]
methods = ['Original', 'Weighted', 'Threshold Adjusted']
group_a_rates = [
    fairness_results.loc['Group 0', 'positive_rate'],
    fairness_weighted.loc['Group 0', 'positive_rate'],
    fairness_adjusted.loc['Group 0', 'positive_rate']
]
group_b_rates = [
    fairness_results.loc['Group 1', 'positive_rate'],
    fairness_weighted.loc['Group 1', 'positive_rate'],
    fairness_adjusted.loc['Group 1', 'positive_rate']
]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, group_a_rates, width, label='Group A', alpha=0.8)
bars2 = ax.bar(x + width/2, group_b_rates, width, label='Group B', alpha=0.8)

ax.set_xlabel('Method')
ax.set_ylabel('Positive Rate')
ax.set_title('Approval Rates by Group')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 格差の可視化
ax = axes[0, 1]
disparities = [abs(a - b) for a, b in zip(group_a_rates, group_b_rates)]
bars = ax.bar(methods, disparities, color=['red', 'orange', 'green'])
ax.set_ylabel('Disparity (|Group A - Group B|)')
ax.set_title('Disparity in Approval Rates')
ax.grid(True, alpha=0.3, axis='y')

# ROC曲線（グループ別）
for idx, (method, y_pred_method) in enumerate([
    ('Original', y_pred_fair),
    ('Weighted', y_pred_weighted)
]):
    ax = axes[1, idx]
    
    for group in [0, 1]:
        mask = group_test == group
        if method == 'Original':
            y_score = y_proba_fair[mask]
        else:
            y_score = model_weighted.predict_proba(X_test_fair[mask])[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test_fair[mask], y_score)
        auc_score = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'Group {["A", "B"][group]} (AUC={auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {method}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 公平性指標のサマリー
print("\n公平性改善のサマリー:")
print(f"承認率の格差:")
print(f"  元のモデル: {abs(group_a_rates[0] - group_b_rates[0]):.3f}")
print(f"  重み付きモデル: {abs(group_a_rates[1] - group_b_rates[1]):.3f}")
print(f"  閾値調整: {abs(group_a_rates[2] - group_b_rates[2]):.3f}")
```

## まとめ

この章では、モデルの評価と改善について学習しました：

- **交差検証**: K-Fold、Stratified K-Fold、時系列分割など
- **評価指標**: 分類（精度、適合率、再現率、F1、AUC）と回帰（MSE、MAE、R²）
- **ハイパーパラメータチューニング**: Grid Search、Random Search、ベイズ最適化
- **過学習の診断**: 学習曲線、検証曲線
- **モデルの改善**: アンサンブル手法、公平性の考慮

次章では、これらの技術を統合したパイプラインとワークフローについて学習します。

[目次に戻る](README.md)