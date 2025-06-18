# 第8章：パイプラインとワークフロー

## 8.1 パイプラインの重要性

機械学習プロジェクトでは、データの前処理からモデルの学習まで多くのステップが必要です。パイプラインは以下の利点を提供します：

- **再現性**: 全ての処理を一貫して実行
- **データ漏洩の防止**: 前処理とモデル学習を適切に分離
- **コードの簡潔性**: 複雑なワークフローを整理
- **本番環境への展開**: 学習済みパイプラインをそのまま使用可能

## 8.2 基本的なパイプライン

### サンプルコード1：シンプルなパイプラインの構築

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# データの生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# パイプラインの構築
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # ステップ1: 標準化
    ('pca', PCA(n_components=10)),     # ステップ2: 次元削減
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ステップ3: 分類
])

# パイプラインの学習
pipeline.fit(X_train, y_train)

# 予測
y_pred = pipeline.predict(X_test)

# 評価
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"パイプラインの精度: {accuracy:.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred))

# パイプラインの各ステップへのアクセス
print("\nパイプラインの構成:")
for name, step in pipeline.named_steps.items():
    print(f"- {name}: {step}")

# 中間結果の取得
X_scaled = pipeline.named_steps['scaler'].transform(X_train)
X_pca = pipeline.named_steps['pca'].transform(X_scaled)
print(f"\nPCA後の次元数: {X_pca.shape[1]}")
```

### サンプルコード2：ColumnTransformerを使った高度な前処理

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# 混合型データの作成
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n_samples),
    'city': np.random.choice(['NY', 'LA', 'Chicago', 'Houston'], n_samples),
    'experience': np.random.randint(0, 30, n_samples),
    'target': np.random.choice([0, 1], n_samples)
})

# 欠損値を追加
data.loc[data.sample(frac=0.1).index, 'income'] = np.nan
data.loc[data.sample(frac=0.05).index, 'age'] = np.nan

print("データの概要:")
print(data.info())
print("\n欠損値:")
print(data.isnull().sum())

# 特徴量とターゲットの分離
X = data.drop('target', axis=1)
y = data['target']

# 数値とカテゴリカルの列を特定
numeric_features = ['age', 'income', 'experience']
categorical_features = ['education', 'city']

# 前処理パイプライン
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformerで統合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 完全なパイプライン
from sklearn.linear_model import LogisticRegression

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 学習と評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

print(f"\n完全パイプラインの精度: {accuracy_score(y_test, y_pred):.3f}")

# 前処理後の特徴量名を取得
feature_names = (numeric_features + 
                list(full_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

print(f"\n前処理後の特徴量数: {len(feature_names)}")
print(f"特徴量名: {feature_names[:10]}...")  # 最初の10個を表示
```

### サンプルコード3：カスタムトランスフォーマーの作成

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    """カスタム特徴量エンジニアリング"""
    
    def __init__(self, add_polynomial=True, add_log=True):
        self.add_polynomial = add_polynomial
        self.add_log = add_log
        
    def fit(self, X, y=None):
        # 特に学習することはない
        return self
    
    def transform(self, X):
        # 入力をコピー
        X_transformed = X.copy()
        
        # 多項式特徴量
        if self.add_polynomial:
            X_squared = X ** 2
            X_transformed = np.hstack([X_transformed, X_squared])
        
        # 対数変換（正の値のみ）
        if self.add_log:
            X_log = np.log1p(np.abs(X))
            X_transformed = np.hstack([X_transformed, X_log])
        
        return X_transformed

class OutlierRemover(BaseEstimator, TransformerMixin):
    """外れ値除去トランスフォーマー"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        # IQRを計算
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.IQR = Q3 - Q1
        self.lower_bound = Q1 - self.factor * self.IQR
        self.upper_bound = Q3 + self.factor * self.IQR
        return self
    
    def transform(self, X):
        # 外れ値をクリッピング
        return np.clip(X, self.lower_bound, self.upper_bound)

# カスタムトランスフォーマーを含むパイプライン
custom_pipeline = Pipeline([
    ('outlier_remover', OutlierRemover(factor=3.0)),
    ('feature_engineering', CustomFeatureEngineering()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# データの準備
X_custom, y_custom = make_classification(n_samples=1000, n_features=10, 
                                        n_informative=8, random_state=42)

# 外れ値を追加
outlier_indices = np.random.choice(len(X_custom), 50, replace=False)
X_custom[outlier_indices] += np.random.normal(0, 10, (50, 10))

# 学習と評価
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_custom, y_custom, test_size=0.3, random_state=42
)

custom_pipeline.fit(X_train_c, y_train_c)
y_pred_c = custom_pipeline.predict(X_test_c)

print(f"カスタムパイプラインの精度: {accuracy_score(y_test_c, y_pred_c):.3f}")

# 各ステップでの変換を確認
print("\n各ステップでのデータ形状:")
X_temp = X_train_c
for name, step in custom_pipeline.named_steps.items():
    if name != 'classifier':
        X_temp = step.transform(X_temp)
        print(f"{name}: {X_temp.shape}")
```

## 8.3 パイプラインとグリッドサーチ

### サンプルコード4：パイプライン内でのハイパーパラメータチューニング

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# パイプラインの構築
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(random_state=42))
])

# パラメータグリッド（パイプラインのステップ名を使用）
param_grid = {
    'pca__n_components': [5, 10, 15],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01]
}

# グリッドサーチ
grid_search = GridSearchCV(pipe_svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最適なパラメータ: {grid_search.best_params_}")
print(f"最適なスコア: {grid_search.best_score_:.3f}")

# 最適なパイプラインでの評価
best_pipeline = grid_search.best_estimator_
y_pred_best = best_pipeline.predict(X_test)
print(f"テストセットでの精度: {accuracy_score(y_test, y_pred_best):.3f}")

# グリッドサーチ結果の可視化
results_df = pd.DataFrame(grid_search.cv_results_)

# PCAコンポーネント数の影響
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for n_comp in param_grid['pca__n_components']:
    mask = results_df['param_pca__n_components'] == n_comp
    subset = results_df[mask].sort_values('param_svm__C')
    plt.plot(subset['param_svm__C'], subset['mean_test_score'], 
             marker='o', label=f'n_components={n_comp}')

plt.xlabel('SVM C parameter')
plt.ylabel('Mean CV Score')
plt.xscale('log')
plt.legend()
plt.title('Performance vs C parameter')
plt.grid(True, alpha=0.3)

# ヒートマップ
plt.subplot(1, 2, 2)
# 特定のgamma値でのヒートマップ
gamma_value = 'scale'
mask = results_df['param_svm__gamma'] == gamma_value
pivot_table = results_df[mask].pivot_table(
    values='mean_test_score',
    index='param_pca__n_components',
    columns='param_svm__C'
)

import seaborn as sns
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
plt.title(f'Grid Search Results (gamma={gamma_value})')
plt.tight_layout()
plt.show()
```

### サンプルコード5：複数のパイプラインの比較

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# 異なるパイプラインの定義
pipelines = {
    'svm': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('classifier', SVC(random_state=42))
    ]),
    'rf': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'gb': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'nb': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('classifier', GaussianNB())
    ])
}

# 各パイプラインの評価
results = []

for name, pipeline in pipelines.items():
    # 交差検証
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    # テストセットでの評価
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    results.append({
        'Pipeline': name,
        'CV Mean': scores.mean(),
        'CV Std': scores.std(),
        'Test Score': test_score
    })
    
    print(f"{name}: CV={scores.mean():.3f} (+/- {scores.std()*2:.3f}), Test={test_score:.3f}")

# 結果の可視化
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['CV Mean'], width, 
                yerr=results_df['CV Std']*2, label='CV Score', capsize=5)
bars2 = ax.bar(x + width/2, results_df['Test Score'], width, 
                label='Test Score')

ax.set_xlabel('Pipeline')
ax.set_ylabel('Accuracy Score')
ax.set_title('Pipeline Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Pipeline'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 8.4 特徴量結合とパイプライン

### サンプルコード6：FeatureUnionを使った特徴量の結合

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif

# 複数の特徴量抽出手法を組み合わせる
combined_features = FeatureUnion([
    ('pca', PCA(n_components=5)),                    # PCA特徴量
    ('svd', TruncatedSVD(n_components=5)),          # SVD特徴量
    ('kbest', SelectKBest(f_classif, k=5))         # 統計的に選択された特徴量
])

# 結合された特徴量を使うパイプライン
union_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('features', combined_features),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 学習と評価
union_pipeline.fit(X_train, y_train)
y_pred_union = union_pipeline.predict(X_test)

print(f"FeatureUnionパイプラインの精度: {accuracy_score(y_test, y_pred_union):.3f}")

# 各特徴量抽出手法の寄与を確認
X_scaled = union_pipeline.named_steps['scaler'].transform(X_train)
X_combined = union_pipeline.named_steps['features'].transform(X_scaled)

print(f"\n結合後の特徴量数: {X_combined.shape[1]}")
print("内訳:")
print(f"  PCA: 5次元")
print(f"  SVD: 5次元")
print(f"  SelectKBest: 5次元")
print(f"  合計: 15次元")
```

### サンプルコード7：並列処理パイプライン

```python
from sklearn.base import clone
from joblib import Parallel, delayed

class ParallelPipeline:
    """複数のパイプラインを並列実行してアンサンブル"""
    
    def __init__(self, pipelines, voting='soft'):
        self.pipelines = pipelines
        self.voting = voting
        self.fitted_pipelines_ = []
        
    def fit(self, X, y):
        # 並列でパイプラインを学習
        def fit_pipeline(name, pipeline, X, y):
            print(f"Training {name}...")
            return name, clone(pipeline).fit(X, y)
        
        self.fitted_pipelines_ = Parallel(n_jobs=-1)(
            delayed(fit_pipeline)(name, pipeline, X, y) 
            for name, pipeline in self.pipelines.items()
        )
        
        return self
    
    def predict(self, X):
        # 各パイプラインの予測を収集
        predictions = []
        
        for name, pipeline in self.fitted_pipelines_:
            if self.voting == 'soft':
                pred = pipeline.predict_proba(X)[:, 1]
            else:
                pred = pipeline.predict(X)
            predictions.append(pred)
        
        # 投票または平均
        predictions = np.array(predictions)
        
        if self.voting == 'soft':
            # 確率の平均
            avg_proba = predictions.mean(axis=0)
            return (avg_proba >= 0.5).astype(int)
        else:
            # 多数決
            return np.round(predictions.mean(axis=0)).astype(int)

# 並列パイプラインの使用例
parallel_pipeline = ParallelPipeline(pipelines)
parallel_pipeline.fit(X_train, y_train)
y_pred_parallel = parallel_pipeline.predict(X_test)

print(f"並列アンサンブルパイプラインの精度: {accuracy_score(y_test, y_pred_parallel):.3f}")
```

## 8.5 実践的なワークフロー

### サンプルコード8：完全な機械学習ワークフロー

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import json
from datetime import datetime

class MLWorkflow:
    """完全な機械学習ワークフロー"""
    
    def __init__(self, name="ml_project"):
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def load_and_explore_data(self, X, y):
        """データの読み込みと探索"""
        print("=== データ探索 ===")
        print(f"データ形状: {X.shape}")
        print(f"クラス分布: {np.bincount(y)}")
        
        # 基本統計量
        if isinstance(X, pd.DataFrame):
            print("\n特徴量の統計:")
            print(X.describe())
        
        self.X = X
        self.y = y
        
    def create_preprocessing_pipeline(self):
        """前処理パイプラインの作成"""
        print("\n=== 前処理パイプライン構築 ===")
        
        # 自動的に数値とカテゴリカルを検出（DataFrameの場合）
        if isinstance(self.X, pd.DataFrame):
            numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
            
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        else:
            # NumPy配列の場合
            self.preprocessor = Pipeline([
                ('scaler', StandardScaler())
            ])
        
        print("前処理パイプライン構築完了")
        
    def evaluate_models(self):
        """複数モデルの評価"""
        print("\n=== モデル評価 ===")
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 各モデルの評価
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            # パイプラインの作成
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # 交差検証
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
            
            # テストセットでの評価
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            self.results[name] = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': test_auc,
                'pipeline': pipeline
            }
            
            print(f"\n{name}:")
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"  Test AUC: {test_auc:.3f}")
        
    def hyperparameter_tuning(self, model_name='RandomForest'):
        """ハイパーパラメータチューニング"""
        print(f"\n=== {model_name}のハイパーパラメータチューニング ===")
        
        # パラメータグリッド
        param_grids = {
            'RandomForest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5]
            }
        }
        
        if model_name not in param_grids:
            print(f"{model_name}のパラメータグリッドが定義されていません")
            return
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # グリッドサーチ
        pipeline = self.results[model_name]['pipeline']
        
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n最適パラメータ: {grid_search.best_params_}")
        print(f"最適スコア: {grid_search.best_score_:.3f}")
        
        # 最適モデルの保存
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
    def save_results(self):
        """結果の保存"""
        print("\n=== 結果の保存 ===")
        
        # モデルの保存
        model_filename = f"{self.name}_model_{self.timestamp}.pkl"
        joblib.dump(self.best_model, model_filename)
        print(f"モデルを保存: {model_filename}")
        
        # 結果のJSON保存
        results_to_save = {}
        for name, result in self.results.items():
            results_to_save[name] = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'test_auc': result['test_auc']
            }
        
        results_to_save['best_params'] = self.best_params
        results_to_save['timestamp'] = self.timestamp
        
        results_filename = f"{self.name}_results_{self.timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"結果を保存: {results_filename}")
        
    def run_complete_workflow(self, X, y):
        """完全なワークフローの実行"""
        print(f"=== {self.name} ワークフロー開始 ===\n")
        
        self.load_and_explore_data(X, y)
        self.create_preprocessing_pipeline()
        self.evaluate_models()
        
        # 最良モデルの選択
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['test_auc'])[0]
        print(f"\n最良モデル: {best_model_name}")
        
        self.hyperparameter_tuning(best_model_name)
        self.save_results()
        
        print(f"\n=== ワークフロー完了 ===")
        
        return self.best_model

# ワークフローの実行例
workflow = MLWorkflow(name="classification_project")

# データの準備
X_workflow, y_workflow = make_classification(
    n_samples=2000, n_features=30, n_informative=20,
    n_redundant=10, n_classes=2, random_state=42
)

# DataFrameに変換（実際のプロジェクトを想定）
feature_names = [f'feature_{i}' for i in range(X_workflow.shape[1])]
X_workflow_df = pd.DataFrame(X_workflow, columns=feature_names)

# 完全なワークフローの実行
best_model = workflow.run_complete_workflow(X_workflow_df, y_workflow)

# 新しいデータでの予測例
X_new = X_workflow_df.iloc[:5]
predictions = best_model.predict(X_new)
probabilities = best_model.predict_proba(X_new)[:, 1]

print("\n新しいデータでの予測:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"サンプル {i}: 予測={pred}, 確率={prob:.3f}")
```

## 練習問題

### 問題1：マルチモーダルパイプライン
1. テキストデータと数値データを含むデータセットを作成
2. それぞれに適した前処理を行うパイプラインを構築
3. 特徴量を結合して分類モデルを学習

### 問題2：動的パイプライン
1. データの特性に応じて自動的に前処理を選択するパイプラインを実装
2. 外れ値の有無、分布の偏り等を検出
3. 適切な変換を自動選択

### 問題3：パイプラインの最適化
1. 複数の前処理手法を含むパイプラインを構築
2. 前処理のパラメータも含めてグリッドサーチ
3. 計算時間と性能のトレードオフを分析

## 解答

### 解答1：マルチモーダルパイプライン

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# マルチモーダルデータの生成
np.random.seed(42)
n_samples = 1000

# テキストデータ
text_templates = [
    "This product is {quality} and {price}",
    "I {feeling} this item, it's {quality}",
    "The service was {service} and {speed}",
    "{quality} product with {service} support"
]

quality_words = ['excellent', 'good', 'average', 'poor', 'terrible']
price_words = ['expensive', 'reasonable', 'cheap', 'overpriced']
feeling_words = ['love', 'like', 'dislike', 'hate']
service_words = ['outstanding', 'satisfactory', 'disappointing']
speed_words = ['fast', 'quick', 'slow', 'delayed']

text_data = []
for _ in range(n_samples):
    template = np.random.choice(text_templates)
    text = template.format(
        quality=np.random.choice(quality_words),
        price=np.random.choice(price_words),
        feeling=np.random.choice(feeling_words),
        service=np.random.choice(service_words),
        speed=np.random.choice(speed_words)
    )
    text_data.append(text)

# 数値データ
numeric_data = pd.DataFrame({
    'rating': np.random.randint(1, 6, n_samples),
    'price': np.random.exponential(50, n_samples),
    'delivery_days': np.random.randint(1, 10, n_samples),
    'return_rate': np.random.beta(2, 10, n_samples)
})

# ターゲット（満足度）
# テキストと数値の組み合わせから生成
y_multimodal = ((numeric_data['rating'] >= 4) & 
                (numeric_data['return_rate'] < 0.2) & 
                (['good' in text or 'excellent' in text for text in text_data])).astype(int)

# カスタムトランスフォーマー for マルチモーダルデータ
class MultiModalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=100)
        self.numeric_scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Xは(text_data, numeric_data)のタプル
        text_data, numeric_data = X
        self.text_vectorizer.fit(text_data)
        self.numeric_scaler.fit(numeric_data)
        return self
    
    def transform(self, X):
        text_data, numeric_data = X
        
        # テキストの変換
        text_features = self.text_vectorizer.transform(text_data)
        
        # 数値の変換
        numeric_features = self.numeric_scaler.transform(numeric_data)
        
        # 特徴量の結合
        return hstack([text_features, numeric_features])

# マルチモーダルパイプライン
multimodal_pipeline = Pipeline([
    ('preprocessor', MultiModalTransformer()),
    ('classifier', LogisticRegression(random_state=42))
])

# データの準備
X_multimodal = (text_data, numeric_data)

# 学習と評価（カスタム分割が必要）
from sklearn.model_selection import train_test_split

# インデックスで分割
indices = np.arange(n_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

# 訓練データ
X_train_multi = (
    [text_data[i] for i in train_idx],
    numeric_data.iloc[train_idx]
)
y_train_multi = y_multimodal[train_idx]

# テストデータ
X_test_multi = (
    [text_data[i] for i in test_idx],
    numeric_data.iloc[test_idx]
)
y_test_multi = y_multimodal[test_idx]

# 学習
multimodal_pipeline.fit(X_train_multi, y_train_multi)

# 予測
y_pred_multi = multimodal_pipeline.predict(X_test_multi)

print(f"マルチモーダルパイプラインの精度: {accuracy_score(y_test_multi, y_pred_multi):.3f}")

# 特徴量の重要度分析
# 係数を取得
coefficients = multimodal_pipeline.named_steps['classifier'].coef_[0]

# テキスト特徴量の重要度
text_feature_names = multimodal_pipeline.named_steps['preprocessor'].text_vectorizer.get_feature_names_out()
text_coef = coefficients[:len(text_feature_names)]

# 数値特徴量の重要度
numeric_feature_names = numeric_data.columns.tolist()
numeric_coef = coefficients[len(text_feature_names):]

# 上位の特徴量を表示
print("\n重要なテキスト特徴量（上位10）:")
top_text_idx = np.argsort(np.abs(text_coef))[-10:][::-1]
for idx in top_text_idx:
    print(f"  {text_feature_names[idx]}: {text_coef[idx]:.3f}")

print("\n重要な数値特徴量:")
for name, coef in zip(numeric_feature_names, numeric_coef):
    print(f"  {name}: {coef:.3f}")
```

### 解答2：動的パイプライン

```python
from scipy import stats
from sklearn.preprocessing import PowerTransformer, RobustScaler

class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """データの特性に応じて前処理を自動選択"""
    
    def __init__(self, outlier_threshold=3, skewness_threshold=1):
        self.outlier_threshold = outlier_threshold
        self.skewness_threshold = skewness_threshold
        self.transformers_ = {}
        
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        
        for i in range(self.n_features_):
            feature_data = X[:, i]
            
            # 外れ値の検出
            z_scores = np.abs(stats.zscore(feature_data))
            has_outliers = np.any(z_scores > self.outlier_threshold)
            
            # 歪度の計算
            skewness = stats.skew(feature_data)
            is_skewed = abs(skewness) > self.skewness_threshold
            
            # 適切な変換を選択
            if has_outliers and is_skewed:
                # 外れ値と歪みの両方がある場合
                transformer = Pipeline([
                    ('robust', RobustScaler()),
                    ('power', PowerTransformer())
                ])
                transformer_name = 'robust_power'
            elif has_outliers:
                # 外れ値のみ
                transformer = RobustScaler()
                transformer_name = 'robust'
            elif is_skewed:
                # 歪みのみ
                transformer = PowerTransformer()
                transformer_name = 'power'
            else:
                # 通常の標準化
                transformer = StandardScaler()
                transformer_name = 'standard'
            
            # 変換器を保存
            self.transformers_[i] = {
                'transformer': transformer,
                'name': transformer_name,
                'has_outliers': has_outliers,
                'skewness': skewness
            }
            
            # 各特徴量に対してfitを実行
            transformer.fit(feature_data.reshape(-1, 1))
            
        return self
    
    def transform(self, X):
        X_transformed = np.zeros_like(X)
        
        for i in range(self.n_features_):
            feature_data = X[:, i].reshape(-1, 1)
            transformer = self.transformers_[i]['transformer']
            X_transformed[:, i] = transformer.transform(feature_data).ravel()
            
        return X_transformed
    
    def get_preprocessing_summary(self):
        """前処理の選択結果を表示"""
        summary = pd.DataFrame([
            {
                'Feature': f'Feature_{i}',
                'Transformer': info['name'],
                'Has_Outliers': info['has_outliers'],
                'Skewness': f"{info['skewness']:.2f}"
            }
            for i, info in self.transformers_.items()
        ])
        return summary

# データの生成（様々な分布を含む）
np.random.seed(42)
n_samples = 1000

# 正規分布
feature1 = np.random.normal(0, 1, n_samples)

# 歪んだ分布
feature2 = np.random.exponential(2, n_samples)

# 外れ値を含む
feature3 = np.random.normal(0, 1, n_samples)
outlier_idx = np.random.choice(n_samples, 50, replace=False)
feature3[outlier_idx] += np.random.normal(0, 10, 50)

# 一様分布
feature4 = np.random.uniform(-5, 5, n_samples)

# データの結合
X_dynamic = np.column_stack([feature1, feature2, feature3, feature4])
y_dynamic = (X_dynamic[:, 0] + np.log1p(X_dynamic[:, 1]) - 0.1 * X_dynamic[:, 2] + 
             np.random.normal(0, 0.5, n_samples) > 1).astype(int)

# 動的パイプラインの構築
dynamic_pipeline = Pipeline([
    ('dynamic_preprocessor', DynamicPreprocessor()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# データの分割
X_train_dyn, X_test_dyn, y_train_dyn, y_test_dyn = train_test_split(
    X_dynamic, y_dynamic, test_size=0.3, random_state=42
)

# 学習
dynamic_pipeline.fit(X_train_dyn, y_train_dyn)

# 前処理の選択結果を表示
preprocessor = dynamic_pipeline.named_steps['dynamic_preprocessor']
print("動的に選択された前処理:")
print(preprocessor.get_preprocessing_summary())

# 評価
y_pred_dyn = dynamic_pipeline.predict(X_test_dyn)
print(f"\n動的パイプラインの精度: {accuracy_score(y_test_dyn, y_pred_dyn):.3f}")

# 前処理前後の分布を可視化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # 元のデータ
    axes[0, i].hist(X_train_dyn[:, i], bins=30, alpha=0.7)
    axes[0, i].set_title(f'Feature {i+1} - Original')
    axes[0, i].set_ylabel('Frequency')
    
    # 変換後のデータ
    X_transformed = preprocessor.transform(X_train_dyn)
    axes[1, i].hist(X_transformed[:, i], bins=30, alpha=0.7, color='orange')
    axes[1, i].set_title(f'Feature {i+1} - {preprocessor.transformers_[i]["name"]}')
    axes[1, i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### 解答3：パイプラインの最適化

```python
import time
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 包括的なパイプライン最適化
class OptimizedPipeline:
    """前処理と計算時間を考慮したパイプライン最適化"""
    
    def __init__(self):
        self.results = []
        
    def create_pipeline_variants(self):
        """異なる前処理の組み合わせを持つパイプラインを生成"""
        
        # スケーラーのオプション
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'none': 'passthrough'
        }
        
        # 次元削減のオプション
        dim_reduction = {
            'pca_5': PCA(n_components=5),
            'pca_10': PCA(n_components=10),
            'selectk_5': SelectKBest(f_classif, k=5),
            'selectk_10': SelectKBest(f_classif, k=10),
            'none': 'passthrough'
        }
        
        # 分類器のオプション
        classifiers = {
            'rf_50': RandomForestClassifier(n_estimators=50, random_state=42),
            'rf_100': RandomForestClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        pipelines = {}
        
        # 全ての組み合わせを生成
        for scaler_name, scaler in scalers.items():
            for dim_name, dim_reducer in dim_reduction.items():
                for clf_name, classifier in classifiers.items():
                    pipeline_name = f"{scaler_name}_{dim_name}_{clf_name}"
                    
                    steps = []
                    if scaler != 'passthrough':
                        steps.append(('scaler', scaler))
                    if dim_reducer != 'passthrough':
                        steps.append(('dim_reduction', dim_reducer))
                    steps.append(('classifier', classifier))
                    
                    pipelines[pipeline_name] = Pipeline(steps)
        
        return pipelines
    
    def evaluate_pipelines(self, X, y, cv=5):
        """各パイプラインを評価"""
        pipelines = self.create_pipeline_variants()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"評価するパイプライン数: {len(pipelines)}")
        print("評価中...")
        
        for name, pipeline in pipelines.items():
            # 訓練時間の計測
            start_time = time.time()
            
            # 交差検証
            try:
                cv_scores = cross_val_score(
                    clone(pipeline), X_train, y_train, 
                    cv=cv, scoring='accuracy', n_jobs=1
                )
                
                # テストセットでの評価
                pipeline.fit(X_train, y_train)
                test_score = pipeline.score(X_test, y_test)
                
                train_time = time.time() - start_time
                
                # 予測時間の計測
                start_time = time.time()
                _ = pipeline.predict(X_test)
                predict_time = time.time() - start_time
                
                self.results.append({
                    'pipeline': name,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score,
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'total_time': train_time + predict_time
                })
                
            except Exception as e:
                print(f"エラー in {name}: {str(e)}")
                continue
        
        return pd.DataFrame(self.results)
    
    def plot_optimization_results(self, results_df):
        """最適化結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 性能 vs 訓練時間
        ax = axes[0, 0]
        scatter = ax.scatter(results_df['train_time'], 
                           results_df['test_score'],
                           c=results_df['cv_mean'],
                           s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Test Score')
        ax.set_title('Performance vs Training Time')
        plt.colorbar(scatter, ax=ax, label='CV Score')
        
        # 最適なものをハイライト
        pareto_optimal = self.find_pareto_optimal(results_df)
        ax.scatter(results_df.loc[pareto_optimal, 'train_time'],
                  results_df.loc[pareto_optimal, 'test_score'],
                  color='red', s=200, marker='*', label='Pareto Optimal')
        ax.legend()
        
        # 2. パイプラインコンポーネント別の平均性能
        ax = axes[0, 1]
        
        # スケーラー別
        scaler_performance = {}
        for pipeline_name in results_df['pipeline']:
            scaler = pipeline_name.split('_')[0]
            if scaler not in scaler_performance:
                scaler_performance[scaler] = []
            idx = results_df[results_df['pipeline'] == pipeline_name].index[0]
            scaler_performance[scaler].append(results_df.loc[idx, 'test_score'])
        
        scaler_means = {k: np.mean(v) for k, v in scaler_performance.items()}
        ax.bar(scaler_means.keys(), scaler_means.values())
        ax.set_xlabel('Scaler Type')
        ax.set_ylabel('Mean Test Score')
        ax.set_title('Performance by Scaler Type')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Top 10パイプライン
        ax = axes[1, 0]
        top_10 = results_df.nlargest(10, 'test_score')
        y_pos = np.arange(len(top_10))
        ax.barh(y_pos, top_10['test_score'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_10['pipeline'], fontsize=8)
        ax.set_xlabel('Test Score')
        ax.set_title('Top 10 Pipelines')
        
        # 4. 時間と性能のトレードオフ
        ax = axes[1, 1]
        # 正規化
        norm_score = (results_df['test_score'] - results_df['test_score'].min()) / \
                    (results_df['test_score'].max() - results_df['test_score'].min())
        norm_time = (results_df['total_time'] - results_df['total_time'].min()) / \
                   (results_df['total_time'].max() - results_df['total_time'].min())
        
        # 効率スコア（性能/時間）
        efficiency = norm_score / (norm_time + 0.01)
        results_df['efficiency'] = efficiency
        
        top_efficient = results_df.nlargest(10, 'efficiency')
        ax.scatter(top_efficient['total_time'], 
                  top_efficient['test_score'],
                  s=200, alpha=0.7, label='Most Efficient')
        ax.scatter(results_df['total_time'], 
                  results_df['test_score'],
                  alpha=0.3, s=50)
        ax.set_xlabel('Total Time (seconds)')
        ax.set_ylabel('Test Score')
        ax.set_title('Efficiency Analysis')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    def find_pareto_optimal(self, results_df):
        """パレート最適なパイプラインを見つける"""
        pareto_optimal = []
        
        for idx, row in results_df.iterrows():
            # このパイプラインが他のすべてに対して劣っていないかチェック
            is_pareto = True
            for idx2, row2 in results_df.iterrows():
                if idx != idx2:
                    # より良い性能かつより速い場合
                    if (row2['test_score'] > row['test_score'] and 
                        row2['total_time'] < row['total_time']):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_optimal.append(idx)
        
        return pareto_optimal

# パイプライン最適化の実行
optimizer = OptimizedPipeline()

# データの準備
X_opt, y_opt = make_classification(
    n_samples=1000, n_features=30, n_informative=20,
    n_redundant=10, random_state=42
)

# 評価の実行
results_df = optimizer.evaluate_pipelines(X_opt, y_opt, cv=3)

print("\n評価完了！")
print(f"評価したパイプライン数: {len(results_df)}")

# 結果の表示
print("\nTop 5 パイプライン（性能）:")
print(results_df.nlargest(5, 'test_score')[['pipeline', 'test_score', 'total_time']])

print("\nTop 5 パイプライン（効率性）:")
print(results_df.nlargest(5, 'efficiency')[['pipeline', 'test_score', 'total_time', 'efficiency']])

# 可視化
optimizer.plot_optimization_results(results_df)

# 最適なパイプラインの詳細分析
best_pipeline_name = results_df.loc[results_df['test_score'].idxmax(), 'pipeline']
print(f"\n最高性能パイプライン: {best_pipeline_name}")
print(f"構成: {best_pipeline_name.split('_')}")

best_efficient_name = results_df.loc[results_df['efficiency'].idxmax(), 'pipeline']
print(f"\n最高効率パイプライン: {best_efficient_name}")
print(f"構成: {best_efficient_name.split('_')}")
```

## まとめ

この章では、パイプラインとワークフローについて学習しました：

- **基本的なパイプライン**: 前処理とモデルの統合
- **ColumnTransformer**: 異なる型の特徴量の処理
- **カスタムトランスフォーマー**: 独自の前処理ステップの実装
- **FeatureUnion**: 複数の特徴量抽出手法の結合
- **パイプラインの最適化**: ハイパーパラメータチューニングと効率性
- **実践的なワークフロー**: プロジェクト全体の自動化

これで、scikit-learnの主要な機能をすべてカバーしました。これらの技術を組み合わせることで、実際のデータサイエンスプロジェクトに対応できます。