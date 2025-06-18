#!/usr/bin/env python3
"""
データ前処理の包括的なサンプルスクリプト
欠損値処理、スケーリング、カテゴリカル変数の処理を示します
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def create_sample_data():
    """サンプルデータの作成（欠損値あり）"""
    np.random.seed(42)
    n_samples = 1000
    
    # 数値データ（一部欠損値）
    age = np.random.randint(18, 80, n_samples).astype(float)
    age[np.random.rand(n_samples) < 0.1] = np.nan
    
    income = np.random.normal(50000, 20000, n_samples)
    income[np.random.rand(n_samples) < 0.05] = np.nan
    
    # カテゴリカルデータ
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    city = np.random.choice(['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka'], n_samples)
    
    # ターゲット
    target = np.random.choice([0, 1], n_samples)
    
    # DataFrameの作成
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'city': city,
        'target': target
    })
    
    return df

def demonstrate_scaling():
    """スケーリング手法の比較"""
    print("=== スケーリング手法の比較 ===\n")
    
    # データの生成
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 20, 100),
        'feature2': np.random.exponential(2, 100),
        'feature3': np.random.uniform(0, 1, 100)
    })
    
    # 元のデータの統計量
    print("元のデータの統計量:")
    print(data.describe())
    print()
    
    # StandardScaler
    scaler_std = StandardScaler()
    data_std = pd.DataFrame(
        scaler_std.fit_transform(data),
        columns=[f'{col}_std' for col in data.columns]
    )
    
    # MinMaxScaler
    scaler_minmax = MinMaxScaler()
    data_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(data),
        columns=[f'{col}_minmax' for col in data.columns]
    )
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元のデータ
    data.boxplot(ax=axes[0])
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Value')
    
    # StandardScaler
    data_std.boxplot(ax=axes[1])
    axes[1].set_title('StandardScaler')
    axes[1].set_ylabel('Value')
    
    # MinMaxScaler
    data_minmax.boxplot(ax=axes[2])
    axes[2].set_title('MinMaxScaler')
    axes[2].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png')
    print("スケーリングの比較を 'scaling_comparison.png' として保存しました\n")

def demonstrate_preprocessing_pipeline():
    """前処理パイプラインの実装"""
    print("=== 前処理パイプラインの実装 ===\n")
    
    # データの作成
    df = create_sample_data()
    
    print("データの概要:")
    print(df.info())
    print("\n欠損値の数:")
    print(df.isnull().sum())
    print()
    
    # 特徴量とターゲットの分離
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数値とカテゴリカルの列を特定
    numeric_features = ['age', 'income']
    categorical_features = ['education', 'city']
    
    # 数値データの前処理パイプライン
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 欠損値を中央値で補完
        ('scaler', StandardScaler())  # 標準化
    ])
    
    # カテゴリカルデータの前処理パイプライン
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
    
    # 完全なパイプライン（前処理 + モデル）
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # モデルの学習
    print("モデルの学習中...")
    clf.fit(X_train, y_train)
    print("学習完了！\n")
    
    # 予測と評価
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"テストデータの精度: {accuracy:.3f}")
    
    # 前処理後のデータ形状
    X_transformed = preprocessor.fit_transform(X_train)
    print(f"\n前処理後のデータ形状: {X_transformed.shape}")
    print("（元の4列から、One-Hot Encodingにより列数が増加）")

def demonstrate_missing_values():
    """欠損値処理の実演"""
    print("\n=== 欠損値処理の実演 ===\n")
    
    # 欠損値を含むデータの作成
    np.random.seed(42)
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, np.nan, 7],
        'B': [10, np.nan, 30, np.nan, 50, 60, 70],
        'C': [100, 200, 300, 400, np.nan, np.nan, 700]
    })
    
    print("元のデータ:")
    print(data)
    print("\n欠損値の数:")
    print(data.isnull().sum())
    
    # 様々な補完方法
    strategies = ['mean', 'median', 'most_frequent', 'constant']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, strategy in enumerate(strategies):
        if strategy == 'constant':
            imputer = SimpleImputer(strategy=strategy, fill_value=0)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        data_imputed = pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
        
        # ヒートマップで可視化
        ax = axes[idx]
        sns.heatmap(data_imputed, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'Imputation Strategy: {strategy}')
    
    plt.tight_layout()
    plt.savefig('missing_value_imputation.png')
    print("\n欠損値処理の比較を 'missing_value_imputation.png' として保存しました")

def main():
    # 1. スケーリング手法の比較
    demonstrate_scaling()
    
    # 2. 前処理パイプラインの実装
    demonstrate_preprocessing_pipeline()
    
    # 3. 欠損値処理の実演
    demonstrate_missing_values()
    
    print("\n=== 前処理のベストプラクティス ===")
    print("1. 常にtrain/testを分けてから前処理を行う")
    print("2. fitはtrainデータのみで行い、testデータにはtransformを適用")
    print("3. パイプラインを使用して前処理とモデルを統合")
    print("4. ColumnTransformerで異なる型の特徴量を適切に処理")

if __name__ == "__main__":
    main()