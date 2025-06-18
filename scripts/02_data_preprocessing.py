#!/usr/bin/env python3
"""
データ前処理の包括的なサンプルスクリプト
欠損値処理、スケーリング、カテゴリカル変数の処理を示します

【このスクリプトで学べること】
1. 欠損値の処理方法（平均値、中央値、最頻値での補完）
2. データのスケーリング（StandardScaler、MinMaxScaler）
3. カテゴリカル変数のエンコーディング（OneHotEncoder）
4. 前処理パイプラインの構築
5. ColumnTransformerを使った効率的な前処理
"""

# 必要なライブラリのインポート
import numpy as np  # 数値計算用
import pandas as pd  # データフレーム操作用
import matplotlib.pyplot as plt  # グラフ描画用
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder  # 前処理用
from sklearn.impute import SimpleImputer  # 欠損値処理用
from sklearn.compose import ColumnTransformer  # 列ごとの前処理を統合
from sklearn.pipeline import Pipeline  # 処理をパイプライン化
from sklearn.model_selection import train_test_split  # データ分割用
from sklearn.ensemble import RandomForestClassifier  # 分類モデル
from sklearn.metrics import accuracy_score  # 評価指標
import seaborn as sns  # 高度な可視化（インポート追加）

def create_sample_data():
    """
    サンプルデータの作成（欠損値あり）
    実際のデータでよくある状況を再現：
    - 数値データに欠損値
    - カテゴリカルデータ
    - 複数の特徴量
    """
    np.random.seed(42)  # 結果を再現可能にする
    n_samples = 1000  # サンプル数
    
    # 年齢データ（一部欠損値あり）
    # 実際のデータでは、アンケートの未回答などで欠損値が発生
    age = np.random.randint(18, 80, n_samples).astype(float)
    age[np.random.rand(n_samples) < 0.1] = np.nan  # 10%を欠損値に
    
    # 収入データ（一部欠損値あり）
    # 正規分布に従う収入データを生成
    income = np.random.normal(50000, 20000, n_samples)
    income[np.random.rand(n_samples) < 0.05] = np.nan  # 5%を欠損値に
    
    # カテゴリカルデータ（教育レベル）
    # 機械学習では文字列データをそのまま扱えないため、変換が必要
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    
    # カテゴリカルデータ（都市）
    city = np.random.choice(['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka'], n_samples)
    
    # ターゲット（予測したい値）
    # ここでは2値分類（0 or 1）
    target = np.random.choice([0, 1], n_samples)
    
    # pandasのDataFrameに変換（実際のデータ分析でよく使う形式）
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'city': city,
        'target': target
    })
    
    return df

def demonstrate_scaling():
    """
    スケーリング手法の比較
    異なるスケールの特徴量を同じスケールに変換する重要性を示す
    """
    print("=== スケーリング手法の比較 ===\n")
    
    # 異なるスケールのデータを生成
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 20, 100),    # 平均100、標準偏差20
        'feature2': np.random.exponential(2, 100),     # 指数分布（偏ったデータ）
        'feature3': np.random.uniform(0, 1, 100)       # 0〜1の一様分布
    })
    
    # 元のデータの統計量を表示
    print("元のデータの統計量:")
    print(data.describe())
    print()
    
    # ============================================================
    # StandardScaler: 平均0、標準偏差1に変換
    # ============================================================
    # 最も一般的なスケーリング手法
    # 外れ値の影響を受けやすいが、正規分布に近いデータに適している
    scaler_std = StandardScaler()
    data_std = pd.DataFrame(
        scaler_std.fit_transform(data),
        columns=[f'{col}_std' for col in data.columns]
    )
    
    # ============================================================
    # MinMaxScaler: 最小値0、最大値1に変換
    # ============================================================
    # データを[0, 1]の範囲に収める
    # 外れ値の影響を受けやすいが、データの分布を保持
    scaler_minmax = MinMaxScaler()
    data_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(data),
        columns=[f'{col}_minmax' for col in data.columns]
    )
    
    # 可視化で違いを比較
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元のデータ（スケールがバラバラ）
    data.boxplot(ax=axes[0])
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Value')
    
    # StandardScaler適用後（平均0、標準偏差1）
    data_std.boxplot(ax=axes[1])
    axes[1].set_title('StandardScaler')
    axes[1].set_ylabel('Value')
    
    # MinMaxScaler適用後（0〜1の範囲）
    data_minmax.boxplot(ax=axes[2])
    axes[2].set_title('MinMaxScaler')
    axes[2].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png')
    print("スケーリングの比較を 'scaling_comparison.png' として保存しました\n")

def demonstrate_preprocessing_pipeline():
    """
    前処理パイプラインの実装
    実際のプロジェクトで使える、本格的な前処理の流れを示す
    """
    print("=== 前処理パイプラインの実装 ===\n")
    
    # サンプルデータの作成
    df = create_sample_data()
    
    # データの基本情報を表示
    print("データの概要:")
    print(df.info())
    print("\n欠損値の数:")
    print(df.isnull().sum())
    print()
    
    # 特徴量とターゲットの分離
    # X: 入力データ（特徴量）
    # y: 出力データ（予測したい値）
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数値とカテゴリカルの列を特定
    # それぞれ異なる前処理が必要
    numeric_features = ['age', 'income']
    categorical_features = ['education', 'city']
    
    # ============================================================
    # 数値データの前処理パイプライン
    # ============================================================
    numeric_transformer = Pipeline(steps=[
        # Step 1: 欠損値を中央値で補完
        # 中央値は外れ値の影響を受けにくい
        ('imputer', SimpleImputer(strategy='median')),
        # Step 2: 標準化（平均0、標準偏差1）
        ('scaler', StandardScaler())
    ])
    
    # ============================================================
    # カテゴリカルデータの前処理パイプライン
    # ============================================================
    categorical_transformer = Pipeline(steps=[
        # Step 1: 欠損値を'missing'という文字列で補完
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # Step 2: One-Hot Encoding（ダミー変数化）
        # 例: 'Tokyo' → [1, 0, 0, 0], 'Osaka' → [0, 1, 0, 0]
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ============================================================
    # ColumnTransformerで統合
    # ============================================================
    # 数値とカテゴリカルで異なる処理を適用
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),      # 数値列の処理
            ('cat', categorical_transformer, categorical_features)  # カテゴリカル列の処理
        ])
    
    # ============================================================
    # 完全なパイプライン（前処理 + モデル）
    # ============================================================
    # これにより、前処理とモデル学習を一連の流れで実行できる
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,  # 決定木の数
            random_state=42    # 結果を再現可能にする
        ))
    ])
    
    # データの分割（訓練用70%、テスト用30%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # モデルの学習
    # パイプラインが自動的に前処理→学習を実行
    print("モデルの学習中...")
    clf.fit(X_train, y_train)
    print("学習完了！\n")
    
    # 予測と評価
    # テストデータも自動的に前処理される
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"テストデータの精度: {accuracy:.3f}")
    
    # 前処理後のデータ形状を確認
    X_transformed = preprocessor.fit_transform(X_train)
    print(f"\n前処理後のデータ形状: {X_transformed.shape}")
    print("（元の4列から、One-Hot Encodingにより列数が増加）")

def demonstrate_missing_values():
    """
    欠損値処理の実演
    様々な補完方法の違いを視覚的に示す
    """
    print("\n=== 欠損値処理の実演 ===\n")
    
    # 欠損値を含む小さなデータを作成（視覚化しやすいサイズ）
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
    
    # 様々な補完方法を試す
    strategies = [
        'mean',          # 平均値で補完
        'median',        # 中央値で補完（外れ値に強い）
        'most_frequent', # 最頻値で補完
        'constant'       # 定数（0）で補完
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, strategy in enumerate(strategies):
        # 補完方法に応じてImputerを設定
        if strategy == 'constant':
            imputer = SimpleImputer(strategy=strategy, fill_value=0)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        # 欠損値を補完
        data_imputed = pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
        
        # ヒートマップで可視化（値の大きさを色で表現）
        ax = axes[idx]
        sns.heatmap(data_imputed, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'Imputation Strategy: {strategy}')
    
    plt.tight_layout()
    plt.savefig('missing_value_imputation.png')
    print("\n欠損値処理の比較を 'missing_value_imputation.png' として保存しました")

def main():
    """
    メイン実行関数
    各種前処理手法を順番に実行
    """
    # 1. スケーリング手法の比較
    demonstrate_scaling()
    
    # 2. 前処理パイプラインの実装
    demonstrate_preprocessing_pipeline()
    
    # 3. 欠損値処理の実演
    demonstrate_missing_values()
    
    # 前処理のベストプラクティスをまとめて表示
    print("\n=== 前処理のベストプラクティス ===")
    print("1. 常にtrain/testを分けてから前処理を行う")
    print("   → テストデータの情報が訓練に漏れるのを防ぐ（データリーク防止）")
    print("2. fitはtrainデータのみで行い、testデータにはtransformを適用")
    print("   → テストデータは「未知のデータ」として扱う")
    print("3. パイプラインを使用して前処理とモデルを統合")
    print("   → コードが簡潔になり、ミスも減る")
    print("4. ColumnTransformerで異なる型の特徴量を適切に処理")
    print("   → 数値とカテゴリカルで異なる処理を自動化")

# ============================================================
# メイン実行部
# ============================================================
if __name__ == "__main__":
    main()