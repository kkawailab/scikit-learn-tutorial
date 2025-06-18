#!/usr/bin/env python3
"""
基本的な分類問題のサンプルスクリプト
scikit-learnを使用した分類の基本的な流れを示します
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def main():
    # 1. データの読み込み
    print("=== Irisデータセットを使用した分類 ===\n")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"データの形状: {X.shape}")
    print(f"クラス数: {len(iris.target_names)}")
    print(f"クラス名: {iris.target_names}")
    print(f"特徴量名: {iris.feature_names}\n")
    
    # 2. データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}\n")
    
    # 3. データの標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. モデルの学習
    print("ロジスティック回帰モデルの学習中...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("学習完了！\n")
    
    # 5. 予測
    y_pred = model.predict(X_test_scaled)
    
    # 6. 評価
    accuracy = accuracy_score(y_test, y_pred)
    print(f"精度: {accuracy:.3f}\n")
    
    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 7. 混同行列の可視化
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n混同行列を 'confusion_matrix.png' として保存しました")
    
    # 8. 予測確率の例
    print("\n=== 予測確率の例 ===")
    sample_idx = 0
    sample_proba = model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])
    print(f"サンプル {sample_idx} の実際のクラス: {iris.target_names[y_test[sample_idx]]}")
    print("予測確率:")
    for i, class_name in enumerate(iris.target_names):
        print(f"  {class_name}: {sample_proba[0][i]:.3f}")

if __name__ == "__main__":
    main()