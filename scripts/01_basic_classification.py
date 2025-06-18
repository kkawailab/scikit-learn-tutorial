#!/usr/bin/env python3
"""
基本的な分類問題のサンプルスクリプト
scikit-learnを使用した分類の基本的な流れを示します

【このスクリプトで学べること】
1. 機械学習の基本的な流れ（データ読み込み→前処理→学習→評価）
2. 分類問題の解き方
3. モデルの評価方法
4. 結果の可視化方法
"""

# 必要なライブラリのインポート
import numpy as np  # 数値計算用ライブラリ
import matplotlib.pyplot as plt  # グラフ描画用ライブラリ
from sklearn.datasets import load_iris  # サンプルデータセット読み込み用
from sklearn.model_selection import train_test_split  # データ分割用
from sklearn.preprocessing import StandardScaler  # データ標準化用
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデル
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 評価指標
import seaborn as sns  # 高度な可視化ライブラリ

def main():
    # ============================================================
    # 1. データの読み込み
    # ============================================================
    # Irisデータセットは、アヤメの花の特徴量から品種を分類する有名なデータセット
    # 初心者が機械学習を学ぶのに最適なサンプルデータです
    print("=== Irisデータセットを使用した分類 ===\n")
    
    # load_iris()でデータセットを読み込み
    iris = load_iris()
    
    # X: 特徴量（花びらの長さ・幅、がく片の長さ・幅の4つ）
    # 形状: (150サンプル, 4特徴量)
    X = iris.data
    
    # y: ターゲット（正解ラベル）- 3種類のアヤメの品種
    # 0: setosa, 1: versicolor, 2: virginica
    y = iris.target
    
    # データの基本情報を表示
    print(f"データの形状: {X.shape}")  # (サンプル数, 特徴量数)
    print(f"クラス数: {len(iris.target_names)}")  # 分類するクラスの数
    print(f"クラス名: {iris.target_names}")  # 各クラスの名前
    print(f"特徴量名: {iris.feature_names}\n")  # 各特徴量の名前
    
    # ============================================================
    # 2. データの分割
    # ============================================================
    # 機械学習では、データを「訓練用」と「テスト用」に分けることが重要
    # 訓練用: モデルの学習に使用
    # テスト用: 学習したモデルの性能評価に使用（モデルが見たことのないデータ）
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,  # 分割するデータとラベル
        test_size=0.3,  # テストデータの割合（30%）
        random_state=42,  # 乱数シード（結果を再現可能にする）
        stratify=y  # 各クラスの比率を保ったまま分割（重要！）
    )
    
    print(f"訓練データ: {X_train.shape}")  # 70%のデータ
    print(f"テストデータ: {X_test.shape}\n")  # 30%のデータ
    
    # ============================================================
    # 3. データの標準化（前処理）
    # ============================================================
    # 特徴量のスケールを揃えることで、モデルの学習が安定します
    # 標準化: 各特徴量を平均0、標準偏差1に変換
    
    scaler = StandardScaler()
    
    # fit_transform: 訓練データで統計量（平均・標準偏差）を計算し、変換
    X_train_scaled = scaler.fit_transform(X_train)
    
    # transform: 訓練データの統計量を使ってテストデータを変換
    # 注意: テストデータでfitしないこと！（データリークを防ぐ）
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 4. モデルの学習
    # ============================================================
    # ロジスティック回帰: 分類問題でよく使われるシンプルで解釈しやすいモデル
    # 線形回帰を分類問題に拡張したもの
    
    print("ロジスティック回帰モデルの学習中...")
    
    # モデルのインスタンスを作成
    model = LogisticRegression(
        max_iter=1000,  # 最大反復回数（収束するまでの計算回数）
        random_state=42  # 乱数シード（結果を再現可能にする）
    )
    
    # fit()メソッドで学習を実行
    # 訓練データの特徴量とラベルから、パターンを学習
    model.fit(X_train_scaled, y_train)
    print("学習完了！\n")
    
    # ============================================================
    # 5. 予測
    # ============================================================
    # 学習したモデルを使って、テストデータのクラスを予測
    # predict()メソッドは、各サンプルが属するクラスを返す
    y_pred = model.predict(X_test_scaled)
    
    # ============================================================
    # 6. 評価
    # ============================================================
    # モデルの性能を様々な指標で評価
    
    # 精度（Accuracy）: 全体の予測のうち、正解した割合
    accuracy = accuracy_score(y_test, y_pred)
    print(f"精度: {accuracy:.3f}\n")  # 小数点以下3桁で表示
    
    # 分類レポート: クラスごとの詳細な評価指標
    # - precision（適合率）: 予測が正解だった割合
    # - recall（再現率）: 実際の正解を予測できた割合
    # - f1-score: precisionとrecallの調和平均
    # - support: 各クラスのサンプル数
    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # ============================================================
    # 7. 混同行列の可視化
    # ============================================================
    # 混同行列（Confusion Matrix）: 予測結果を視覚的に理解するための表
    # 縦軸: 実際のクラス、横軸: 予測されたクラス
    # 対角線上の数値が大きいほど、正しく分類できている
    
    cm = confusion_matrix(y_test, y_pred)
    
    # ヒートマップとして可視化
    plt.figure(figsize=(8, 6))  # 図のサイズを設定
    sns.heatmap(
        cm,  # 混同行列のデータ
        annot=True,  # 各セルに数値を表示
        fmt='d',  # 整数で表示
        cmap='Blues',  # カラーマップ（青系）
        xticklabels=iris.target_names,  # x軸のラベル
        yticklabels=iris.target_names   # y軸のラベル
    )
    plt.ylabel('True Label')  # y軸のタイトル
    plt.xlabel('Predicted Label')  # x軸のタイトル
    plt.title('Confusion Matrix')  # グラフのタイトル
    plt.tight_layout()  # レイアウトを自動調整
    plt.savefig('confusion_matrix.png')  # 画像として保存
    print("\n混同行列を 'confusion_matrix.png' として保存しました")
    
    # ============================================================
    # 8. 予測確率の例
    # ============================================================
    # predict_proba()メソッド: 各クラスに属する確率を返す
    # これにより、モデルの「自信度」がわかる
    
    print("\n=== 予測確率の例 ===")
    sample_idx = 0  # 最初のテストサンプルを例として使用
    
    # 1つのサンプルに対する予測確率を取得
    # 注意: 入力は2次元配列である必要があるため、[sample_idx:sample_idx+1]とする
    sample_proba = model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])
    
    print(f"サンプル {sample_idx} の実際のクラス: {iris.target_names[y_test[sample_idx]]}")
    print("予測確率:")
    
    # 各クラスの予測確率を表示
    for i, class_name in enumerate(iris.target_names):
        print(f"  {class_name}: {sample_proba[0][i]:.3f}")
    
    # 補足: 最も高い確率のクラスが、predict()メソッドの予測結果になる

# ============================================================
# メイン実行部
# ============================================================
# このスクリプトが直接実行された場合のみ、main()関数を呼び出す
# モジュールとしてインポートされた場合は実行されない
if __name__ == "__main__":
    main()