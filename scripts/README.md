# scikit-learn サンプルスクリプト集

このディレクトリには、scikit-learnの主要な機能を実演するサンプルスクリプトが含まれています。

## スクリプト一覧

### 1. 01_basic_classification.py
**基本的な分類問題**
- Irisデータセットを使用した分類
- データの読み込み、前処理、学習、評価の基本的な流れ
- 混同行列の可視化
- 予測確率の取得

### 2. 02_data_preprocessing.py
**データ前処理**
- 欠損値の処理（SimpleImputer）
- データのスケーリング（StandardScaler、MinMaxScaler）
- カテゴリカル変数の処理（OneHotEncoder）
- 前処理パイプラインの構築

### 3. 03_regression_analysis.py
**回帰分析**
- 線形回帰の基本
- 正則化（Ridge、Lasso、ElasticNet）
- 多項式回帰
- ランダムフォレスト回帰
- 残差分析

### 4. 04_clustering_examples.py
**クラスタリング**
- K-meansクラスタリング（エルボー法）
- 階層的クラスタリング（樹形図）
- DBSCAN（密度ベースクラスタリング）
- 異なるデータ形状での比較
- 高次元データのクラスタリング

### 5. 05_model_evaluation.py
**モデル評価と改善**
- 交差検証
- グリッドサーチとランダムサーチ
- 学習曲線と検証曲線
- 特徴量選択
- ROC曲線とPR曲線

## 実行方法

各スクリプトは独立して実行可能です：

```bash
python 01_basic_classification.py
```

## 必要なライブラリ

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## 出力ファイル

各スクリプトは実行時に以下のような画像ファイルを生成します：
- confusion_matrix.png
- scaling_comparison.png
- learning_curves.png
- clustering_comparison.png
- など

## 学習の進め方

1. まず `01_basic_classification.py` で基本的な流れを理解
2. `02_data_preprocessing.py` で前処理の重要性を学習
3. タスクに応じて分類（01）、回帰（03）、クラスタリング（04）を選択
4. `05_model_evaluation.py` でモデルの改善方法を習得

## カスタマイズ

各スクリプトのパラメータを変更して実験してみてください：
- データセットのサイズ
- モデルのハイパーパラメータ
- 可視化の方法
- 評価指標

## トラブルシューティング

エラーが発生した場合：
1. 必要なライブラリがインストールされているか確認
2. Pythonのバージョンが3.6以上か確認
3. データセットのダウンロードに失敗していないか確認