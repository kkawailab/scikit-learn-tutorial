#!/usr/bin/env python3
"""
回帰分析の包括的なサンプルスクリプト
線形回帰、正則化、非線形回帰を含む

【このスクリプトで学べること】
1. 線形回帰の基本と残差分析
2. 正則化手法（Ridge、Lasso、ElasticNet）の比較
3. 多項式回帰による非線形パターンの学習
4. ランダムフォレスト回帰と特徴量の重要度
5. 交差検証による適切なモデル評価
"""

# 必要なライブラリのインポート
import numpy as np  # 数値計算用
import pandas as pd  # データフレーム操作用
import matplotlib.pyplot as plt  # グラフ描画用
from sklearn.model_selection import train_test_split, cross_val_score  # データ分割と交差検証
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # 線形回帰モデル群
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # 前処理用
from sklearn.pipeline import make_pipeline  # パイプライン構築用
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 評価指標
from sklearn.ensemble import RandomForestRegressor  # アンサンブル学習
import warnings
warnings.filterwarnings('ignore')  # 警告メッセージを非表示

def generate_synthetic_data():
    """
    合成データの生成
    線形データと非線形データの両方を作成して、
    異なる回帰手法の特性を理解しやすくする
    """
    np.random.seed(42)  # 結果を再現可能にする
    n_samples = 200    # サンプル数
    
    # ============================================================
    # 線形データの生成（y = 2x + 1 + ノイズ）
    # ============================================================
    # 単純な線形関係を持つデータ
    X_linear = np.random.rand(n_samples, 1) * 10  # 0〜10の範囲の特徴量
    y_linear = 2 * X_linear.squeeze() + 1 + np.random.randn(n_samples) * 2  # 真の関係式 + ノイズ
    
    # ============================================================
    # 非線形データの生成（サイン波 + ノイズ）
    # ============================================================
    # 線形回帰では捉えられない複雑なパターン
    X_nonlinear = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)  # ソートして滑らかに
    y_nonlinear = np.sin(X_nonlinear).squeeze() * 10 + np.random.randn(n_samples) * 2
    
    return X_linear, y_linear, X_nonlinear, y_nonlinear

def demonstrate_linear_regression(X, y):
    """
    基本的な線形回帰の実演
    最小二乗法による回帰直線の当てはめと残差分析
    """
    print("=== 線形回帰の実演 ===\n")
    
    # データの分割（訓練用70%、テスト用30%）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ============================================================
    # モデルの学習
    # ============================================================
    # LinearRegression: 最小二乗法で回帰係数を求める
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # 訓練データで学習
    
    # 予測
    y_pred_train = lr.predict(X_train)  # 訓練データに対する予測
    y_pred_test = lr.predict(X_test)    # テストデータに対する予測
    
    # ============================================================
    # 評価指標の計算
    # ============================================================
    # R²スコア: 決定係数（1に近いほど良い）
    # RMSE: 二乗平均平方根誤差（小さいほど良い）
    print(f"訓練データ R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"テストデータ R²: {r2_score(y_test, y_pred_test):.3f}")
    print(f"テストデータ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
    print(f"係数: {lr.coef_[0]:.3f}")      # 回帰直線の傾き
    print(f"切片: {lr.intercept_:.3f}")    # 回帰直線のy切片
    
    # ============================================================
    # 結果の可視化
    # ============================================================
    plt.figure(figsize=(12, 5))
    
    # 左図: 予測値のプロット
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, alpha=0.5, label='Actual')  # 実際の値
    plt.plot(X_test, y_pred_test, 'r-', linewidth=2, label='Predicted')  # 予測直線
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression: Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右図: 残差プロット（予測の偏りをチェック）
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_test  # 残差 = 実際の値 - 予測値
    plt.scatter(y_pred_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')  # y=0の基準線
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # 理想的な残差プロット: ランダムに0周辺に分布
    # パターンがある場合: モデルが何かを見逃している可能性
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png')
    print("\n結果を 'linear_regression_results.png' として保存しました\n")

def demonstrate_regularization():
    """
    正則化手法の比較
    過学習を防ぎ、多重共線性に対処する方法を示す
    """
    print("=== 正則化手法の比較 ===\n")
    
    # ============================================================
    # 多重共線性のあるデータを生成
    # ============================================================
    # 特徴量同士が相関している状況を再現
    np.random.seed(42)
    n_samples = 100
    n_features = 20  # 特徴量を多くして過学習しやすい状況を作る
    
    X = np.random.randn(n_samples, n_features)
    # いくつかの特徴量を意図的に相関させる
    for i in range(5):
        X[:, i+5] = X[:, i] + np.random.randn(n_samples) * 0.5
    
    # 真の係数（スパース = ほとんどが0）
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3, -2, 1, 0, -1]  # 最初の5つだけが重要
    y = X @ true_coef + np.random.randn(n_samples) * 0.5  # @ は行列積
    
    # データの分割と標準化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 正則化では標準化が重要（各特徴量のスケールを揃える）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 各正則化手法の比較
    # ============================================================
    models = {
        'Linear Regression': LinearRegression(),          # 通常の線形回帰（正則化なし）
        'Ridge (α=1.0)': Ridge(alpha=1.0),               # L2正則化（係数を小さくする）
        'Lasso (α=0.1)': Lasso(alpha=0.1),               # L1正則化（係数を0にする）
        'ElasticNet (α=0.1)': ElasticNet(alpha=0.1)      # L1とL2の組み合わせ
    }
    
    results = []
    coefficients = {}
    
    for name, model in models.items():
        # モデルの学習
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # 結果を記録
        results.append({
            'Model': name,
            'Train R²': model.score(X_train_scaled, y_train),
            'Test R²': model.score(X_test_scaled, y_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Non-zero Coefs': np.sum(np.abs(model.coef_) > 1e-5)  # 0でない係数の数
        })
        
        coefficients[name] = model.coef_
    
    # 結果の表示
    results_df = pd.DataFrame(results)
    print(results_df.round(3))
    
    # ============================================================
    # 係数の可視化
    # ============================================================
    # 各手法がどのように係数を推定するかを比較
    plt.figure(figsize=(12, 8))
    
    for i, (name, coef) in enumerate(coefficients.items()):
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(coef)), coef)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title(name)
        plt.grid(True, alpha=0.3)
    
    # 観察ポイント:
    # - Linear Regression: 過学習により係数が大きくなりがち
    # - Ridge: すべての係数を小さくする（0にはしない）
    # - Lasso: 不要な係数を0にする（特徴選択効果）
    # - ElasticNet: RidgeとLassoの中間的な性質
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png')
    print("\n正則化の比較を 'regularization_comparison.png' として保存しました\n")

def demonstrate_polynomial_regression(X, y):
    """
    多項式回帰の実演
    非線形パターンを線形モデルで学習する方法
    """
    print("=== 多項式回帰の実演 ===\n")
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 異なる次数で比較（1次 = 線形、高次 = より複雑）
    degrees = [1, 3, 5, 9]
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # ============================================================
        # 多項式回帰パイプライン
        # ============================================================
        # PolynomialFeatures: x → [1, x, x², x³, ...]に変換
        # LinearRegression: 変換後の特徴量で線形回帰
        poly_model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        # 学習
        poly_model.fit(X_train, y_train)
        
        # 滑らかな予測曲線を描くためのデータ
        X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_plot = poly_model.predict(X_plot)
        
        # 評価
        train_score = poly_model.score(X_train, y_train)
        test_score = poly_model.score(X_test, y_test)
        
        # プロット
        plt.subplot(2, 2, i+1)
        plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
        plt.scatter(X_test, y_test, alpha=0.5, label='Test data')
        plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Degree {degree}: Train R²={train_score:.3f}, Test R²={test_score:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 観察ポイント:
    # - 低次（1,3）: アンダーフィッティング（単純すぎる）
    # - 適切な次数: データのパターンをうまく捉える
    # - 高次（9）: オーバーフィッティング（複雑すぎる）
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_comparison.png')
    print("多項式回帰の比較を 'polynomial_regression_comparison.png' として保存しました\n")

def demonstrate_ensemble_regression():
    """
    アンサンブル回帰の実演
    ランダムフォレストによる非線形回帰と特徴量の重要度分析
    """
    print("=== ランダムフォレスト回帰の実演 ===\n")
    
    # ============================================================
    # 現実的なデータの生成（住宅価格予測風）
    # ============================================================
    np.random.seed(42)
    n_samples = 500
    
    # 特徴量の生成
    data = pd.DataFrame({
        'rooms': np.random.normal(6, 2, n_samples),        # 部屋数
        'area': np.random.normal(100, 30, n_samples),      # 面積
        'age': np.random.randint(0, 50, n_samples),        # 築年数
        'distance': np.random.exponential(3, n_samples),   # 駅からの距離
        'crime_rate': np.random.exponential(2, n_samples)  # 犯罪率
    })
    
    # 価格の生成（非線形な関係を含む）
    price = (
        50 * data['rooms'] +
        0.5 * data['area'] +
        -2 * data['age'] +
        -10 * data['distance'] +
        -5 * data['crime_rate'] +
        20 * np.sin(data['rooms']) +  # 非線形成分（部屋数による非線形効果）
        np.random.normal(0, 20, n_samples)  # ノイズ
    )
    
    X = data.values
    y = price
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ============================================================
    # モデルの比較
    # ============================================================
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,  # 決定木の数
            random_state=42
        )
    }
    
    for name, model in models.items():
        # 交差検証による評価（より信頼性の高い評価）
        # cv=5: データを5分割して5回学習・評価を繰り返す
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # 学習と予測
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n{name}:")
        print(f"交差検証 R² (平均 ± 標準偏差): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"テストデータ R²: {r2_score(y_test, y_pred):.3f}")
        print(f"テストデータ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    
    # ============================================================
    # ランダムフォレストの特徴量重要度
    # ============================================================
    # どの特徴量が予測に重要かを示す
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': data.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest: Feature Importances')
    plt.gca().invert_yaxis()  # 重要度の高い順に表示
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("\n特徴量の重要度を 'feature_importances.png' として保存しました")

def main():
    """
    メイン実行関数
    各種回帰手法を順番に実行し、それぞれの特性を示す
    """
    print("=== scikit-learn 回帰分析サンプル ===\n")
    
    # データの生成
    X_linear, y_linear, X_nonlinear, y_nonlinear = generate_synthetic_data()
    
    # 1. 線形回帰の実演（線形データに対して）
    demonstrate_linear_regression(X_linear, y_linear)
    
    # 2. 正則化手法の比較（高次元データに対して）
    demonstrate_regularization()
    
    # 3. 多項式回帰（非線形データに対して）
    demonstrate_polynomial_regression(X_nonlinear, y_nonlinear)
    
    # 4. アンサンブル回帰（複雑なデータに対して）
    demonstrate_ensemble_regression()
    
    # まとめ
    print("\n=== まとめ ===")
    print("1. 線形回帰: シンプルで解釈しやすいが、非線形パターンを捉えられない")
    print("2. 正則化: 過学習を防ぎ、多重共線性に対処できる")
    print("   - Ridge: すべての特徴量を使いつつ係数を小さく")
    print("   - Lasso: 不要な特徴量を自動的に除外")
    print("3. 多項式回帰: 非線形パターンを捉えられるが、次数が高いと過学習しやすい")
    print("4. ランダムフォレスト: 非線形パターンを捉え、特徴量の重要度も分かる")

# ============================================================
# メイン実行部
# ============================================================
if __name__ == "__main__":
    main()