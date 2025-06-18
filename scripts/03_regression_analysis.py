#!/usr/bin/env python3
"""
回帰分析の包括的なサンプルスクリプト
線形回帰、正則化、非線形回帰を含む
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data():
    """合成データの生成"""
    np.random.seed(42)
    n_samples = 200
    
    # 線形データ + ノイズ
    X_linear = np.random.rand(n_samples, 1) * 10
    y_linear = 2 * X_linear.squeeze() + 1 + np.random.randn(n_samples) * 2
    
    # 非線形データ
    X_nonlinear = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
    y_nonlinear = np.sin(X_nonlinear).squeeze() * 10 + np.random.randn(n_samples) * 2
    
    return X_linear, y_linear, X_nonlinear, y_nonlinear

def demonstrate_linear_regression(X, y):
    """基本的な線形回帰の実演"""
    print("=== 線形回帰の実演 ===\n")
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # モデルの学習
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # 予測
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    # 評価
    print(f"訓練データ R²: {r2_score(y_train, y_pred_train):.3f}")
    print(f"テストデータ R²: {r2_score(y_test, y_pred_test):.3f}")
    print(f"テストデータ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
    print(f"係数: {lr.coef_[0]:.3f}")
    print(f"切片: {lr.intercept_:.3f}")
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    # 予測値のプロット
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
    plt.plot(X_test, y_pred_test, 'r-', linewidth=2, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression: Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差プロット
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png')
    print("\n結果を 'linear_regression_results.png' として保存しました\n")

def demonstrate_regularization():
    """正則化手法の比較"""
    print("=== 正則化手法の比較 ===\n")
    
    # 多重共線性のあるデータを生成
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # いくつかの特徴量を相関させる
    for i in range(5):
        X[:, i+5] = X[:, i] + np.random.randn(n_samples) * 0.5
    
    # 真の係数（スパース）
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3, -2, 1, 0, -1]
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # データの分割と標準化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 各モデルの学習
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'ElasticNet (α=0.1)': ElasticNet(alpha=0.1)
    }
    
    results = []
    coefficients = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results.append({
            'Model': name,
            'Train R²': model.score(X_train_scaled, y_train),
            'Test R²': model.score(X_test_scaled, y_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Non-zero Coefs': np.sum(np.abs(model.coef_) > 1e-5)
        })
        
        coefficients[name] = model.coef_
    
    # 結果の表示
    results_df = pd.DataFrame(results)
    print(results_df.round(3))
    
    # 係数の可視化
    plt.figure(figsize=(12, 8))
    
    for i, (name, coef) in enumerate(coefficients.items()):
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(coef)), coef)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title(name)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png')
    print("\n正則化の比較を 'regularization_comparison.png' として保存しました\n")

def demonstrate_polynomial_regression(X, y):
    """多項式回帰の実演"""
    print("=== 多項式回帰の実演 ===\n")
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 異なる次数で比較
    degrees = [1, 3, 5, 9]
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # 多項式回帰パイプライン
        poly_model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        # 学習
        poly_model.fit(X_train, y_train)
        
        # 予測用のデータ
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
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_comparison.png')
    print("多項式回帰の比較を 'polynomial_regression_comparison.png' として保存しました\n")

def demonstrate_ensemble_regression():
    """アンサンブル回帰の実演"""
    print("=== ランダムフォレスト回帰の実演 ===\n")
    
    # ボストン住宅価格風のデータを生成
    np.random.seed(42)
    n_samples = 500
    
    # 特徴量の生成
    data = pd.DataFrame({
        'rooms': np.random.normal(6, 2, n_samples),
        'area': np.random.normal(100, 30, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'distance': np.random.exponential(3, n_samples),
        'crime_rate': np.random.exponential(2, n_samples)
    })
    
    # 価格の生成（非線形な関係）
    price = (
        50 * data['rooms'] +
        0.5 * data['area'] +
        -2 * data['age'] +
        -10 * data['distance'] +
        -5 * data['crime_rate'] +
        20 * np.sin(data['rooms']) +  # 非線形成分
        np.random.normal(0, 20, n_samples)
    )
    
    X = data.values
    y = price
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # モデルの比較
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        # 交差検証
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # 学習と予測
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"\n{name}:")
        print(f"交差検証 R² (平均 ± 標準偏差): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"テストデータ R²: {r2_score(y_test, y_pred):.3f}")
        print(f"テストデータ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    
    # ランダムフォレストの特徴量重要度
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': data.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest: Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("\n特徴量の重要度を 'feature_importances.png' として保存しました")

def main():
    print("=== scikit-learn 回帰分析サンプル ===\n")
    
    # データの生成
    X_linear, y_linear, X_nonlinear, y_nonlinear = generate_synthetic_data()
    
    # 1. 線形回帰の実演
    demonstrate_linear_regression(X_linear, y_linear)
    
    # 2. 正則化手法の比較
    demonstrate_regularization()
    
    # 3. 多項式回帰（非線形データ）
    demonstrate_polynomial_regression(X_nonlinear, y_nonlinear)
    
    # 4. アンサンブル回帰
    demonstrate_ensemble_regression()
    
    print("\n=== まとめ ===")
    print("1. 線形回帰: シンプルで解釈しやすいが、非線形パターンを捉えられない")
    print("2. 正則化: 過学習を防ぎ、多重共線性に対処できる")
    print("3. 多項式回帰: 非線形パターンを捉えられるが、次数が高いと過学習しやすい")
    print("4. ランダムフォレスト: 非線形パターンを捉え、特徴量の重要度も分かる")

if __name__ == "__main__":
    main()