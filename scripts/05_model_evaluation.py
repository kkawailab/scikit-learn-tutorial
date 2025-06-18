#!/usr/bin/env python3
"""
モデル評価と改善のサンプルスクリプト
交差検証、ハイパーパラメータチューニング、特徴量選択を含む
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

def demonstrate_cross_validation():
    """交差検証の実演"""
    print("=== 交差検証の実演 ===\n")
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # モデルの定義
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 通常の train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf.fit(X_train, y_train)
    simple_score = rf.score(X_test, y_test)
    
    print(f"単純な train/test split での精度: {simple_score:.3f}")
    
    # 様々な交差検証手法
    cv_methods = {
        '5-Fold CV': 5,
        '10-Fold CV': 10,
        'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    }
    
    results = []
    for name, cv in cv_methods.items():
        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        results.append({
            'Method': name,
            'Mean Score': scores.mean(),
            'Std Dev': scores.std(),
            'Min Score': scores.min(),
            'Max Score': scores.max()
        })
        print(f"\n{name}:")
        print(f"  平均スコア: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  各分割のスコア: {scores}")
    
    # 結果の可視化
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    plt.bar(x, results_df['Mean Score'], yerr=results_df['Std Dev'], 
            capsize=10, alpha=0.7)
    plt.xticks(x, results_df['Method'])
    plt.ylabel('Accuracy Score')
    plt.title('Cross-Validation Methods Comparison')
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(results_df['Mean Score'], results_df['Std Dev'])):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png')
    print("\n交差検証の比較を 'cross_validation_comparison.png' として保存しました")

def demonstrate_grid_search():
    """グリッドサーチによるハイパーパラメータチューニング"""
    print("\n=== グリッドサーチ ===\n")
    
    # データの準備
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # パイプラインの構築
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])
    
    # パラメータグリッド
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1],
        'svm__kernel': ['rbf', 'linear']
    }
    
    # グリッドサーチ
    print("グリッドサーチ実行中...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n最適なパラメータ: {grid_search.best_params_}")
    print(f"最適なクロスバリデーションスコア: {grid_search.best_score_:.3f}")
    
    # テストセットでの評価
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"テストセットでの精度: {test_score:.3f}")
    
    # 結果の可視化（RBFカーネルの場合）
    results_df = pd.DataFrame(grid_search.cv_results_)
    rbf_results = results_df[results_df['param_svm__kernel'] == 'rbf']
    
    # ヒートマップの作成
    pivot_table = rbf_results.pivot_table(
        values='mean_test_score',
        index='param_svm__gamma',
        columns='param_svm__C'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Grid Search Results: SVM (RBF Kernel)')
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.tight_layout()
    plt.savefig('grid_search_heatmap.png')
    print("\nグリッドサーチ結果を 'grid_search_heatmap.png' として保存しました")

def demonstrate_randomized_search():
    """ランダムサーチによるハイパーパラメータチューニング"""
    print("\n=== ランダムサーチ ===\n")
    
    from scipy.stats import uniform, randint
    
    # データの準備（前回と同じ）
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ランダムフォレストのパラメータ分布
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.3, 0.7)  # 30%から100%の間
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # ランダムサーチ
    print("ランダムサーチ実行中...")
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=5, scoring='accuracy',
        n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"\n最適なパラメータ: {random_search.best_params_}")
    print(f"最適なクロスバリデーションスコア: {random_search.best_score_:.3f}")
    print(f"テストセットでの精度: {random_search.score(X_test, y_test):.3f}")
    
    # パラメータの重要性を可視化
    results = pd.DataFrame(random_search.cv_results_)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    params_to_plot = ['n_estimators', 'max_depth', 'min_samples_split', 
                     'min_samples_leaf', 'max_features']
    
    for idx, param in enumerate(params_to_plot):
        param_col = f'param_{param}'
        axes[idx].scatter(results[param_col], results['mean_test_score'], alpha=0.5)
        axes[idx].set_xlabel(param)
        axes[idx].set_ylabel('Mean CV Score')
        axes[idx].set_title(f'Score vs {param}')
        axes[idx].grid(True, alpha=0.3)
    
    # 最後のサブプロットは空白
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('random_search_results.png')
    print("\nランダムサーチ結果を 'random_search_results.png' として保存しました")

def demonstrate_learning_curves():
    """学習曲線の可視化"""
    print("\n=== 学習曲線 ===\n")
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # 複数のモデルで比較
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', gamma='scale', random_state=42)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (name, model) in enumerate(models.items()):
        # 学習曲線の計算
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # 平均と標準偏差
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # プロット
        ax = axes[idx]
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f'Learning Curve: {name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 1.05)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("学習曲線を 'learning_curves.png' として保存しました")

def demonstrate_validation_curves():
    """検証曲線の可視化"""
    print("\n=== 検証曲線 ===\n")
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # Random Forestの木の深さの影響を調査
    param_range = np.arange(1, 21)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy'
    )
    
    # 平均と標準偏差
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation score')
    
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='red')
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Score')
    plt.title('Validation Curve: Random Forest Max Depth')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.85, 1.02)
    
    # 最適な深さを表示
    best_depth = param_range[val_mean.argmax()]
    plt.axvline(x=best_depth, color='green', linestyle='--', 
                label=f'Best depth: {best_depth}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('validation_curve.png')
    print("検証曲線を 'validation_curve.png' として保存しました")

def demonstrate_feature_selection():
    """特徴量選択の実演"""
    print("\n=== 特徴量選択 ===\n")
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=10,
                              n_redundant=10, n_repeated=10, n_clusters_per_class=1,
                              random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 元のモデルの性能
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    baseline_score = rf.score(X_test, y_test)
    print(f"全特徴量を使用した精度: {baseline_score:.3f}")
    
    # 1. SelectKBest
    k_features = [5, 10, 20, 30, 40, 50]
    selectk_scores = []
    
    for k in k_features:
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_train_selected, y_train)
        score = rf_temp.score(X_test_selected, y_test)
        selectk_scores.append(score)
    
    # 2. Recursive Feature Elimination (RFE)
    rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
              n_features_to_select=10)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rfe.fit(X_train_rfe, y_train)
    rfe_score = rf_rfe.score(X_test_rfe, y_test)
    
    print(f"\nRFE (10特徴量) の精度: {rfe_score:.3f}")
    
    # 結果の可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # SelectKBestの結果
    axes[0].plot(k_features, selectk_scores, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=baseline_score, color='red', linestyle='--', 
                    label=f'Baseline ({baseline_score:.3f})')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].set_title('SelectKBest: Score vs Number of Features')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 特徴量の重要度
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    
    axes[1].bar(range(20), importances[indices])
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Importance')
    axes[1].set_title('Top 20 Feature Importances')
    axes[1].set_xticks(range(20))
    axes[1].set_xticklabels(indices)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_selection.png')
    print("\n特徴量選択の結果を 'feature_selection.png' として保存しました")

def demonstrate_roc_pr_curves():
    """ROC曲線とPR曲線の描画"""
    print("\n=== ROC曲線とPR曲線 ===\n")
    
    # データの準備
    X, y = make_classification(n_samples=1000, n_classes=2, n_features=20,
                              n_informative=15, n_redundant=5, 
                              weights=[0.7, 0.3], random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    
    # 複数のモデルで比較
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for name, model in models.items():
        # モデルの学習
        model.fit(X_train, y_train)
        
        # 予測確率
        y_score = model.predict_proba(X_test)[:, 1]
        
        # ROC曲線
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, linewidth=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        # PR曲線
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        
        axes[1].plot(recall, precision, linewidth=2, label=name)
    
    # ROC曲線の設定
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # PR曲線の設定
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png')
    print("ROC曲線とPR曲線を 'roc_pr_curves.png' として保存しました")

def main():
    print("=== scikit-learn モデル評価と改善サンプル ===\n")
    
    # 1. 交差検証
    demonstrate_cross_validation()
    
    # 2. グリッドサーチ
    demonstrate_grid_search()
    
    # 3. ランダムサーチ
    demonstrate_randomized_search()
    
    # 4. 学習曲線
    demonstrate_learning_curves()
    
    # 5. 検証曲線
    demonstrate_validation_curves()
    
    # 6. 特徴量選択
    demonstrate_feature_selection()
    
    # 7. ROC曲線とPR曲線
    demonstrate_roc_pr_curves()
    
    print("\n=== まとめ ===")
    print("1. 交差検証: より信頼性の高いモデル評価")
    print("2. ハイパーパラメータチューニング: 最適なモデル設定の探索")
    print("3. 学習曲線: 過学習/未学習の診断")
    print("4. 特徴量選択: モデルの簡素化と性能向上")
    print("5. 評価指標: タスクに応じた適切な指標の選択")

if __name__ == "__main__":
    main()