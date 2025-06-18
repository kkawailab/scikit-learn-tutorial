# 第3章：教師あり学習 - 分類

## 3.1 分類問題の基礎

分類問題は、入力データを事前に定義されたカテゴリ（クラス）のいずれかに割り当てる機械学習タスクです。

### 分類問題の種類

- **二値分類**: 2つのクラスから選択（例：スパムメール判定）
- **多クラス分類**: 3つ以上のクラスから選択（例：手書き数字認識）
- **多ラベル分類**: 複数のラベルを同時に割り当て（例：画像のタグ付け）

## 3.2 ロジスティック回帰

### サンプルコード1：二値分類の基本

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# データセットの生成
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ロジスティック回帰モデルの学習
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 予測
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

# 精度の評価
accuracy = accuracy_score(y_test, y_pred)
print(f"精度: {accuracy:.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred))

# 決定境界の可視化
def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

plot_decision_boundary(X_test, y_test, log_reg, 'Logistic Regression Decision Boundary')
```

### サンプルコード2：多クラス分類

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Irisデータセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 多クラスロジスティック回帰
log_reg_multi = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
log_reg_multi.fit(X_train, y_train)

# 予測と評価
y_pred = log_reg_multi.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"多クラス分類の精度: {accuracy:.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 混同行列
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Multiclass Classification')
plt.show()

# 各クラスの予測確率
sample_idx = 0
sample_proba = log_reg_multi.predict_proba(X_test[sample_idx:sample_idx+1])
print(f"\nサンプル{sample_idx}の予測確率:")
for i, class_name in enumerate(iris.target_names):
    print(f"{class_name}: {sample_proba[0][i]:.3f}")
```

## 3.3 決定木とランダムフォレスト

### サンプルコード3：決定木の可視化と解釈

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# 決定木モデルの学習
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 決定木の可視化
plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importances - Decision Tree')
plt.show()

print("特徴量の重要度:")
print(feature_importance)
```

### サンプルコード4：ランダムフォレストの最適化

```python
from sklearn.model_selection import GridSearchCV, validation_curve

# ランダムフォレストの基本モデル
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print(f"ランダムフォレスト（デフォルト）の精度: {rf.score(X_test, y_test):.3f}")

# ハイパーパラメータの最適化
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\n最適なパラメータ: {grid_search.best_params_}")
print(f"最適なスコア: {grid_search.best_score_:.3f}")
print(f"テストデータでの精度: {grid_search.score(X_test, y_test):.3f}")

# 木の数と性能の関係
n_estimators_range = [10, 20, 50, 100, 200, 300, 500]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=5,
    scoring='accuracy'
)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(n_estimators_range, val_scores.mean(axis=1), 'o-', label='Validation score')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 3.4 サポートベクターマシン（SVM）

### サンプルコード5：SVMの実装とカーネルトリック

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons

# 非線形に分離可能なデータセットの生成
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Moonsデータセット
X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

# Circlesデータセット
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

datasets = [
    (X_moons, y_moons, "Moons Dataset"),
    (X_circles, y_circles, "Circles Dataset")
]

kernels = ['linear', 'rbf', 'poly']

for i, (X, y, title) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for j, kernel in enumerate(kernels):
        ax = axes[i, j]
        
        # SVMモデルの学習
        svm = SVC(kernel=kernel, gamma='auto', C=1.0)
        svm.fit(X_train, y_train)
        
        # 決定境界の描画
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
        ax.set_title(f'{title}\nKernel: {kernel}\nAccuracy: {svm.score(X_test, y_test):.3f}')

plt.tight_layout()
plt.show()
```

### サンプルコード6：SVMのハイパーパラメータチューニング

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

# パラメータの分布を定義
param_distributions = {
    'C': loguniform(0.001, 100),
    'gamma': loguniform(0.001, 10),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# ランダムサーチ
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Irisデータセットで実験
random_search.fit(X_train, y_train)

print(f"最適なパラメータ: {random_search.best_params_}")
print(f"最適なスコア: {random_search.best_score_:.3f}")

# 最適なモデルでの予測
best_svm = random_search.best_estimator_
y_pred = best_svm.predict(X_test)

print(f"\nテストデータでの精度: {accuracy_score(y_test, y_pred):.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# パラメータの重要性を可視化
results = pd.DataFrame(random_search.cv_results_)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# C値と性能の関係
ax = axes[0]
for kernel in param_distributions['kernel']:
    kernel_results = results[results['param_kernel'] == kernel]
    ax.scatter(kernel_results['param_C'], 
               kernel_results['mean_test_score'],
               label=kernel, alpha=0.6)
ax.set_xscale('log')
ax.set_xlabel('C')
ax.set_ylabel('Mean CV Score')
ax.set_title('Performance vs C parameter')
ax.legend()

# gamma値と性能の関係
ax = axes[1]
rbf_results = results[results['param_kernel'] == 'rbf']
scatter = ax.scatter(rbf_results['param_C'], 
                    rbf_results['param_gamma'],
                    c=rbf_results['mean_test_score'],
                    cmap='viridis')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('C')
ax.set_ylabel('Gamma')
ax.set_title('RBF Kernel: C vs Gamma')
plt.colorbar(scatter, ax=ax, label='Mean CV Score')

plt.tight_layout()
plt.show()
```

## 3.5 その他の分類アルゴリズム

### サンプルコード7：アンサンブル学習

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# 複数の基本分類器を定義
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(probability=True, random_state=42)
clf4 = GaussianNB()
clf5 = KNeighborsClassifier(n_neighbors=5)

# 投票分類器
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3),
        ('nb', clf4),
        ('knn', clf5)
    ],
    voting='soft'  # 確率に基づく投票
)

# 各モデルの性能比較
classifiers = {
    'Logistic Regression': clf1,
    'Random Forest': clf2,
    'SVM': clf3,
    'Naive Bayes': clf4,
    'KNN': clf5,
    'Voting Classifier': voting_clf,
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    results.append({
        'Classifier': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Overfit': train_score - test_score
    })

results_df = pd.DataFrame(results).sort_values('Test Score', ascending=False)
print(results_df)

# 結果の可視化
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['Train Score'], width, label='Train Score')
bars2 = ax.bar(x + width/2, results_df['Test Score'], width, label='Test Score')

ax.set_xlabel('Classifier')
ax.set_ylabel('Score')
ax.set_title('Classifier Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### サンプルコード8：確率的予測とキャリブレーション

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# 確率予測が必要なデータセットの準備
X_binary, y_binary = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# 様々な分類器の確率予測
classifiers_prob = [
    ('Logistic Regression', LogisticRegression()),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('SVM', SVC(probability=True)),
    ('Naive Bayes', GaussianNB())
]

plt.figure(figsize=(15, 10))

for i, (name, clf) in enumerate(classifiers_prob):
    # モデルの学習
    clf.fit(X_train_b, y_train_b)
    
    # キャリブレーション前の予測確率
    prob_pos = clf.predict_proba(X_test_b)[:, 1]
    
    # キャリブレーション
    calibrated_clf = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
    calibrated_clf.fit(X_train_b, y_train_b)
    prob_pos_calibrated = calibrated_clf.predict_proba(X_test_b)[:, 1]
    
    # キャリブレーション曲線
    plt.subplot(2, 2, i+1)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_b, prob_pos, n_bins=10
    )
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
        y_test_b, prob_pos_calibrated, n_bins=10
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, 
             's-', label=f'{name} (uncalibrated)')
    plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, 
             's-', label=f'{name} (calibrated)')
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration plot - {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 練習問題

### 問題1：不均衡データの処理
1. 不均衡なデータセット（クラス比が9:1）を生成
2. 通常の分類器で学習し、問題点を確認
3. SMOTE、クラス重み調整、アンダーサンプリングを実装
4. 各手法の性能を比較（Precision、Recall、F1-score）

### 問題2：多ラベル分類
1. 多ラベル分類のデータセットを作成
2. Binary Relevance、Classifier Chainsを実装
3. 各ラベルの予測精度を評価

### 問題3：実践的な分類問題
1. 顧客の離脱予測データセットを作成（特徴量10個以上）
2. 前処理パイプラインを構築
3. 複数のアルゴリズムを比較
4. 最適なモデルを選択し、ビジネス的な解釈を提供

## 解答

### 解答1：不均衡データの処理

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# 不均衡データセットの生成
X_imb, y_imb = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0.02,
    random_state=42
)

print(f"クラス分布: {Counter(y_imb)}")

# データの分割
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

# 1. 通常の分類器
rf_normal = RandomForestClassifier(random_state=42)
rf_normal.fit(X_train_imb, y_train_imb)
y_pred_normal = rf_normal.predict(X_test_imb)

print("\n通常の分類器:")
print(classification_report(y_test_imb, y_pred_normal))

# 2. クラス重み調整
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train_imb, y_train_imb)
y_pred_weighted = rf_weighted.predict(X_test_imb)

print("\nクラス重み調整:")
print(classification_report(y_test_imb, y_pred_weighted))

# 3. SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imb, y_train_imb)
print(f"\nSMOTE後のクラス分布: {Counter(y_train_smote)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test_imb)

print("\nSMOTE:")
print(classification_report(y_test_imb, y_pred_smote))

# 4. アンダーサンプリング
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train_imb, y_train_imb)
print(f"\nアンダーサンプリング後のクラス分布: {Counter(y_train_under)}")

rf_under = RandomForestClassifier(random_state=42)
rf_under.fit(X_train_under, y_train_under)
y_pred_under = rf_under.predict(X_test_imb)

print("\nアンダーサンプリング:")
print(classification_report(y_test_imb, y_pred_under))

# ROC曲線の比較
plt.figure(figsize=(10, 8))

methods = {
    'Normal': (rf_normal, y_pred_normal),
    'Class Weight': (rf_weighted, y_pred_weighted),
    'SMOTE': (rf_smote, y_pred_smote),
    'Undersampling': (rf_under, y_pred_under)
}

for name, (model, y_pred) in methods.items():
    y_proba = model.predict_proba(X_test_imb)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_imb, y_proba)
    auc = roc_auc_score(y_test_imb, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Imbalanced Data Handling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 解答2：多ラベル分類

```python
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import hamming_loss, jaccard_score

# 多ラベルデータセットの生成
X_multi, y_multi = make_multilabel_classification(
    n_samples=1000,
    n_features=20,
    n_classes=5,
    n_labels=2,
    random_state=42
)

print(f"データ形状: X={X_multi.shape}, y={y_multi.shape}")
print(f"ラベル数の分布:")
print(pd.Series(y_multi.sum(axis=1)).value_counts().sort_index())

# データの分割
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# 1. Binary Relevance
br_classifier = MultiOutputClassifier(RandomForestClassifier(random_state=42))
br_classifier.fit(X_train_m, y_train_m)
y_pred_br = br_classifier.predict(X_test_m)

# 2. Classifier Chain
cc_classifier = ClassifierChain(RandomForestClassifier(random_state=42))
cc_classifier.fit(X_train_m, y_train_m)
y_pred_cc = cc_classifier.predict(X_test_m)

# 評価
print("\n評価結果:")
print(f"Binary Relevance - Hamming Loss: {hamming_loss(y_test_m, y_pred_br):.3f}")
print(f"Classifier Chain - Hamming Loss: {hamming_loss(y_test_m, y_pred_cc):.3f}")

# 各ラベルごとの精度
from sklearn.metrics import precision_recall_fscore_support

print("\n各ラベルの性能 (Binary Relevance):")
precision, recall, f1, _ = precision_recall_fscore_support(y_test_m, y_pred_br, average=None)
for i in range(y_multi.shape[1]):
    print(f"Label {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

print("\n各ラベルの性能 (Classifier Chain):")
precision, recall, f1, _ = precision_recall_fscore_support(y_test_m, y_pred_cc, average=None)
for i in range(y_multi.shape[1]):
    print(f"Label {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

# ラベルの相関を可視化
label_corr = np.corrcoef(y_train_m.T)
plt.figure(figsize=(8, 6))
sns.heatmap(label_corr, annot=True, cmap='coolwarm', center=0,
            xticklabels=[f'Label {i}' for i in range(5)],
            yticklabels=[f'Label {i}' for i in range(5)])
plt.title('Label Correlation Matrix')
plt.show()
```

### 解答3：実践的な分類問題

```python
# 顧客離脱予測データセットの作成
np.random.seed(42)
n_customers = 2000

# 特徴量の生成
customer_data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n_customers),  # 契約期間（月）
    'monthly_charges': np.random.uniform(20, 120, n_customers),  # 月額料金
    'total_charges': np.random.uniform(100, 8000, n_customers),  # 総支払額
    'num_services': np.random.randint(1, 8, n_customers),  # 利用サービス数
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
    'online_security': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
    'device_protection': np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65]),
    'streaming_tv': np.random.choice(['Yes', 'No'], n_customers, p=[0.45, 0.55]),
    'streaming_movies': np.random.choice(['Yes', 'No'], n_customers, p=[0.45, 0.55])
})

# 離脱フラグの生成（ビジネスロジックに基づく）
churn_probability = 0.2
churn_factors = (
    (customer_data['contract_type'] == 'Month-to-month') * 0.3 +
    (customer_data['tenure'] < 12) * 0.2 +
    (customer_data['monthly_charges'] > 80) * 0.1 +
    (customer_data['tech_support'] == 'No') * 0.1 +
    (customer_data['payment_method'] == 'Mailed check') * 0.1
)
churn_probability_adjusted = churn_probability + churn_factors
customer_data['churn'] = (np.random.rand(n_customers) < churn_probability_adjusted).astype(int)

print("データセットの概要:")
print(customer_data.info())
print(f"\n離脱率: {customer_data['churn'].mean():.2%}")

# 前処理パイプラインの構築
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 特徴量の分離
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'num_services']
categorical_features = ['contract_type', 'payment_method', 'paperless_billing', 
                       'tech_support', 'online_security', 'device_protection',
                       'streaming_tv', 'streaming_movies']

# 前処理パイプライン
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# データの準備
X = customer_data.drop('churn', axis=1)
y = customer_data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 複数のモデルを比較
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

results = []

for name, model in models.items():
    # パイプラインの作成
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 学習と予測
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 評価
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1-Score': report['1']['f1-score']
    })

results_df = pd.DataFrame(results).round(3)
print("\nモデル比較:")
print(results_df)

# 最適モデルの特徴量重要度
best_model_name = results_df.loc[results_df['AUC'].idxmax(), 'Model']
print(f"\n最適モデル: {best_model_name}")

# Random Forestの特徴量重要度を可視化
if best_model_name == 'Random Forest':
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    
    # 特徴量名の取得
    feature_names = (numeric_features + 
                    list(rf_pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features)))
    
    # 重要度の取得
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    
    # 可視化
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances - Customer Churn Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ビジネス的な解釈
print("\nビジネス的な洞察:")
print("1. 契約タイプ（月次契約）が離脱の最も重要な要因")
print("2. 契約期間が短い顧客ほど離脱リスクが高い")
print("3. テクニカルサポートの有無が顧客維持に影響")
print("4. 高額プランの顧客は離脱しやすい傾向")
print("\n推奨施策:")
print("- 月次契約顧客への長期契約への移行促進")
print("- 新規顧客への特別なオンボーディングプログラム")
print("- テクニカルサポートの利用促進")
```

## まとめ

この章では、教師あり学習の分類問題について学習しました：

- ロジスティック回帰による線形分類
- 決定木とランダムフォレストによる非線形分類
- SVMとカーネルトリック
- アンサンブル学習による性能向上
- 不均衡データの処理手法
- 多ラベル分類
- 実践的な分類問題の解決アプローチ

次章では、教師あり学習の回帰問題について詳しく学習します。

[目次に戻る](README.md)