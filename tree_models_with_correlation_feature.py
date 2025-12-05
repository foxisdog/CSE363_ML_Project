import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 데이터 불러오기 및 상관계수 계산
# ============================================================================
print("=" * 80)
print("Step 1: Loading data and calculating correlations")
print("=" * 80)

df = pd.read_csv('DED_Dataset_preprocessed.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 타겟과 특성 분리
X = df.drop('Dry Eye Disease', axis=1)
y = df['Dry Eye Disease']

# 각 특성과 타겟 간의 상관계수 계산
correlations = X.corrwith(y)
print("\n각 특성과 Dry Eye Disease 간의 상관계수:")
print(correlations.sort_values(ascending=False))

# ============================================================================
# 2. 상관계수 가중 특성 생성
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Creating weighted correlation feature")
print("=" * 80)

# 새로운 특성: correlation * value의 선형결합
# 각 데이터 포인트에 대해 (상관계수 * 특성값)의 합을 계산
weighted_feature = np.zeros(len(X))
for col in X.columns:
    weighted_feature += correlations[col] * X[col].values

# 새로운 특성을 데이터프레임에 추가
X_with_corr = X.copy()
X_with_corr['Correlation_Weighted_Feature'] = weighted_feature

print(f"Original features: {X.shape[1]}")
print(f"Features with correlation-weighted feature: {X_with_corr.shape[1]}")
print(f"New feature statistics:")
print(f"  Mean: {weighted_feature.mean():.4f}")
print(f"  Std: {weighted_feature.std():.4f}")
print(f"  Min: {weighted_feature.min():.4f}")
print(f"  Max: {weighted_feature.max():.4f}")

# ============================================================================
# 3. 9개의 트리 기반 모델 정의
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Defining 9 tree-based models")
print("=" * 80)

models = {
    '1. Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),
    '2. Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    '3. Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    ),
    '4. AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        random_state=42,
        algorithm='SAMME'
    ),
    '5. Extra Trees': ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    '6. XGBoost': XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ),
    '7. LightGBM': LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    '8. CatBoost': CatBoostClassifier(
        iterations=100,
        random_state=42,
        verbose=0
    ),
    '9. Histogram-based Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42
    )
}

print(f"Total models: {len(models)}")
for model_name in models.keys():
    print(f"  - {model_name}")

# ============================================================================
# 4. 모델 평가 (원본 특성 vs 상관계수 특성 추가)
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Evaluating models with cross-validation")
print("=" * 80)

results = []

scoring = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']

for model_name, model in models.items():
    print(f"\n{model_name}...")

    # 원본 특성으로 평가
    print("  - Evaluating with original features...")
    cv_results_original = cross_validate(
        model, X, y,
        cv=5,
        scoring=scoring,
        n_jobs=-1
    )

    # 상관계수 특성 추가 후 평가
    print("  - Evaluating with correlation-weighted feature...")
    cv_results_corr = cross_validate(
        model, X_with_corr, y,
        cv=5,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        'Model': model_name,
        'Original_Accuracy': cv_results_original['test_accuracy'].mean(),
        'Original_Accuracy_Std': cv_results_original['test_accuracy'].std(),
        'Original_Precision': cv_results_original['test_precision'].mean(),
        'Original_Recall': cv_results_original['test_recall'].mean(),
        'Original_ROC_AUC': cv_results_original['test_roc_auc'].mean(),
        'Original_F1': cv_results_original['test_f1'].mean(),
        'CorrFeature_Accuracy': cv_results_corr['test_accuracy'].mean(),
        'CorrFeature_Accuracy_Std': cv_results_corr['test_accuracy'].std(),
        'CorrFeature_Precision': cv_results_corr['test_precision'].mean(),
        'CorrFeature_Recall': cv_results_corr['test_recall'].mean(),
        'CorrFeature_ROC_AUC': cv_results_corr['test_roc_auc'].mean(),
        'CorrFeature_F1': cv_results_corr['test_f1'].mean(),
        'Accuracy_Improvement': cv_results_corr['test_accuracy'].mean() - cv_results_original['test_accuracy'].mean()
    })

# ============================================================================
# 5. 결과 정리 및 출력
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Results Summary")
print("=" * 80)

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("DETAILED RESULTS TABLE")
print("=" * 80)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

# 원본 특성 성능
print("\n[Original Features Performance]")
print(results_df[['Model', 'Original_Accuracy', 'Original_Precision', 'Original_Recall', 'Original_ROC_AUC', 'Original_F1']].to_string(index=False))

# 상관계수 특성 추가 후 성능
print("\n[With Correlation-Weighted Feature Performance]")
print(results_df[['Model', 'CorrFeature_Accuracy', 'CorrFeature_Precision', 'CorrFeature_Recall', 'CorrFeature_ROC_AUC', 'CorrFeature_F1']].to_string(index=False))

# 성능 개선 분석
print("\n[Accuracy Improvement Analysis]")
improvement_df = results_df[['Model', 'Original_Accuracy', 'CorrFeature_Accuracy', 'Accuracy_Improvement']].copy()
improvement_df = improvement_df.sort_values('Accuracy_Improvement', ascending=False)
print(improvement_df.to_string(index=False))

print("\n" + "=" * 80)
print("BEST PERFORMING MODELS")
print("=" * 80)

# 원본 특성 기준 최고 성능
best_original = results_df.loc[results_df['Original_Accuracy'].idxmax()]
print(f"\nBest with Original Features:")
print(f"  Model: {best_original['Model']}")
print(f"  Accuracy: {best_original['Original_Accuracy']:.4f} ± {best_original['Original_Accuracy_Std']:.4f}")
print(f"  Precision: {best_original['Original_Precision']:.4f}")
print(f"  Recall: {best_original['Original_Recall']:.4f}")
print(f"  ROC-AUC: {best_original['Original_ROC_AUC']:.4f}")
print(f"  F1-Score: {best_original['Original_F1']:.4f}")

# 상관계수 특성 추가 기준 최고 성능
best_corr = results_df.loc[results_df['CorrFeature_Accuracy'].idxmax()]
print(f"\nBest with Correlation-Weighted Feature:")
print(f"  Model: {best_corr['Model']}")
print(f"  Accuracy: {best_corr['CorrFeature_Accuracy']:.4f} ± {best_corr['CorrFeature_Accuracy_Std']:.4f}")
print(f"  Precision: {best_corr['CorrFeature_Precision']:.4f}")
print(f"  Recall: {best_corr['CorrFeature_Recall']:.4f}")
print(f"  ROC-AUC: {best_corr['CorrFeature_ROC_AUC']:.4f}")
print(f"  F1-Score: {best_corr['CorrFeature_F1']:.4f}")

# 가장 큰 개선을 보인 모델
best_improvement = results_df.loc[results_df['Accuracy_Improvement'].idxmax()]
print(f"\nMost Improved Model:")
print(f"  Model: {best_improvement['Model']}")
print(f"  Improvement: {best_improvement['Accuracy_Improvement']:.4f}")
print(f"  From {best_improvement['Original_Accuracy']:.4f} to {best_improvement['CorrFeature_Accuracy']:.4f}")

# 결과를 CSV로 저장
results_df.to_csv('tree_models_comparison_results.csv', index=False)
print("\n" + "=" * 80)
print("Results saved to: tree_models_comparison_results.csv")
print("=" * 80)
