"""
Titanic Survival Prediction - Enhanced Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

os.makedirs('../plots', exist_ok=True)

# -------------------------------------------------------------------
# 1. LOAD & INITIAL EXPLORATION
# -------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv('../data/train.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0], "\n")

print(f"Survival rate: {df['Survived'].mean():.3f}")

# -------------------------------------------------------------------
# 2. ADVANCED FEATURE ENGINEERING
# -------------------------------------------------------------------
print("\nEngineering features...")
df_fe = df.copy()

# Title extraction
df_fe['Title'] = df_fe['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
    'Capt': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare', 'Countess': 'Rare',
    'Don': 'Rare', 'Jonkheer': 'Rare'
}
df_fe['Title'] = df_fe['Title'].map(title_mapping).fillna('Rare')
title_dummies = pd.get_dummies(df_fe['Title'], prefix='Title')

# Cabin features
df_fe['CabinLetter'] = df_fe['Cabin'].dropna().str[0]
df_fe['CabinLetter'] = df_fe['CabinLetter'].fillna('U')
has_cabin = df_fe['Cabin'].notna().astype(int)

# Family features
df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)

# Group ticket features
ticket_group = df_fe.groupby('Ticket')['PassengerId'].transform('count')
df_fe['TicketGroupSize'] = ticket_group
df_fe['TicketGroupShared'] = (ticket_group > 1).astype(int)

# Name length (some signal in name detail)
df_fe['NameLength'] = df_fe['Name'].str.len()

# Age groups (from median when missing, refined after imputation)
df_fe['AgeGroup'] = pd.cut(df_fe['Age'], bins=[0, 5, 12, 18, 30, 50, 80], labels=['Child','Teen','Young','Adult','Middle','Senior'])

# -------------------------------------------------------------------
# 3. SOPHISTICATED IMPUTATION
# -------------------------------------------------------------------
print("Imputing missing values...")

# Embarked: fill with mode
df_fe['Embarked'] = df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0])

# Age: KNN imputation on selected features
age_features = pd.DataFrame({
    'Age': df_fe['Age'],
    'Pclass': df_fe['Pclass'],
    'SibSp': df_fe['SibSp'],
    'Parch': df_fe['Parch'],
    'Fare': df_fe['Fare'],
    'Title_Mr': title_dummies.get('Title_Mr', 0),
    'Title_Mrs': title_dummies.get('Title_Mrs', 0),
    'Title_Miss': title_dummies.get('Title_Miss', 0),
    'Title_Master': title_dummies.get('Title_Master', 0),
})
age_imputer = KNNImputer(n_neighbors=5)
age_imputed = age_imputer.fit_transform(age_features)
df_fe['Age'] = age_imputed[:, 0]

# Reassign AgeGroup after imputation
df_fe['AgeGroup'] = pd.cut(df_fe['Age'], bins=[0, 5, 12, 18, 30, 50, 80], labels=['Child','Teen','Young','Adult','Middle','Senior'])

# Fare: fill missing with median by Pclass
df_fe['Fare'] = df_fe.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

# -------------------------------------------------------------------
# 4. ENCODING & FINAL FEATURE SET
# -------------------------------------------------------------------
print("Encoding features...")

df_model = df_fe.copy()
df_model['Sex'] = df_model['Sex'].map({'male': 0, 'female': 1})
df_model['Embarked'] = df_model['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# AgeGroup dummies
age_dummies = pd.get_dummies(df_model['AgeGroup'], prefix='AgeGroup')

# Select final features
features = pd.concat([
    df_model[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
              'FamilySize', 'IsAlone', 'TicketGroupSize', 'TicketGroupShared',
              'NameLength', 'Embarked', 'CabinLetter', 'AgeGroup']].drop('AgeGroup', axis=1),
    title_dummies,
    age_dummies,
    pd.DataFrame({'HasCabin': has_cabin}, index=df_model.index),
], axis=1)

# Encode CabinLetter
features['CabinLetter'] = LabelEncoder().fit_transform(features['CabinLetter'])

X = features.copy()
y = df_fe['Survived'].values

print(f"Feature shape: {X.shape}")

# -------------------------------------------------------------------
# 5. TRAIN/VALIDATION SPLIT & SCALING
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 6. MODEL DEFINITIONS
# -------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
}

# -------------------------------------------------------------------
# 7. CROSS-VALIDATION
# -------------------------------------------------------------------
print("\nCross-validating models (Stratified 5-Fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_results.append({'Model': name, 'Mean CV Acc': scores.mean(), 'Std CV Acc': scores.std()})
    print(f"  {name:22s}  CV Acc: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_df = pd.DataFrame(cv_results).sort_values('Mean CV Acc', ascending=False)
print(f"\nBest CV model: {cv_df.iloc[0]['Model']} ({cv_df.iloc[0]['Mean CV Acc']:.4f})")

# -------------------------------------------------------------------
# 8. HYPERPARAMETER TUNING (top 2 models)
# -------------------------------------------------------------------
print("\nTuning hyperparameters...")

best_models = {}

# Tune Random Forest
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_models['Random Forest'] = rf_grid.best_estimator_
print(f"  RF: {rf_grid.best_params_} -> CV Acc: {rf_grid.best_score_:.4f}")

# Tune XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.05, 0.1],
}
xgb_grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=42),
                        xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)
best_models['XGBoost'] = xgb_grid.best_estimator_
print(f"  XGB: {xgb_grid.best_params_} -> CV Acc: {xgb_grid.best_score_:.4f}")

# -------------------------------------------------------------------
# 9. ENSEMBLE (Stacking + Voting)
# -------------------------------------------------------------------
print("\nBuilding ensembles...")

# Voting classifier (top individual models)
voting = VotingClassifier(
    estimators=[
        ('rf', best_models['Random Forest']),
        ('xgb', best_models['XGBoost']),
        ('lr', LogisticRegression(max_iter=2000, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
    ],
    voting='soft'
)
voting.fit(X_train_scaled, y_train)
best_models['Voting Ensemble'] = voting

# Stacking classifier
stack = StackingClassifier(
    estimators=[
        ('rf', best_models['Random Forest']),
        ('xgb', best_models['XGBoost']),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5
)
stack.fit(X_train_scaled, y_train)
best_models['Stacking Ensemble'] = stack

# -------------------------------------------------------------------
# 10. FINAL EVALUATION
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

results = []
plt.figure(figsize=(10, 8))

for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    results.append({'Model': name, 'Test Acc': acc, 'ROC-AUC': auc_score})

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')

    print(f"\n{'─' * 40}")
    print(f"{name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc_score:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:")
    print(f"{classification_report(y_test, y_pred)}")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Titanic Survival Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../plots/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

results_df = pd.DataFrame(results).sort_values('Test Acc', ascending=False)
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(results_df.to_string(index=False))

# -------------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# -------------------------------------------------------------------
print(f"\n\nTop features (from best model):")
best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]

if hasattr(best_model, 'estimators_') and isinstance(best_model, VotingClassifier):
    # For voting ensemble, use XGBoost's importance
    imp_model = best_model.named_estimators_['xgb']
elif hasattr(best_model, 'estimators_') and isinstance(best_model, StackingClassifier):
    imp_model = best_model.named_estimators_['xgb']
else:
    imp_model = best_model

if hasattr(imp_model, 'feature_importances_'):
    imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': imp_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(imp.head(15).to_string(index=False))

# -------------------------------------------------------------------
# 12. LEARNING CURVE (best model)
# -------------------------------------------------------------------
print("\nGenerating learning curve...")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_scaled, y_train,
    cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', label='Training Acc')
plt.plot(train_sizes, test_mean, 'o-', label='Validation Acc')
plt.fill_between(train_sizes,
                 train_mean - np.std(train_scores, axis=1),
                 train_mean + np.std(train_scores, axis=1), alpha=0.2)
plt.fill_between(train_sizes,
                 test_mean - np.std(test_scores, axis=1),
                 test_mean + np.std(test_scores, axis=1), alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve - {best_model_name}')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('../plots/learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ All plots saved to 'plots/' folder.")
print(f"✅ Best model: {best_model_name} with {results_df.iloc[0]['Test Acc']:.4f} test accuracy")
