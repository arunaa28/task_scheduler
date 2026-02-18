"""
ProManage ML Model Training Script
Trains a Random Forest model to predict project profitability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print(" " * 15 + "ProManage ML Model Training")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1/7] Loading Historical Data...")
print("-" * 70)

try:
    df = pd.read_csv('data/historical_projects.csv')
    print(f"✓ Successfully loaded {len(df)} historical projects")
    print(f"✓ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ ERROR: Could not find 'data/historical_projects.csv'")
    print("   Please make sure the file exists in the data folder.")
    exit(1)

print("\n📊 First 5 projects:")
print(df.head())

print("\n📈 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2/7] Preprocessing Data...")
print("-" * 70)

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print("⚠️  Missing values found:")
    print(missing[missing > 0])
else:
    print("✓ No missing values found")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['project_type', 'complexity']

print("\n🔢 Encoding categorical variables:")
for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  ✓ {col:20s}: {len(le.classes_):2d} unique values")
    print(f"    Categories: {', '.join(le.classes_[:5])}{'...' if len(le.classes_) > 5 else ''}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 3/7] Engineering Features...")
print("-" * 70)

# Calculate revenue efficiency (actual vs estimated)
df['revenue_efficiency'] = (df['estimated_revenue'] * 0.97)  # Assume 97% efficiency on average
df['revenue_per_person'] = df['estimated_revenue'] / df['team_size']
df['deadline_revenue_ratio'] = df['estimated_revenue'] / (df['deadline_days'] * 10000)

print("✓ Created derived features:")
print("  • revenue_efficiency: How efficiently we deliver revenue")
print("  • revenue_per_person: Revenue productivity per team member")
print("  • deadline_revenue_ratio: Revenue vs time pressure")

# ============================================================================
# STEP 4: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[STEP 4/7] Preparing Features and Target...")
print("-" * 70)

# Select features for training
feature_columns = [
    'deadline_days',
    'estimated_revenue',
    'team_size',
    'project_type_encoded',
    'complexity_encoded',
    'client_satisfaction',
    'revenue_per_person',
    'deadline_revenue_ratio'
]

X = df[feature_columns]
y = df['was_profitable']

print(f"✓ Selected {len(feature_columns)} features:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print(f"\n📊 Target Variable Distribution:")
profitable_count = y.sum()
not_profitable_count = len(y) - y.sum()
print(f"  ✓ Profitable:     {profitable_count:3d} projects ({profitable_count/len(y)*100:.1f}%)")
print(f"  ✓ Not Profitable: {not_profitable_count:3d} projects ({not_profitable_count/len(y)*100:.1f}%)")

# ============================================================================
# STEP 5: SPLIT DATA
# ============================================================================
print("\n[STEP 5/7] Splitting Data into Train/Test Sets...")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Training Set:   {len(X_train):3d} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Testing Set:    {len(X_test):3d} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================
print("\n[STEP 6/7] Training Random Forest Model...")
print("-" * 70)

# Initialize model with optimized parameters
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Maximum depth of trees
    min_samples_split=5,     # Minimum samples to split node
    min_samples_leaf=2,      # Minimum samples in leaf
    random_state=42,
    class_weight='balanced', # Handle class imbalance
    n_jobs=-1               # Use all CPU cores
)

print("⚙️  Model Parameters:")
print(f"  • n_estimators: {model.n_estimators}")
print(f"  • max_depth: {model.max_depth}")
print(f"  • min_samples_split: {model.min_samples_split}")
print(f"  • min_samples_leaf: {model.min_samples_leaf}")

print("\n🔄 Training in progress...")
model.fit(X_train, y_train)
print("✓ Model training completed!")

# Cross-validation
print("\n🔍 Performing 5-Fold Cross-Validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"✓ CV Scores: {cv_scores}")
print(f"✓ Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================
print("\n[STEP 7/7] Evaluating Model Performance...")
print("-" * 70)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 70)
print(" " * 20 + "MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"\n  Accuracy:  {accuracy*100:6.2f}%  (Overall correctness)")
print(f"  Precision: {precision*100:6.2f}%  (When predicting profitable, how often correct)")
print(f"  Recall:    {recall*100:6.2f}%  (Of all profitable projects, how many found)")
print(f"  F1-Score:  {f1*100:6.2f}%  (Harmonic mean of precision & recall)")

print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(
    y_test, y_pred, 
    target_names=['Not Profitable', 'Profitable'],
    digits=3
))

# Confusion Matrix
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"\n                    Predicted")
print(f"                Not Profitable | Profitable")
print(f"Actual Not Profitable:    {cm[0][0]:3d}    |    {cm[0][1]:3d}")
print(f"Actual Profitable:        {cm[1][0]:3d}    |    {cm[1][1]:3d}")

# Feature Importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Features ranked by importance:\n")
for idx, row in feature_importance.iterrows():
    bar_length = int(row['importance'] * 50)
    bar = '█' * bar_length
    print(f"  {row['feature']:30s} {bar} {row['importance']:.4f}")

# ============================================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 70)

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Package everything needed for prediction
model_package = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'feature_importance': feature_importance.to_dict('records')
}

# Save model
joblib.dump(model_package, 'models/project_predictor.pkl')
print("✓ Model saved to: models/project_predictor.pkl")

# Save metadata
metadata = {
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'cv_mean_score': float(cv_scores.mean()),
    'features': feature_columns
}

import json
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved to: models/model_metadata.json")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("viridis", len(feature_importance))
sns.barplot(
    data=feature_importance, 
    x='importance', 
    y='feature',
    palette=colors,
    ax=ax
)
ax.set_title('Feature Importance for Project Profitability Prediction', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/feature_importance.png")
plt.close()

# 2. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Not Profitable', 'Profitable'],
    yticklabels=['Not Profitable', 'Profitable'],
    cbar_kws={'label': 'Count'},
    ax=ax
)
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/confusion_matrix.png")
plt.close()

# 3. Performance Metrics Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors_metrics = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = ax.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black')
ax.set_ylim([0, 1.1])
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Threshold')
ax.legend()

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value*100:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('models/performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/performance_metrics.png")
plt.close()

# 4. Distribution of Predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Probability distribution for profitable projects
profitable_probs = y_pred_proba[y_test == 1][:, 1]
not_profitable_probs = y_pred_proba[y_test == 0][:, 1]

axes[0].hist(profitable_probs, bins=20, alpha=0.7, color='green', label='Actually Profitable', edgecolor='black')
axes[0].hist(not_profitable_probs, bins=20, alpha=0.7, color='red', label='Actually Not Profitable', edgecolor='black')
axes[0].set_xlabel('Predicted Probability of Being Profitable', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Prediction Probabilities', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')

# Revenue vs Profitability
profitable_mask = df['was_profitable'] == 1
axes[1].scatter(
    df[profitable_mask]['estimated_revenue'], 
    df[profitable_mask]['team_size'],
    c='green', alpha=0.6, label='Profitable', s=50, edgecolors='black'
)
axes[1].scatter(
    df[~profitable_mask]['estimated_revenue'], 
    df[~profitable_mask]['team_size'],
    c='red', alpha=0.6, label='Not Profitable', s=50, edgecolors='black'
)
axes[1].set_xlabel('Estimated Revenue (₹)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Team Size', fontsize=11, fontweight='bold')
axes[1].set_title('Revenue vs Team Size by Profitability', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('models/prediction_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/prediction_analysis.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print(" " * 25 + "✅ TRAINING COMPLETE!")
print("=" * 70)

print(f"""
📊 SUMMARY:
  • Training Samples:     {len(X_train)}
  • Testing Samples:      {len(X_test)}
  • Model Accuracy:       {accuracy*100:.2f}%
  • Cross-Validation:     {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)
  
📁 OUTPUT FILES:
  • models/project_predictor.pkl      (Trained model)
  • models/model_metadata.json         (Model info)
  • models/feature_importance.png      (Feature ranking)
  • models/confusion_matrix.png        (Prediction accuracy)
  • models/performance_metrics.png     (All metrics)
  • models/prediction_analysis.png     (Detailed analysis)

🚀 NEXT STEPS:
  1. Review the visualization images in the models/ folder
  2. Test the model with new predictions
  3. Integrate into your application
  
💡 The model is ready to make predictions!
""")

print("=" * 70)
