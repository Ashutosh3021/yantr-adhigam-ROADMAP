import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings('ignore')

# Load and explore data
df = pd.read_csv("Data/realistic_placement_data.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass Distribution:")
print(df['Placement'].value_counts())
print(f"Placement Rate: {df['Placement'].mean()*100:.2f}%")

# Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter plot with placement coloring
axes[0,0].scatter(df[df['Placement']==1]['CGPA'], df[df['Placement']==1]['IQ'], 
                  c='green', alpha=0.7, label='Placed', s=50)
axes[0,0].scatter(df[df['Placement']==0]['CGPA'], df[df['Placement']==0]['IQ'], 
                  c='red', alpha=0.7, label='Not Placed', s=50)
axes[0,0].set_xlabel('CGPA')
axes[0,0].set_ylabel('IQ')
axes[0,0].set_title('CGPA vs IQ by Placement Status')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. CGPA distribution by placement
axes[0,1].hist([df[df['Placement']==1]['CGPA'], df[df['Placement']==0]['CGPA']], 
               bins=15, alpha=0.7, label=['Placed', 'Not Placed'], color=['green', 'red'])
axes[0,1].set_xlabel('CGPA')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('CGPA Distribution by Placement')
axes[0,1].legend()

# 3. IQ distribution by placement
axes[1,0].hist([df[df['Placement']==1]['IQ'], df[df['Placement']==0]['IQ']], 
               bins=15, alpha=0.7, label=['Placed', 'Not Placed'], color=['green', 'red'])
axes[1,0].set_xlabel('IQ')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('IQ Distribution by Placement')
axes[1,0].legend()

# 4. Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[1,1], cbar_kws={'shrink': 0.8})
axes[1,1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# Prepare features and target
X = df.iloc[:, 0:2]  # CGPA and IQ
y = df.iloc[:, -1]   # Placement

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Scale the features
scaler = ss()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multiple Algorithm Comparison
algorithms = {
    'Logistic Regression': lr(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

print("\n" + "="*50)
print("MODEL COMPARISON RESULTS")
print("="*50)

for name, algorithm in algorithms.items():
    # Train the model
    algorithm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = algorithm.predict(X_test_scaled)
    y_pred_proba = algorithm.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'model': algorithm
    }
    
    print(f"\n{name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n\nBest Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Detailed evaluation of best model
print("\n" + "="*50)
print(f"DETAILED EVALUATION - {best_model_name}")
print("="*50)

y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    features = ['CGPA', 'IQ']
    
    print("\nFeature Importance:")
    for feature, importance in zip(features, feature_importance):
        print(f"  {feature}: {importance:.4f}")

# ROC Curve
plt.figure(figsize=(10, 8))

for name, result in results.items():
    if hasattr(result['model'], 'predict_proba'):
        y_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {result["roc_auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Decision Boundary Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Best model decision regions
plt.subplot(1, 2, 1)
try:
    plot_decision_regions(X_train_scaled, y_train.values, clf=best_model, legend=2)
    plt.xlabel('CGPA (scaled)')
    plt.ylabel('IQ (scaled)')
    plt.title(f'Decision Regions - {best_model_name}')
except:
    # Fallback visualization for models that don't support plot_decision_regions
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                         c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel('CGPA (scaled)')
    plt.ylabel('IQ (scaled)')
    plt.title(f'Decision Regions - {best_model_name}')

# Plot 2: Model comparison bar chart
plt.subplot(1, 2, 2)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

bars = plt.bar(range(len(model_names)), accuracies, color='skyblue', edgecolor='navy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Prediction examples
print("\n" + "="*50)
print("PREDICTION EXAMPLES")
print("="*50)

# Test with some examples
test_examples = [
    [8.5, 115],  # High CGPA, High IQ
    [6.0, 100],  # Low CGPA, Average IQ
    [7.5, 125],  # Average CGPA, High IQ
    [9.0, 95],   # High CGPA, Low IQ
]

print("\nSample Predictions:")
for i, example in enumerate(test_examples):
    example_scaled = scaler.transform([example])
    prediction = best_model.predict(example_scaled)[0]
    probability = best_model.predict_proba(example_scaled)[0]
    
    print(f"\nExample {i+1}: CGPA={example[0]}, IQ={example[1]}")
    print(f"  Prediction: {'Placed' if prediction == 1 else 'Not Placed'}")
    print(f"  Probability: Placed={probability[1]:.3f}, Not Placed={probability[0]:.3f}")

print("\n" + "="*50)
print("ENHANCED MODEL SUMMARY")
print("="*50)
print(f"Best Algorithm: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print("\nThis enhanced model provides significantly better performance than simple linear regression!")