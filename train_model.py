import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def train_and_export():
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # 1. Load data
    print("Loading patient dataset...")
    df = pd.read_csv('patient_data.csv')
    
    # 2. Preprocess data
    numerical_features = [
        'age', 'time_in_hospital', 'num_lab_procedures', 
        'num_procedures', 'num_medications', 'number_outpatient', 
        'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]
    categorical_features = ['gender', 'primary_diagnosis']
    boolean_features = ['high_A1C', 'med_change', 'on_diabetes_med']
    
    # Calculate and store scaling parameters
    scaling_params = {}
    X_processed = pd.DataFrame()
    
    for col in numerical_features:
        mean = df[col].mean()
        std = df[col].std()
        # Prevent division by zero
        if std == 0:
            std = 1.0
        X_processed[col] = (df[col] - mean) / std
        scaling_params[col] = {'mean': float(mean), 'std': float(std)}
        
    # One-hot encode categorical features and store category lists
    categorical_encodings = {}
    for col in categorical_features:
        # Sort categories to maintain strict order
        categories = sorted(list(df[col].unique()))
        categorical_encodings[col] = categories
        for cat in categories:
            X_processed[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            
    # Process boolean features
    for col in boolean_features:
        X_processed[col] = df[col].astype(int)
        
    y = df['readmitted']
    
    # Keep track of final columns
    feature_names = list(X_processed.columns)
    print(f"Total features after preprocessing: {len(feature_names)}")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 5. Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 6. Generate and Save Plots
    print("\nGenerating evaluation plots...")
    
    # Theme configuration for dark mode charts (matching the dashboard aesthetics)
    plt.style.use('dark_background')
    bg_color = '#0f172a' # Tailwind slate-900
    card_color = '#1e293b' # Tailwind slate-800
    accent_color = '#a855f7' # Tailwind purple-500
    cyan_color = '#06b6d4' # Tailwind cyan-500
    
    # 6.1 ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=bg_color)
    ax.set_facecolor(card_color)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=cyan_color, lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='#94a3b8', fontsize=11, labelpad=8)
    ax.set_ylabel('True Positive Rate', color='#94a3b8', fontsize=11, labelpad=8)
    ax.set_title('Receiver Operating Characteristic (ROC)', color='#f8fafc', fontsize=13, pad=15)
    ax.legend(loc="lower right", facecolor=card_color, edgecolor='#334155')
    ax.grid(True, color='#334155', linestyle=':', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    plt.tight_layout()
    plt.savefig('assets/roc_curve.png', dpi=150, facecolor=bg_color)
    plt.close()
    
    # 6.2 Feature Importance (Sorted Coefficients)
    coefs = model.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'AbsCoef': np.abs(coefs)
    }).sort_values(by='AbsCoef', ascending=True)
    
    # Take top 15 features or all if less
    importance_df = importance_df.tail(15)
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=bg_color)
    ax.set_facecolor(card_color)
    
    # Color positive coefficients green/cyan, negative red/orange
    colors = [accent_color if c >= 0 else '#f97316' for c in importance_df['Coefficient']]
    
    bars = ax.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, height=0.6)
    ax.axvline(0, color='#64748b', lw=1.2)
    ax.set_xlabel('Coefficient Value (Log-Odds Impact)', color='#94a3b8', fontsize=11, labelpad=8)
    ax.set_title('Model Feature Importances (Top Factors)', color='#f8fafc', fontsize=13, pad=15)
    ax.grid(True, color='#334155', axis='x', linestyle=':', alpha=0.6)
    
    # Add subtle grid lines for features
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    plt.savefig('assets/feature_importance.png', dpi=150, facecolor=bg_color)
    plt.close()
    
    # 7. Export parameters to JSON
    print("Exporting model parameters to JSON...")
    model_export = {
        'intercept': float(model.intercept_[0]),
        'coefficients': {name: float(coef) for name, coef in zip(feature_names, coefs)},
        'numerical_scaling': scaling_params,
        'categorical_encodings': categorical_encodings,
        'boolean_features': boolean_features,
        'metrics': {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': [[int(val) for val in row] for row in cm]
        }
    }
    
    with open('model_export.json', 'w') as f:
        json.dump(model_export, f, indent=2)
        
    print("Model training, plotting, and parameter export completed successfully!")

if __name__ == '__main__':
    train_and_export()
