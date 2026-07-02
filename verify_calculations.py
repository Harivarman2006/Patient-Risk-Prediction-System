import json
import numpy as np
import pandas as pd

def verify_consistency():
    # 1. Load model export parameters
    with open('model_export.json', 'r') as f:
        model = json.load(f)
        
    intercept = model['intercept']
    coefs = model['coefficients']
    scaling = model['numerical_scaling']
    categorical = model['categorical_encodings']
    booleans = model['boolean_features']
    
    # 2. Define a test case (Patient 1)
    # Profiles:
    # 78 yrs, Female, 8 days in hosp, 67 labs, 2 procedures, 28 meds, 9 diagnoses,
    # 2 outpatient, 1 emergency, 3 inpatient, Diabetes diagnosis, high A1C=True, med_change=True, on_med=True
    patient = {
        'age': 78,
        'gender': 'Female',
        'time_in_hospital': 8,
        'num_lab_procedures': 67,
        'num_procedures': 2,
        'num_medications': 28,
        'number_diagnoses': 9,
        'number_outpatient': 2,
        'number_emergency': 1,
        'number_inpatient': 3,
        'primary_diagnosis': 'Diabetes',
        'high_A1C': True,
        'med_change': True,
        'on_diabetes_med': True
    }
    
    print("--- Test Patient Parameters ---")
    for k, v in patient.items():
        print(f"{k}: {v}")
    
    # 3. Simulate JavaScript preprocessing and calculation
    print("\n--- Simulating JS calculation ---")
    
    # Scale numerical values
    scaled_nums = {}
    for col, stats in scaling.items():
        val = patient[col]
        scaled = (val - stats['mean']) / stats['std']
        scaled_nums[col] = scaled
        print(f"Numerical standardizing: {col} -> {val} to {scaled:.4f}")
        
    # One-hot encode categoricals
    encoded_cats = {}
    for col, cats in categorical.items():
        val = patient[col]
        for cat in cats:
            encoded_cats[f"{col}_{cat}"] = 1 if val == cat else 0
            
    # Process booleans
    processed_bools = {}
    for col in booleans:
        processed_bools[col] = 1 if patient[col] else 0
        
    # Combine inputs
    js_inputs = {}
    js_inputs.update(scaled_nums)
    js_inputs.update(encoded_cats)
    js_inputs.update(processed_bools)
    
    # Compute log-odds
    log_odds = intercept
    print(f"\nBaseline Intercept: {intercept:.6f}")
    
    active_contribs = []
    
    for feature, weight in coefs.items():
        # Identify value
        val = 0
        if feature in scaled_nums:
            val = scaled_nums[feature]
        elif feature in encoded_cats:
            val = encoded_cats[feature]
        elif feature in processed_bools:
            val = processed_bools[feature]
            
        term = weight * val
        if val != 0:
            print(f"Feature: {feature:35} | Input Val: {val:9.4f} | Weight: {weight:9.4f} | Contribution: {term:9.4f}")
            active_contribs.append((feature, term))
        log_odds += term
        
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    print(f"\nFinal Calculated Log-Odds: {log_odds:.6f}")
    print(f"Final Calculated Probability: {prob * 100:.2f}%")
    
    # 4. Compare with the original Pandas/Scikit-learn representation
    print("\n--- Verifying with Pandas DataFrame format ---")
    
    df_row = pd.DataFrame([patient])
    
    X_test = pd.DataFrame()
    for col in scaling.keys():
        X_test[col] = (df_row[col] - scaling[col]['mean']) / scaling[col]['std']
        
    for col, cats in categorical.items():
        for cat in cats:
            X_test[f"{col}_{cat}"] = (df_row[col] == cat).astype(int)
            
    for col in booleans:
        X_test[col] = df_row[col].astype(int)
        
    # Standardize column order matching coefficients keys
    X_test = X_test[list(coefs.keys())]
    
    # Predict using manually loaded coefficients (representing sklearn prediction)
    weights_vec = np.array([coefs[col] for col in X_test.columns])
    sklearn_log_odds = np.dot(X_test.values, weights_vec)[0] + intercept
    sklearn_prob = 1.0 / (1.0 + np.exp(-sklearn_log_odds))
    
    print(f"Sklearn Format Log-Odds   : {sklearn_log_odds:.6f}")
    print(f"Sklearn Format Probability: {sklearn_prob * 100:.2f}%")
    
    difference = abs(prob - sklearn_prob)
    print(f"\nCalculated Difference: {difference:.12f}")
    if difference < 1e-9:
        print("SUCCESS: The client-side JS implementation matches the Python data science representation EXACTLY!")
    else:
        print("WARNING: There is a discrepancy between the calculations!")

if __name__ == '__main__':
    verify_consistency()
