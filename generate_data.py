import numpy as np
import pandas as pd

def generate_patient_dataset(num_patients=5000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate Demographic Features
    age = np.random.randint(30, 91, size=num_patients)
    gender = np.random.choice(['Male', 'Female'], size=num_patients, p=[0.48, 0.52])
    
    # 2. Generate Hospital stay features
    # Length of stay follows a geometric-like distribution (mostly short, few long)
    time_in_hospital = np.random.geometric(p=0.25, size=num_patients)
    time_in_hospital = np.clip(time_in_hospital, 1, 14) # Clip to 1 to 14 days
    
    # Number of lab procedures (normally distributed, centered around 43)
    num_lab_procedures = np.random.normal(loc=43, scale=18, size=num_patients).astype(int)
    num_lab_procedures = np.clip(num_lab_procedures, 1, 100)
    
    # Number of non-lab procedures (mostly 0, 1, 2)
    num_procedures = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=num_patients, p=[0.45, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02])
    
    # Number of medications (log-normal distribution, centered around 15 meds)
    num_medications = np.random.lognormal(mean=2.6, sigma=0.5, size=num_patients).astype(int)
    num_medications = np.clip(num_medications, 1, 50)
    
    # 3. Prior Utilization (zero-inflated poisson-like)
    def zero_inflated_visits(max_val, p_zero, lam):
        val = np.random.poisson(lam, size=num_patients)
        zero_mask = np.random.binomial(1, p_zero, size=num_patients)
        val[zero_mask == 1] = 0
        return np.clip(val, 0, max_val)
        
    number_outpatient = zero_inflated_visits(10, 0.70, 1.5)
    number_emergency = zero_inflated_visits(10, 0.85, 1.0)
    number_inpatient = zero_inflated_visits(10, 0.80, 1.2)
    
    # 4. Clinical conditions
    diagnoses_list = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
    primary_diagnosis = np.random.choice(diagnoses_list, size=num_patients, p=[0.30, 0.14, 0.09, 0.08, 0.07, 0.05, 0.05, 0.04, 0.18])
    
    # Number of diagnoses (normally distributed, centered around 7)
    number_diagnoses = np.random.normal(loc=7.2, scale=2.0, size=num_patients).astype(int)
    number_diagnoses = np.clip(number_diagnoses, 1, 16)
    
    # Medical indicators
    high_A1C = np.random.choice([True, False], size=num_patients, p=[0.12, 0.88])
    med_change = np.random.choice([True, False], size=num_patients, p=[0.46, 0.54])
    on_diabetes_med = np.random.choice([True, False], size=num_patients, p=[0.77, 0.23])
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': [f'P{10000 + i}' for i in range(num_patients)],
        'age': age,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'primary_diagnosis': primary_diagnosis,
        'number_diagnoses': number_diagnoses,
        'high_A1C': high_A1C,
        'med_change': med_change,
        'on_diabetes_med': on_diabetes_med
    })
    
    # 5. Compute Risk Score & Target variable 'readmitted'
    # Log-odds calculation
    log_odds = -2.6 # Base intercept
    
    # Age factor (relative to center of 60)
    log_odds += 0.025 * (df['age'] - 60)
    
    # Stay factor
    log_odds += 0.08 * df['time_in_hospital']
    
    # Lab and clinical complexity
    log_odds += 0.005 * df['num_lab_procedures']
    log_odds += 0.015 * df['num_medications']
    log_odds += 0.06 * df['number_diagnoses']
    
    # Prior utilization factors (strong impact)
    log_odds += 0.08 * df['number_outpatient']
    log_odds += 0.28 * df['number_emergency']
    log_odds += 0.55 * df['number_inpatient']
    
    # Medical signs
    log_odds += 0.35 * df['high_A1C'].astype(int)
    log_odds += 0.30 * df['med_change'].astype(int)
    log_odds += 0.15 * df['on_diabetes_med'].astype(int)
    
    # Primary Diagnosis risk coefficients
    diag_coefs = {
        'Circulatory': 0.40,
        'Respiratory': 0.30,
        'Diabetes': 0.50,
        'Injury': -0.20,
        'Musculoskeletal': -0.35,
        'Neoplasms': 0.20,
        'Genitourinary': 0.15,
        'Digestive': 0.10,
        'Other': 0.00
    }
    
    diag_factor = df['primary_diagnosis'].map(diag_coefs)
    log_odds += diag_factor
    
    # Convert log-odds to probability
    probability = 1 / (1 + np.exp(-log_odds))
    
    # Draw binary outcome based on probability
    readmitted = np.random.binomial(1, probability)
    
    df['readmitted'] = readmitted
    
    # Return dataframe
    return df

if __name__ == '__main__':
    print("Generating synthetic patient dataset...")
    df = generate_patient_dataset(num_patients=5000, seed=42)
    output_path = 'patient_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path} successfully!")
    print(f"Total records: {len(df)}")
    print(f"Readmission Rate: {df['readmitted'].mean() * 100:.2f}%")
    print("\nFirst 5 rows:")
    print(df.head())
