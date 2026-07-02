// ==========================================================================
// AEGIS READMISSION RISK AI - CORE APPLICATION LOGIC
// ==========================================================================

// Global Model Parameters variable (initialized with fallback coefficients)
let modelParams = {
    "intercept": -0.7836670062865966,
    "coefficients": {
        "age": 0.47128678912743216,
        "time_in_hospital": 0.20906420508105103,
        "num_lab_procedures": 0.09557657031099542,
        "num_procedures": -0.07388533221707973,
        "num_medications": 0.12999280332540245,
        "number_outpatient": 0.02986010246523097,
        "number_emergency": 0.14828090519140108,
        "number_inpatient": 0.3372070184052924,
        "number_diagnoses": 0.1066090982146178,
        "gender_Female": -0.3774319969173122,
        "gender_Male": -0.31956159687779057,
        "primary_diagnosis_Circulatory": 0.0627939941904614,
        "primary_diagnosis_Diabetes": 0.30911191746212907,
        "primary_diagnosis_Digestive": -0.12492997970598399,
        "primary_diagnosis_Genitourinary": -0.06670855412078368,
        "primary_diagnosis_Injury": -0.38298460683061647,
        "primary_diagnosis_Musculoskeletal": -0.45650629097502143,
        "primary_diagnosis_Neoplasms": -0.09738409878352386,
        "primary_diagnosis_Other": -0.09953700913212067,
        "primary_diagnosis_Respiratory": 0.15915103410035583,
        "high_A1C": 0.46768486391600256,
        "med_change": 0.3008063300857385,
        "on_diabetes_med": 0.1920855294854173
    },
    "numerical_scaling": {
        "age": { "mean": 60.2024, "std": 17.561136436534756 },
        "time_in_hospital": { "mean": 3.932, "std": 3.143970882591331 },
        "num_lab_procedures": { "mean": 42.4916, "std": 17.686358893174695 },
        "num_procedures": { "mean": 1.281, "std": 1.5550313902056463 },
        "num_medications": { "mean": 14.8352, "std": 7.836116506559922 },
        "number_outpatient": { "mean": 0.456, "std": 0.9626262669727921 },
        "number_emergency": { "mean": 0.1514, "std": 0.5319159961172301 },
        "number_inpatient": { "mean": 0.2306, "std": 0.682287874250063 },
        "number_diagnoses": { "mean": 6.7266, "std": 2.007550370899405 }
    },
    "categorical_encodings": {
        "gender": ["Female", "Male"],
        "primary_diagnosis": ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Injury", "Musculoskeletal", "Neoplasms", "Other", "Respiratory"]
    },
    "boolean_features": ["high_A1C", "med_change", "on_diabetes_med"]
};

// Current active evaluation session data
let currentEvaluation = null;

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

// Main initialization function
function initApp() {
    // 1. Load model parameters from JSON (if served on a server)
    fetchModelParams();
    
    // 2. Setup Navigation Routing
    setupNavigation();
    
    // 3. Setup form input controllers (sliders, counters, buttons)
    setupFormControllers();
    
    // 4. Load Saved Patient Log
    loadSavedHistory();
    
    // 5. Initialize Lucide Icons
    lucide.createIcons();
}

// Fetch model parameters via AJAX, fail gracefully to fallback values
function fetchModelParams() {
    fetch('model_export.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not OK');
            }
            return response.json();
        })
        .then(data => {
            console.log('Successfully loaded model_export.json from server!');
            modelParams = data;
            // Update UI with model statistics if available
            updateModelInfoUI();
        })
        .catch(err => {
            console.warn('Could not fetch model_export.json directly (likely running locally via file:// protocol). Using robust offline fallback weights.', err);
            updateModelInfoUI();
        });
}

// Update model stats displayed on the tabs
function updateModelInfoUI() {
    if (modelParams.metrics) {
        const aucVal = document.getElementById('model-auc-val');
        if (aucVal) {
            aucVal.textContent = modelParams.metrics.roc_auc.toFixed(3);
        }
    }
}

// 1. Navigation Controller (Sidebar Tabs)
function setupNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const contentPanels = document.querySelectorAll('.content-panel');
    const tabTitle = document.getElementById('tab-title');
    const tabDescription = document.getElementById('tab-description');

    const tabMetadata = {
        'calculator': {
            title: 'Patient Risk Calculator',
            desc: 'Assess 30-day hospital readmission risk for discharged patients using machine learning models.'
        },
        'cohort-analytics': {
            title: 'Cohort Analytics',
            desc: 'Explore demographics, admission types, and statistical trends across the 5,000 patient cohort.'
        },
        'model-performance': {
            title: 'Model Performance & Validation',
            desc: 'Review data science pipeline metrics, confusion matrices, ROC curves, and mathematical formulations.'
        },
        'history': {
            title: 'Saved Evaluations History',
            desc: 'Access saved patient evaluation logs, search records, and filter by clinical risk levels.'
        }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Toggle buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle panels
            contentPanels.forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');
            
            // Update headers
            if (tabMetadata[targetTab]) {
                tabTitle.textContent = tabMetadata[targetTab].title;
                tabDescription.textContent = tabMetadata[targetTab].desc;
            }
            
            // Re-render history list if history tab is selected
            if (targetTab === 'history') {
                loadSavedHistory();
            }
        });
    });
}

// 2. Form Input Handling (Sliders, Counters, Toggles)
function setupFormControllers() {
    // Range Slider Bubbles
    const sliders = [
        { id: 'age', valId: 'age-val' },
        { id: 'time_in_hospital', valId: 'time_in_hospital-val' },
        { id: 'num_lab_procedures', valId: 'num_lab_procedures-val' },
        { id: 'num_medications', valId: 'num_medications-val' }
    ];

    sliders.forEach(sliderInfo => {
        const slider = document.getElementById(sliderInfo.id);
        const bubble = document.getElementById(sliderInfo.valId);
        
        if (slider && bubble) {
            // Update bubble text on input
            slider.addEventListener('input', () => {
                bubble.textContent = slider.value;
            });
        }
    });

    // Custom Numeric Counter Controls (+ / - Buttons)
    const decButtons = document.querySelectorAll('.btn-num-dec');
    const incButtons = document.querySelectorAll('.btn-num-inc');

    decButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const input = e.target.parentElement.querySelector('input[type="number"]');
            if (input) {
                const min = parseInt(input.getAttribute('min')) || 0;
                let currentVal = parseInt(input.value) || 0;
                if (currentVal > min) {
                    input.value = currentVal - 1;
                }
            }
        });
    });

    incButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const input = e.target.parentElement.querySelector('input[type="number"]');
            if (input) {
                const max = parseInt(input.getAttribute('max')) || 100;
                let currentVal = parseInt(input.value) || 0;
                if (currentVal < max) {
                    input.value = currentVal + 1;
                }
            }
        });
    });

    // Fill Demo Patient button
    const fillRandomBtn = document.getElementById('btn-fill-random');
    if (fillRandomBtn) {
        fillRandomBtn.addEventListener('click', () => {
            fillDemoPatientData();
        });
    }

    // Reset button listener to hide results panel
    const form = document.getElementById('risk-calculator-form');
    if (form) {
        form.addEventListener('reset', () => {
            document.getElementById('result-details').style.display = 'none';
            document.getElementById('result-placeholder').style.display = 'flex';
            currentEvaluation = null;
            
            // Reset slider val bubbles manually
            setTimeout(() => {
                sliders.forEach(s => {
                    const el = document.getElementById(s.id);
                    const valEl = document.getElementById(s.valId);
                    if (el && valEl) valEl.textContent = el.value;
                });
            }, 50);
        });

        // Submit form (calculate risk)
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            calculatePatientRisk();
        });
    }

    // Save record button
    const saveBtn = document.getElementById('btn-save-record');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            saveCurrentEvaluation();
        });
    }

    // Print report button
    const printBtn = document.getElementById('btn-print-report');
    if (printBtn) {
        printBtn.addEventListener('click', () => {
            window.print();
        });
    }

    // Search and Filter History Listeners
    const searchHistory = document.getElementById('search-history');
    if (searchHistory) {
        searchHistory.addEventListener('input', filterHistoryTable);
    }
    const filterRisk = document.getElementById('filter-risk-level');
    if (filterRisk) {
        filterRisk.addEventListener('change', filterHistoryTable);
    }
    
    // Clear history button
    const clearHistoryBtn = document.getElementById('btn-clear-history');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the entire saved evaluation log? This cannot be undone.')) {
                localStorage.removeItem('aegis_evaluations');
                loadSavedHistory();
            }
        });
    }
}

// Fill random realistic clinical data into the form for demo purposes
function fillDemoPatientData() {
    const ageSlider = document.getElementById('age');
    const hospitalSlider = document.getElementById('time_in_hospital');
    const labSlider = document.getElementById('num_lab_procedures');
    const medSlider = document.getElementById('num_medications');
    
    const numProceduresInput = document.getElementById('num_procedures');
    const numDiagInput = document.getElementById('number_diagnoses');
    const numOutpatientInput = document.getElementById('number_outpatient');
    const numEmergencyInput = document.getElementById('number_emergency');
    const numInpatientInput = document.getElementById('number_inpatient');
    
    const primaryDiagSelect = document.getElementById('primary_diagnosis');
    
    const highA1CCheck = document.getElementById('high_A1C');
    const medChangeCheck = document.getElementById('med_change');
    const onDiabetesCheck = document.getElementById('on_diabetes_med');

    // Profiles:
    const profiles = [
        // 1. High-risk elderly diabetic patient
        {
            age: 78, time_in_hospital: 8, num_lab_procedures: 67, num_medications: 28,
            num_procedures: 2, number_diagnoses: 9, outpatient: 2, emergency: 1, inpatient: 3,
            diagnosis: 'Diabetes', high_A1C: true, med_change: true, on_med: true, gender: 'Female'
        },
        // 2. Young healthy patient post-injury
        {
            age: 34, time_in_hospital: 2, num_lab_procedures: 25, num_medications: 8,
            num_procedures: 1, number_diagnoses: 3, outpatient: 0, emergency: 0, inpatient: 0,
            diagnosis: 'Injury', high_A1C: false, med_change: false, on_med: false, gender: 'Male'
        },
        // 3. Moderate-risk circulatory patient
        {
            age: 63, time_in_hospital: 5, num_lab_procedures: 48, num_medications: 18,
            num_procedures: 0, number_diagnoses: 6, outpatient: 1, emergency: 0, inpatient: 1,
            diagnosis: 'Circulatory', high_A1C: false, med_change: true, on_med: true, gender: 'Male'
        },
        // 4. High-risk respiratory patient
        {
            age: 82, time_in_hospital: 6, num_lab_procedures: 52, num_medications: 22,
            num_procedures: 1, number_diagnoses: 8, outpatient: 3, emergency: 2, inpatient: 1,
            diagnosis: 'Respiratory', high_A1C: false, med_change: false, on_med: true, gender: 'Female'
        }
    ];

    // Pick a random profile
    const profile = profiles[Math.floor(Math.random() * profiles.length)];
    
    // Fill sliders
    ageSlider.value = profile.age;
    document.getElementById('age-val').textContent = profile.age;
    
    hospitalSlider.value = profile.time_in_hospital;
    document.getElementById('time_in_hospital-val').textContent = profile.time_in_hospital;
    
    labSlider.value = profile.num_lab_procedures;
    document.getElementById('num_lab_procedures-val').textContent = profile.num_lab_procedures;
    
    medSlider.value = profile.num_medications;
    document.getElementById('num_medications-val').textContent = profile.num_medications;

    // Fill numbers
    numProceduresInput.value = profile.num_procedures;
    numDiagInput.value = profile.number_diagnoses;
    numOutpatientInput.value = profile.outpatient;
    numEmergencyInput.value = profile.emergency;
    numInpatientInput.value = profile.inpatient;

    // Selects
    primaryDiagSelect.value = profile.diagnosis;
    
    // Radios
    const genderRadios = document.getElementsByName('gender');
    genderRadios.forEach(radio => {
        radio.checked = (radio.value === profile.gender);
    });

    // Checkboxes
    highA1CCheck.checked = profile.high_A1C;
    medChangeCheck.checked = profile.med_change;
    onDiabetesCheck.checked = profile.on_med;
}

// 3. Clinical Calculation Engine (Logistic Regression Prediction)
function calculatePatientRisk() {
    const form = document.getElementById('risk-calculator-form');
    const formData = new FormData(form);
    
    // Gather patient inputs
    const age = parseInt(formData.get('age'));
    const gender = formData.get('gender');
    const timeInHospital = parseInt(formData.get('time_in_hospital'));
    const numLabProcedures = parseInt(formData.get('num_lab_procedures'));
    const numProcedures = parseInt(formData.get('num_procedures'));
    const numMedications = parseInt(formData.get('num_medications'));
    const numberDiagnoses = parseInt(formData.get('number_diagnoses'));
    const numberOutpatient = parseInt(formData.get('number_outpatient'));
    const numberEmergency = parseInt(formData.get('number_emergency'));
    const numberInpatient = parseInt(formData.get('number_inpatient'));
    const primaryDiagnosis = formData.get('primary_diagnosis');
    
    const highA1C = document.getElementById('high_A1C').checked;
    const medChange = document.getElementById('med_change').checked;
    const onDiabetesMed = document.getElementById('on_diabetes_med').checked;

    // 3.1 Preprocess input vector (Standardization of numerical values)
    const scale = (val, col) => {
        const stats = modelParams.numerical_scaling[col];
        return (val - stats.mean) / stats.std;
    };

    const scaledInputs = {
        'age': scale(age, 'age'),
        'time_in_hospital': scale(timeInHospital, 'time_in_hospital'),
        'num_lab_procedures': scale(numLabProcedures, 'num_lab_procedures'),
        'num_procedures': scale(numProcedures, 'num_procedures'),
        'num_medications': scale(numMedications, 'num_medications'),
        'number_outpatient': scale(numberOutpatient, 'number_outpatient'),
        'number_emergency': scale(numberEmergency, 'number_emergency'),
        'number_inpatient': scale(numberInpatient, 'number_inpatient'),
        'number_diagnoses': scale(numberDiagnoses, 'number_diagnoses')
    };

    // One-hot encode inputs
    const oneHotInputs = {};
    modelParams.categorical_encodings.gender.forEach(g => {
        oneHotInputs[`gender_${g}`] = (gender === g) ? 1 : 0;
    });
    modelParams.categorical_encodings.primary_diagnosis.forEach(diag => {
        oneHotInputs[`primary_diagnosis_${diag}`] = (primaryDiagnosis === diag) ? 1 : 0;
    });

    // Boolean inputs
    const boolInputs = {
        'high_A1C': highA1C ? 1 : 0,
        'med_change': medChange ? 1 : 0,
        'on_diabetes_med': onDiabetesMed ? 1 : 0
    };

    // 3.2 Compute raw log-odds (dot product)
    let logOdds = modelParams.intercept;
    const contributions = [];

    // Loop through coefficients
    for (const [feature, weight] of Object.entries(modelParams.coefficients)) {
        let value = 0;
        let displayName = feature;
        let displayValStr = '';
        
        if (feature in scaledInputs) {
            value = scaledInputs[feature];
            
            // Format labels for UI feature breakdown
            const prettyNames = {
                'number_inpatient': 'Prior Inpatient Admissions',
                'number_emergency': 'Prior Emergency Visits',
                'age': 'Patient Age',
                'time_in_hospital': 'Length of Stay',
                'num_medications': 'Medication Complexity',
                'number_diagnoses': 'Comorbidities (Diagnoses Count)',
                'num_lab_procedures': 'Lab Testing Intensity',
                'number_outpatient': 'Prior Outpatient Visits',
                'num_procedures': 'In-Hospital Procedures'
            };
            displayName = prettyNames[feature];
            
            // Format input value display
            const rawVal = {
                'age': `${age} yrs`,
                'time_in_hospital': `${timeInHospital} days`,
                'num_lab_procedures': `${numLabProcedures} tests`,
                'num_medications': `${numMedications} medications`,
                'number_diagnoses': `${numberDiagnoses} diagnoses`,
                'number_inpatient': `${numberInpatient} visits`,
                'number_emergency': `${numberEmergency} visits`,
                'number_outpatient': `${numberOutpatient} visits`,
                'num_procedures': `${numProcedures} procedures`
            }[feature];
            displayValStr = rawVal;
            
        } else if (feature in oneHotInputs) {
            value = oneHotInputs[feature];
            
            // Only add active categories to explanations
            if (value === 1) {
                if (feature.startsWith('primary_diagnosis_')) {
                    const diagnosisName = feature.replace('primary_diagnosis_', '');
                    displayName = `Primary Diagnosis: ${diagnosisName}`;
                    displayValStr = 'Active';
                } else if (feature.startsWith('gender_')) {
                    const genderName = feature.replace('gender_', '');
                    displayName = `Gender: ${genderName}`;
                    displayValStr = 'Active';
                }
            } else {
                // Skip inactive categorical columns in dot product sum
                // (Since value is 0, they add 0, so logOdds += 0. Just proceed.)
                continue;
            }
        } else if (feature in boolInputs) {
            value = boolInputs[feature];
            if (value === 1) {
                const prettyBools = {
                    'high_A1C': 'Elevated HbA1c (>8%)',
                    'med_change': 'Discharge Medication Changed',
                    'on_diabetes_med': 'On Active Diabetes Meds'
                };
                displayName = prettyBools[feature];
                displayValStr = 'Yes';
            } else {
                continue; // Skip inactive boolean features
            }
        }

        const contribution = weight * value;
        logOdds += contribution;

        // Save active contributions for visual bar chart explanation
        contributions.push({
            name: displayName,
            value: displayValStr,
            coef: weight,
            impact: contribution // Impact on log-odds
        });
    }

    // 3.3 Apply logistic sigmoid to get probability
    const probability = 1 / (1 + Math.exp(-logOdds));
    const riskPercentage = Math.round(probability * 100);

    // Save calculation session data
    currentEvaluation = {
        patientId: `P${10000 + Math.floor(Math.random() * 9000)}`,
        date: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
        age: age,
        gender: gender,
        primaryDiagnosis: primaryDiagnosis,
        timeInHospital: timeInHospital,
        riskScore: riskPercentage,
        rawProbability: probability,
        contributions: contributions,
        inputs: {
            age, gender, timeInHospital, numLabProcedures, numProcedures, numMedications, 
            numberDiagnoses, numberOutpatient, numberEmergency, numberInpatient, primaryDiagnosis,
            highA1C, medChange, onDiabetesMed
        }
    };

    // 3.4 Render outputs on the UI
    renderPredictionResults(currentEvaluation);
}

// Render calculation outputs in the UI (gauge, indicators, recommendations)
function renderPredictionResults(data) {
    const placeholder = document.getElementById('result-placeholder');
    const details = document.getElementById('result-details');
    const scoreVal = document.getElementById('risk-score-value');
    const gaugeFill = document.getElementById('gauge-fill-circle');
    const badge = document.getElementById('risk-badge-level');
    const interpretation = document.getElementById('risk-interpretation-text');
    const patientIdDisplay = document.getElementById('patient-id-display');
    const contributionsList = document.getElementById('contributions-list');
    const carePlanList = document.getElementById('care-plan-list');
    const saveBtn = document.getElementById('btn-save-record');

    // Show result details panel, hide placeholder
    placeholder.style.display = 'none';
    details.style.display = 'flex';

    // Set Patient ID tag
    patientIdDisplay.textContent = data.patientId;
    saveBtn.disabled = false;
    saveBtn.innerHTML = '<i data-lucide="save"></i> Save to Evaluation Log';
    
    // Set score text
    scoreVal.textContent = `${data.riskScore}%`;

    // Animate SVG Gauge
    // Circumference of circle with r=50 is 2 * PI * 50 = 314.16
    const circumference = 314.16;
    const offset = circumference - (data.riskScore / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;

    // Risk category threshold levels
    let category = '';
    let categoryClass = '';
    let descriptionText = '';

    if (data.riskScore < 25) {
        category = 'Low Risk';
        categoryClass = 'low';
        descriptionText = 'The patient has a low risk of readmission within 30 days. Plan standard outpatient follow-up protocols.';
    } else if (data.riskScore >= 25 && data.riskScore < 45) {
        category = 'Moderate Risk';
        categoryClass = 'moderate';
        descriptionText = 'The patient shows moderate readmission warning signs. Recommended telephone check-in within 72 hours.';
    } else if (data.riskScore >= 45 && data.riskScore < 70) {
        category = 'High Risk';
        categoryClass = 'high';
        descriptionText = 'The patient is at high risk for readmission. Prioritize a primary care/specialist visit within 5 days and set up transitional nursing care.';
    } else {
        category = 'Critical Risk';
        categoryClass = 'critical';
        descriptionText = 'The patient is at critical risk of returning to the hospital. Establish immediate care protocols, verify medication compliance, and schedule home health nursing within 48 hours.';
    }

    // Set gauge color and badge class
    gaugeFill.className.baseVal = `gauge-fill risk-${categoryClass}`;
    badge.textContent = category;
    badge.className = `risk-badge ${categoryClass}`;
    interpretation.textContent = descriptionText;

    // Render Explainable AI Feature Breakdown
    contributionsList.innerHTML = '';
    
    // Sort contributions by absolute impact (highest absolute impact first)
    const sortedContribs = [...data.contributions].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
    
    // Take top 5 key drivers
    const topContribs = sortedContribs.slice(0, 5);

    // Calculate max absolute impact to normalize progress bar widths
    const maxImpact = Math.max(...topContribs.map(c => Math.abs(c.impact)), 0.1);

    topContribs.forEach(c => {
        const item = document.createElement('div');
        item.className = 'contrib-item';
        
        const isPositive = c.impact >= 0;
        const impactPercent = Math.min(Math.round((Math.abs(c.impact) / maxImpact) * 100), 100);
        
        const impactText = isPositive 
            ? `+${Math.abs(c.impact * 10).toFixed(1)}% Risk Shift`
            : `-${Math.abs(c.impact * 10).toFixed(1)}% Protective`;

        item.innerHTML = `
            <div class="contrib-meta">
                <span class="contrib-name">${c.name} <span style="color: var(--text-muted); font-size: 0.75rem;">(${c.value})</span></span>
                <span class="contrib-score ${isPositive ? 'positive' : 'negative'}">${impactText}</span>
            </div>
            <div class="contrib-bar-wrapper">
                <div class="contrib-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${impactPercent}%"></div>
            </div>
        `;
        contributionsList.appendChild(item);
    });

    // Render Tailored Care Plan / Interventions list
    carePlanList.innerHTML = '';
    const carePlanItems = [];

    // Core Risk-based interventions
    if (data.riskScore < 25) {
        carePlanItems.push({
            title: 'Primary Care Follow-Up (Routine)',
            desc: 'Schedule a standard follow-up appointment within 10-14 days of discharge.',
            severity: 'standard'
        });
        carePlanItems.push({
            title: 'Medication Teach-Back',
            desc: 'Verify understanding of current outpatient drugs and discharge changes.',
            severity: 'standard'
        });
    } else if (data.riskScore >= 25 && data.riskScore < 45) {
        carePlanItems.push({
            title: 'Nurse Telephone Follow-up (72 Hours)',
            desc: 'Initiate post-discharge contact to review symptoms, medications, and wellness.',
            severity: 'warning'
        });
        carePlanItems.push({
            title: 'Outpatient Follow-Up (7 Days)',
            desc: 'Schedule appointment with primary care provider (PCP) within 7 days.',
            severity: 'standard'
        });
        carePlanItems.push({
            title: 'Red Flags Warning Review',
            desc: 'Educate patient/family on specific symptoms requiring immediate emergency care.',
            severity: 'warning'
        });
    } else if (data.riskScore >= 45 && data.riskScore < 70) {
        carePlanItems.push({
            title: 'Urgent PCP Appointment (3-5 Days)',
            desc: 'Schedule priority check-in with primary doctor or relevant specialist.',
            severity: 'critical'
        });
        carePlanItems.push({
            title: 'Home Health Transition Visit (5 Days)',
            desc: 'Deploy transitional care nurse to perform safety, physical, and compliance reviews at home.',
            severity: 'warning'
        });
        carePlanItems.push({
            title: 'Pharmacist Prescription Reconciliation',
            desc: 'Dedicated clinical pharmacist reconciles pre-admission, in-hospital, and discharge drugs.',
            severity: 'standard'
        });
    } else {
        carePlanItems.push({
            title: 'Priority Doctor Appointment (48 Hours)',
            desc: 'Arrange priority clinic evaluation within 48 hours. Ensure transport is lined up.',
            severity: 'critical'
        });
        carePlanItems.push({
            title: 'Urgent Home Nurse Visit (24-48 Hours)',
            desc: 'Schedule immediate home nursing care to evaluate vital signs, wound care, and drug adherence.',
            severity: 'critical'
        });
        carePlanItems.push({
            title: 'Transition Case Manager Assignment',
            desc: 'A dedicated clinical manager oversees the patient\'s care plan and coordinates follow-ups.',
            severity: 'warning'
        });
        carePlanItems.push({
            title: '24/7 Nurse Help Line access',
            desc: 'Provide direct contact number for urgent questions to prevent ED utilization.',
            severity: 'warning'
        });
    }

    // Input-specific interventions (triggered by key patient attributes)
    if (data.inputs.numberInpatient >= 2 || data.inputs.numberEmergency >= 1) {
        carePlanItems.push({
            title: 'High-Utilizer Care Coordination Protocol',
            desc: 'Enroll patient in the hospital\'s multidisciplinary outpatient support program for complex cases.',
            severity: 'critical'
        });
    }
    
    if (data.inputs.highA1C) {
        carePlanItems.push({
            title: 'Endocrine Referral & Diabetes Education',
            desc: 'Schedule referral to outpatient diabetes educator and nutritionist. Perform insulin teach-back.',
            severity: 'critical'
        });
    }

    if (data.inputs.medChange) {
        carePlanItems.push({
            title: 'Medication Reconciliation & Delivery',
            desc: 'Provide physical list of altered dosages. Arrange direct delivery of new drugs to patient\'s bedside prior to discharge.',
            severity: 'warning'
        });
    }

    if (data.inputs.timeInHospital >= 7) {
        carePlanItems.push({
            title: 'Physical Therapy Assessment',
            desc: 'Assess patient mobility, durability, and require physical/occupational rehab services.',
            severity: 'standard'
        });
    }

    // Append to UI list
    carePlanItems.forEach(item => {
        const li = document.createElement('li');
        li.className = `care-plan-item ${item.severity}-recommendation`;
        
        let iconType = 'info';
        if (item.severity === 'critical') iconType = 'alert-triangle';
        else if (item.severity === 'warning') iconType = 'phone-call';
        else iconType = 'check-circle2';

        li.innerHTML = `
            <i data-lucide="${iconType}"></i>
            <div class="care-plan-text">
                <span class="care-plan-title">${item.title}</span>
                <span class="care-plan-desc">${item.desc}</span>
            </div>
        `;
        carePlanList.appendChild(li);
    });

    // Re-trigger Lucide icon render for new items
    lucide.createIcons();
}

// 4. Local Storage Saved History Manager
function saveCurrentEvaluation() {
    if (!currentEvaluation) return;

    // Get existing logs
    let savedList = [];
    const localData = localStorage.getItem('aegis_evaluations');
    if (localData) {
        try {
            savedList = JSON.parse(localData);
        } catch (e) {
            console.error('Error parsing saved local evaluations, clearing corrupted log.', e);
        }
    }

    // Check if patient was already saved in this session (avoid duplicate button clicking)
    const exists = savedList.find(p => p.patientId === currentEvaluation.patientId);
    if (exists) {
        alert('This evaluation record has already been logged.');
        return;
    }

    // Save to beginning of array
    savedList.unshift(currentEvaluation);
    localStorage.setItem('aegis_evaluations', JSON.stringify(savedList));

    // Disable button to show it worked
    const saveBtn = document.getElementById('btn-save-record');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i data-lucide="check"></i> Saved to Evaluation Log';
    lucide.createIcons();
}

// Load evaluations from localStorage and populate the history tab table
function loadSavedHistory() {
    const tableBody = document.getElementById('history-table-body');
    if (!tableBody) return;

    let savedList = [];
    const localData = localStorage.getItem('aegis_evaluations');
    if (localData) {
        try {
            savedList = JSON.parse(localData);
        } catch (e) {
            console.error(e);
        }
    }

    tableBody.innerHTML = '';

    if (savedList.length === 0) {
        tableBody.innerHTML = `
            <tr class="table-placeholder">
                <td colspan="8">
                    <div class="table-placeholder-content">
                        <i data-lucide="folder-open"></i>
                        <p>No patient records evaluated and saved yet.</p>
                    </div>
                </td>
            </tr>
        `;
        lucide.createIcons();
        return;
    }

    savedList.forEach(p => {
        const tr = document.createElement('tr');
        tr.className = 'history-row';
        tr.setAttribute('data-patient-id', p.patientId);

        let riskClass = 'low';
        if (p.riskScore >= 25 && p.riskScore < 45) riskClass = 'moderate';
        else if (p.riskScore >= 45 && p.riskScore < 70) riskClass = 'high';
        else if (p.riskScore >= 70) riskClass = 'critical';

        tr.innerHTML = `
            <td><strong>${p.patientId}</strong></td>
            <td>${p.date.split(',')[0]} <span style="color: var(--text-muted); font-size: 0.75rem;">${p.date.split(',')[1] || ''}</span></td>
            <td>${p.age} yrs / ${p.gender}</td>
            <td>${p.primaryDiagnosis}</td>
            <td>${p.timeInHospital} Days</td>
            <td><strong class="indicator-${riskClass}">${p.riskScore}%</strong></td>
            <td><span class="badge badge-${riskClass === 'low' ? 'success' : riskClass === 'moderate' ? 'warning' : 'danger'}">${riskClass}</span></td>
            <td>
                <div class="action-row-buttons">
                    <button class="btn btn-secondary btn-xs btn-load-patient" data-id="${p.patientId}">
                        <i data-lucide="external-link" style="width:12px;height:12px;"></i> Load
                    </button>
                    <button class="btn btn-secondary btn-xs btn-delete-patient" data-id="${p.patientId}" style="border-color: rgba(239,68,68,0.15); color: var(--critical);">
                        <i data-lucide="trash-2" style="width:12px;height:12px;"></i>
                    </button>
                </div>
            </td>
        `;
        tableBody.appendChild(tr);
    });

    // Add load button listeners
    document.querySelectorAll('.btn-load-patient').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const pId = btn.getAttribute('data-id');
            loadPatientIntoCalculator(pId);
        });
    });

    // Add delete button listeners
    document.querySelectorAll('.btn-delete-patient').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const pId = btn.getAttribute('data-id');
            deletePatientRecord(pId);
        });
    });

    lucide.createIcons();
}

// Reload a saved patient's inputs back into the calculator form
function loadPatientIntoCalculator(patientId) {
    const localData = localStorage.getItem('aegis_evaluations');
    if (!localData) return;

    const savedList = JSON.parse(localData);
    const patient = savedList.find(p => p.patientId === patientId);
    if (!patient) return;

    // Load inputs
    const inputs = patient.inputs;

    document.getElementById('age').value = inputs.age;
    document.getElementById('age-val').textContent = inputs.age;

    document.getElementById('time_in_hospital').value = inputs.timeInHospital;
    document.getElementById('time_in_hospital-val').textContent = inputs.timeInHospital;

    document.getElementById('num_lab_procedures').value = inputs.numLabProcedures;
    document.getElementById('num_lab_procedures-val').textContent = inputs.numLabProcedures;

    document.getElementById('num_medications').value = inputs.numMedications;
    document.getElementById('num_medications-val').textContent = inputs.numMedications;

    document.getElementById('num_procedures').value = inputs.numProcedures;
    document.getElementById('number_diagnoses').value = inputs.numberDiagnoses;
    document.getElementById('number_outpatient').value = inputs.numberOutpatient;
    document.getElementById('number_emergency').value = inputs.numberEmergency;
    document.getElementById('number_inpatient').value = inputs.numberInpatient;

    document.getElementById('primary_diagnosis').value = inputs.primaryDiagnosis;

    const genderRadios = document.getElementsByName('gender');
    genderRadios.forEach(radio => {
        radio.checked = (radio.value === inputs.gender);
    });

    document.getElementById('high_A1C').checked = inputs.highA1C;
    document.getElementById('med_change').checked = inputs.medChange;
    document.getElementById('on_diabetes_med').checked = inputs.onDiabetesMed;

    // Switch to Calculator tab
    const calcBtn = document.querySelector('.nav-btn[data-tab="calculator"]');
    if (calcBtn) calcBtn.click();

    // Trigger calculation
    calculatePatientRisk();

    // Restore Patient ID display from history (so it keeps its generated ID)
    currentEvaluation.patientId = patientId;
    document.getElementById('patient-id-display').textContent = patientId;
    
    // Disable save button since it's already saved
    const saveBtn = document.getElementById('btn-save-record');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i data-lucide="check"></i> Saved to Evaluation Log';
    lucide.createIcons();
}

// Delete a single patient record from localStorage
function deletePatientRecord(patientId) {
    const localData = localStorage.getItem('aegis_evaluations');
    if (!localData) return;

    let savedList = JSON.parse(localData);
    savedList = savedList.filter(p => p.patientId !== patientId);
    
    localStorage.setItem('aegis_evaluations', JSON.stringify(savedList));
    loadSavedHistory();
}

// Filter the Saved History table by Patient ID search and Risk Category dropdown
function filterHistoryTable() {
    const query = document.getElementById('search-history').value.toLowerCase().trim();
    const filter = document.getElementById('filter-risk-level').value;
    const rows = document.querySelectorAll('.history-row');

    rows.forEach(row => {
        const patientId = row.querySelector('td strong').textContent.toLowerCase();
        const badge = row.querySelector('.badge').textContent.toLowerCase(); // 'low', 'moderate', 'high', 'critical'
        
        let matchQuery = patientId.includes(query);
        let matchFilter = (filter === 'ALL') || (badge === filter.toLowerCase());

        if (matchQuery && matchFilter) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
    
    // Show placeholder if all rows are hidden
    const tableBody = document.getElementById('history-table-body');
    const visibleRows = Array.from(rows).filter(r => r.style.display !== 'none');
    
    // Remove old search placeholders if they exist
    const oldSearchPlaceholder = tableBody.querySelector('.search-placeholder');
    if (oldSearchPlaceholder) oldSearchPlaceholder.remove();

    if (visibleRows.length === 0 && rows.length > 0) {
        const placeholderTr = document.createElement('tr');
        placeholderTr.className = 'search-placeholder';
        placeholderTr.innerHTML = `
            <td colspan="8">
                <div class="table-placeholder-content">
                    <i data-lucide="search-code"></i>
                    <p>No matching evaluations found for search criteria.</p>
                </div>
            </td>
        `;
        tableBody.appendChild(placeholderTr);
        lucide.createIcons();
    }
}
