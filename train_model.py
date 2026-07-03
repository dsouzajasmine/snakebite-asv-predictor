import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# Load cleaned data
df = pd.read_csv('snakebite_cleaned.csv')

# Create severity categories
bins   = [-1, 0, 15, 30, 100]
labels = ['None', 'Low', 'Moderate', 'High']
df['ASV_Severity'] = pd.cut(df['No of ASV vials'], bins=bins, labels=labels)

# Features
features = [
    'Gender(1,2)', 'Age (Cat)', 'Co morbidity',
    'Morning, night', 'Time taken to reach 3',
    'Site - 1-5', 'Activity during bite', 'Torniquet',
    'Local S/S', 'Systemic s/s', '20WBCT', 'Snake identified',
    'Treatment received before reaching teriary center',
    'Hb', 'Platelet', 'PT', 'INR', 'aPTT',
    'Urea', 'Creat', 'SGOT', 'SGPT', 'CRP'
]

X = df[features].copy()
y = df['ASV_Severity'].copy()

# Handle missing values
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

category_cols = X.select_dtypes(include=['object']).columns
for col in category_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode
X_encoded = pd.get_dummies(X, drop_first=True)

# Encode target
severity_order = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
y_encoded = y.map(severity_order)

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

model = XGBClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, random_state=42,
    eval_metric='mlogloss',
    scale_pos_weight=3
)
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'asv_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_features.pkl')

print("✅ Model retrained and saved successfully!")