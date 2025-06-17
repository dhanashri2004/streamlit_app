# ml_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Dataset
df = pd.read_csv("incident_event_log.csv")

# 2. Drop null or irrelevant columns
df.dropna(inplace=True)
df.drop(columns=['number', 'sys_id'], errors='ignore', inplace=True)  # Drop ID columns if exist

# 3. Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 4. Define features and target
X = df.drop("incident_state", axis=1)
y = df["incident_state"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save the model and encoders
joblib.dump(model, "incident_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
