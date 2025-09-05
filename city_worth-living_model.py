import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("city_data.csv")

print(df.head())

df.drop_duplicates(inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)   
    else:
        df[col].fillna(df[col].mean(), inplace=True)

X = df[["AQI", "CrimeRate", "HealthcareIndex", "EducationIndex", "JobOpportunities"]] 
y = df["WorthLiving"]

label = LabelEncoder()
print(label)
y = label.fit_transform(y) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print("\n Model Accuracy:", accuracy_score(y_test, prediction))
print("\n Classification Report:")
print(classification_report(y_test, prediction, target_names=label.classes_))

def new_city_data():
    print("\n🌆 Enter new city parameters:")
    AQI = float(input("Enter AQI: "))
    CrimeRate = float(input("Enter Crime Rate: "))
    HealthcareIndex = float(input("Enter Healthcare Index: "))
    EducationIndex = float(input("Enter Education Index : "))
    JobOpportunities = float(input("Enter Job Opportunities Score: "))

    new_data = [[AQI, CrimeRate, HealthcareIndex, EducationIndex, JobOpportunities]]
    new_data_scaled = scaler.transform(new_data)

    pred = knn.predict(new_data_scaled)[0]
    result = label.inverse_transform([pred])[0]

    print(f"\n Prediction: This city is '{result}' worth living.")
