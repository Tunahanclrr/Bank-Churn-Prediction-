import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Veri yükle
df = pd.read_csv("Churn_Modelling.csv")

# Gereksiz sütunları sil
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# X ve y ayır
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Preprocessing
geography_col = ['Geography']
gender_col = ['Gender']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

preprocessor = ColumnTransformer([
    ('geo_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), geography_col),
    ('gender_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), gender_col),
    ('num_scaler', StandardScaler(), numerical_cols)
], remainder='drop')

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Model eğit
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
model.fit(X_train_transformed, y_train)

# Model kaydet
joblib.dump(model, "churn_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("✅ Model başarıyla eğitildi ve kaydedildi!")
print(f"Model doğruluğu: {model.score(X_test_transformed, y_test):.4f}")
