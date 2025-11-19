import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Bank Churn Prediction | Banka Müşteri Churn Tahmini", page_icon="💳", layout="wide")

# Başlık
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💳 Banka Müşteri Churn Tahmini")
with col2:
    st.metric("Model", "Random Forest", "87.08%")

st.markdown("""
---
### Müşteri bilgilerinizi girin ve bankadan ayrılma riskinizi öğrenin.
*Enter customer information and learn the churn risk.*
""")

# --- Sidebar ---
st.sidebar.header("👤 Müşteri Bilgileri | Customer Info")

col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Yaş | Age", 18, 100, 35)
    tenure = st.slider("Kalış Süresi | Tenure (Year)", 0, 10, 3)
    num_products = st.slider("Ürün Sayısı | Products", 1, 4, 1)

with col2:
    geography = st.selectbox("Ülke | Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Cinsiyet | Gender", ["Male", "Female"])
    is_active = st.selectbox("Aktif Üye? | Active?", ["Yes", "No"])

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Finansal Bilgi | Financial Info")

col1, col2 = st.sidebar.columns(2)
with col1:
    credit_score = st.number_input("Kredi Skoru | Credit Score", 350, 850, 600)
    balance = st.number_input("Bakiye | Balance", 0.0, 250000.0, 5000.0)

with col2:
    est_salary = st.number_input("Maaş | Salary", 0.0, 200000.0, 35000.0)
    has_cr_card = st.selectbox("Kredi Kartı? | Credit Card?", ["Yes", "No"])

st.sidebar.markdown("---")

# --- Data Preparation ---
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0

data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card_val],
    "IsActiveMember": [is_active_val],
    "EstimatedSalary": [est_salary]
})

# --- Model Loading ---
import os
try:
    # Try multiple paths for pickle files
    base_path = os.path.dirname(__file__)
    preprocessor_path = os.path.join(base_path, "preprocessor.pkl")
    model_path = os.path.join(base_path, "churn_model.pkl")
    
    # If not found, try current directory
    if not os.path.exists(preprocessor_path):
        preprocessor_path = "preprocessor.pkl"
    if not os.path.exists(model_path):
        model_path = "churn_model.pkl"
    
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    model_loaded = True
except Exception as e:
    st.error(f"Model yükleme hatası: {str(e)}")
    model_loaded = False

# --- Prediction ---
if model_loaded:
    X_transformed = preprocessor.transform(data)
    churn_prob = model.predict_proba(X_transformed)[0][1]
    churn_pred = model.predict(X_transformed)[0]
else:
    churn_prob = None
    churn_pred = None

# --- Main Content ---
st.markdown("---")

if model_loaded and churn_prob is not None:
    # Risk göstergesi
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### 📊 Churn Riski")
        
        # Risk seviyesi
        if churn_prob < 0.2:
            risk_level = "🟢 ÇOK DÜŞÜK"
            risk_color = "green"
        elif churn_prob < 0.4:
            risk_level = "🟡 DÜŞÜK"
            risk_color = "yellow"
        elif churn_prob < 0.6:
            risk_level = "🟠 ORTA"
            risk_color = "orange"
        elif churn_prob < 0.8:
            risk_level = "🔴 YÜKSEK"
            risk_color = "red"
        else:
            risk_level = "⛔ ÇOK YÜKSEK"
            risk_color = "darkred"
        
        st.markdown(f"## {risk_level}")
        st.metric("Terk Etme Olasılığı", f"{churn_prob*100:.1f}%")
    
    with col2:
        # Gauge chart simülasyonu
        st.markdown("### Tahmin Sonucu")
        if churn_pred == 1:
            st.error(f"""
            ⚠️ **YÜKSEK RISK - Müşteri Bankadan Ayrılabilir**
            
            Bu müşterinin bankadan ayrılma olasılığı **{churn_prob*100:.1f}%** dir.
            
            **Önerilen Aksiyonlar:**
            - Müşteri ile iletişime geç
            - Özel indirim/promosyon sununu
            - Hizmet kalitesini artır
            - İlişki yöneticisi ata
            """)
        else:
            st.success(f"""
            ✅ **DÜŞÜK RISK - Müşteri Muhtemelen Kalacak**
            
            Bu müşterinin bankada kalma olasılığı **{(1-churn_prob)*100:.1f}%** dir.
            
            **Önerilen Aksiyonlar:**
            - Mevcut hizmeti devam et
            - Yeni ürün/hizmet sunularını değerlendir
            - Müşteri memnuniyetini kontrol et
            """)
    
    with col3:
        st.markdown("### 📈 Risk Faktörleri")
        
        risk_factors = []
        if age > 40:
            risk_factors.append("👴 Yaş (40+)")
        if balance == 0:
            risk_factors.append("💰 Sıfır Bakiye")
        if geography == "Germany":
            risk_factors.append("🇩🇪 Almanya")
        if num_products < 2:
            risk_factors.append("📦 Az Ürün")
        if is_active_val == 0:
            risk_factors.append("😴 Pasif Üye")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.info("✅ Belirgin risk faktörü yok")

else:
    st.error("❌ Model dosyaları bulunamadı! Lütfen train_model.py dosyasını çalıştırın.")
    st.info("""
    Modeli eğitmek için terminalde şunu çalıştırın:
    ```
    python train_model.py
    ```
    """)

st.markdown("---")
st.caption("© 2025 Bank Churn Prediction | Random Forest Model | 87.08% Accuracy")
