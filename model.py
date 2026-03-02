import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model():
    print("⏳ Memuat dataset 'cleaned_energy_data.csv'...")
    try:
        df = pd.read_csv("cleaned_energy_data.csv")
    except FileNotFoundError:
        print("❌ ERROR: File 'cleaned_energy_data.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return

    # Definisi Target dan Fitur
    target = 'CO2_per_capita_tons'
    features = [
        'gdp_per_capita', 'Access to electricity (% of population)',
        'Renewable energy share in the total final energy consumption (%)',
        'Primary energy consumption per capita (kWh/person)',
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
        'Low-carbon electricity (% electricity)', 'Years_Since_2000',
        'Prev_Year_Renewable_Share', '3_Year_Avg_GDP_Growth'
    ]

    # Chronological Train Split (Sesuai to-do list: Train 2000-2016)
    print("⚙️ Memisahkan data training (2000-2016) dan melatih model Random Forest...")
    train_data = df[df['Year'] <= 2016].dropna()
    
    X_train = train_data[features]
    y_train = train_data[target]

    # Inisialisasi dan Training Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Export Model menggunakan joblib
    print("💾 Menyimpan model ke 'rf_model_energy.pkl'...")
    joblib.dump(rf_model, 'rf_model_energy.pkl')
    
    print("✅ SELESAI! Model berhasil disimpan.")
    print("🚀 Langkah selanjutnya: Jalankan perintah 'streamlit run app.py' di terminal.")

if __name__ == "__main__":
    train_and_save_model()