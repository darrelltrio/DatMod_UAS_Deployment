import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (UI/UX)
# ==========================================
st.set_page_config(page_title="EcoSim: CO2 Predictor", page_icon="🌍", layout="wide")

# Custom CSS agar metrik dan font terlihat modern
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #2e7b32;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e1e1e;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL (Menggunakan Cache)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('rf_model_energy.pkl')

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_energy_data.csv")

try:
    rf_model = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Model atau Data tidak ditemukan! Pastikan Anda sudah menjalankan 'python model.py' terlebih dahulu.")
    st.stop()

# ==========================================
# 3. HEADER & BANNER IMAGE
# ==========================================
st.image("https://images.unsplash.com/photo-1466611653911-95081537e5b7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
st.title("🌍 EcoSim: What-If CO2 Emissions Simulator")
st.markdown("*Interactive Scenario Analysis for Global Energy Transition (2020 ➔ 2030)*")
st.divider()

# ==========================================
# 4. SIDEBAR (USER INTERFACE PANEL)
# ==========================================
st.sidebar.header("🎛️ Scenario Control Panel")

# Fitur Dropdown dan Slider
countries_2020 = df[df['Year'] == 2020]['Entity'].unique()
selected_country = st.sidebar.selectbox("Pilih Negara:", sorted(countries_2020), index=int(sorted(countries_2020).index('Indonesia') if 'Indonesia' in countries_2020 else 0))

st.sidebar.markdown("### 📈 Asumsi Ekonomi (Hingga 2030)")
gdp_growth = st.sidebar.slider("Pertumbuhan GDP Tahunan (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
energy_growth = st.sidebar.slider("Lonjakan Konsumsi Energi Tahunan (%)", min_value=1.0, max_value=8.0, value=2.0, step=0.5)

st.sidebar.markdown("### 🍃 Target Transisi Energi (2030)")
target_renewable = st.sidebar.slider("Target Energi Terbarukan (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
target_lowcarbon = st.sidebar.slider("Target Listrik Rendah Karbon (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

# ==========================================
# 5. BACKEND PREDICTION LOGIC
# ==========================================
features = [
    'gdp_per_capita', 'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'Primary energy consumption per capita (kWh/person)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    'Low-carbon electricity (% electricity)', 'Years_Since_2000',
    'Prev_Year_Renewable_Share', '3_Year_Avg_GDP_Growth'
]

# Baseline 2020
base_data = df[(df['Entity'] == selected_country) & (df['Year'] == 2020)].iloc[0]
actual_co2_2020 = base_data['CO2_per_capita_tons']

# Skenario Masa Depan 2030
years_ahead = 10
future_features = base_data[features].copy()
future_features['Years_Since_2000'] = 30
future_features['Access to electricity (% of population)'] = 100.0

# Aplikasikan input dari slider ke variabel
future_features['gdp_per_capita'] *= ((1 + (gdp_growth/100)) ** years_ahead)
future_features['3_Year_Avg_GDP_Growth'] = gdp_growth
future_features['Primary energy consumption per capita (kWh/person)'] *= ((1 + (energy_growth/100)) ** years_ahead)

future_features['Renewable energy share in the total final energy consumption (%)'] = target_renewable
future_features['Prev_Year_Renewable_Share'] = target_renewable - 2.0
future_features['Low-carbon electricity (% electricity)'] = target_lowcarbon

# Eksekusi Prediksi
pred_co2_2030 = rf_model.predict(future_features.values.reshape(1, -1))[0]
delta_pct = ((pred_co2_2030 - actual_co2_2020) / actual_co2_2020) * 100

# ==========================================
# 6. MAIN DASHBOARD DISPLAY
# ==========================================
st.markdown(f"### Analisis Skenario Tahun 2030: **{selected_country}**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid #6c757d;">
            <div class="metric-label">Actual CO2 (2020)</div>
            <div class="metric-value">{actual_co2_2020:.2f} <span style="font-size:1rem;">Tons</span></div>
            <div style="color: #6c757d; font-size: 0.9rem; margin-top:10px;">Baseline</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    color = "#d32f2f" if delta_pct > 0 else "#2e7b32"
    trend = "🔺 Naik" if delta_pct > 0 else "📉 Turun"
    
    st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid {color};">
            <div class="metric-label">Predicted CO2 (2030)</div>
            <div class="metric-value" style="color: {color};">{pred_co2_2030:.2f} <span style="font-size:1rem; color:#1e1e1e;">Tons</span></div>
            <div style="color: {color}; font-size: 0.9rem; font-weight:bold; margin-top:10px;">{trend} {abs(delta_pct):.1f}% dari 2020</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid #1976d2;">
            <div class="metric-label">Target Energi Hijau</div>
            <div class="metric-value">{target_renewable:.0f} <span style="font-size:1rem;">%</span></div>
            <div style="color: #6c757d; font-size: 0.9rem; margin-top:10px;">Skenario Pilihan User</div>
        </div>
    """, unsafe_allow_html=True)

st.divider()

st.info(f"""
**💡 Insight Model:** Jika **{selected_country}** mencetak pertumbuhan ekonomi rata-rata {gdp_growth}% dan konsumsi energi melonjak {energy_growth}% per tahun hingga 2030, emisi CO2 diproyeksikan berada di angka **{pred_co2_2030:.2f} Ton per kapita**. 
Gunakan *slider* di sebelah kiri untuk menyimulasikan intervensi kebijakan transisi energi dan pantau perubahan emisi secara *real-time*!
""")

# ==========================================
# 7. TEAM & CREDITS PANEL
# ==========================================
st.divider()
st.markdown("### 👨‍💻 Meet Our Team - Group [Nomor/Nama Grup]")
st.markdown("Proyek ini disusun untuk memenuhi tugas visualisasi dan pemodelan data energi global.")

# Membuat 4 kolom agar sejajar (sesuaikan jumlahnya jika anggota grup kurang/lebih)
team1, team2, team3, team4 = st.columns(4)

with team1:
    # Menggunakan UI Avatars sebagai placeholder dinamis
    st.image("https://ui-avatars.com/api/?name=Darrell&background=2e7b32&color=fff&size=150", width=150)
    st.markdown("**Darrell**")
    st.caption("Machine Learning & Deployment Lead")

with team2:
    st.image("https://ui-avatars.com/api/?name=Anggota+2&background=1976d2&color=fff&size=150", width=150)
    st.markdown("**[Nama Anggota 2]**")
    st.caption("Data Prep & Cleaning Lead")

with team3:
    st.image("https://ui-avatars.com/api/?name=Anggota+3&background=d32f2f&color=fff&size=150", width=150)
    st.markdown("**[Nama Anggota 3]**")
    st.caption("Data Visualization Lead")
    
with team4:
    st.image("https://ui-avatars.com/api/?name=Anggota+4&background=ffb300&color=fff&size=150", width=150)
    st.markdown("**[Nama Anggota 4]**")
    st.caption("Research & Documentation")

st.markdown("<br><center><p style='color: #6c757d; font-size: 0.8rem;'>© 2026 EcoSim Project. All rights reserved.</p></center>", unsafe_allow_html=True)