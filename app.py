import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (UI/UX)
# ==========================================
st.set_page_config(page_title="EcoSim: CO2 Predictor", page_icon="🌍", layout="wide")

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
# 2. LOAD DATA & MODEL (Dengan Cache)
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
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📌 Navigasi Panel")
menu = st.sidebar.radio("Pilih Halaman:", ["📊 Simulator (Deployment)", "👥 Meet the Team"])
st.sidebar.divider()

# ==========================================
# PANEL 1: SIMULATOR DEPLOYMENT
# ==========================================
if menu == "📊 Simulator (Deployment)":
    
    st.image("https://images.unsplash.com/photo-1466611653911-95081537e5b7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_column_width=True)
    st.title("🌍 EcoSim: What-If CO2 Emissions Simulator")
    st.markdown("*Interactive Scenario Analysis for Global Energy Transition (2020 ➔ 2030)*")
    st.divider()

    st.sidebar.header("🎛️ Scenario Control Panel")

    countries_2020 = df[df['Year'] == 2020]['Entity'].unique()
    selected_country = st.sidebar.selectbox("Pilih Negara:", sorted(countries_2020), index=int(sorted(countries_2020).index('Indonesia') if 'Indonesia' in countries_2020 else 0))

    st.sidebar.markdown("### 📈 Asumsi Ekonomi & Efisiensi")
    gdp_growth = st.sidebar.slider("Pertumbuhan GDP Tahunan (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    energy_growth = st.sidebar.slider("Lonjakan Konsumsi Energi Tahunan (%)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
    
    # SLIDER BARU: Efisiensi Energi (Menurunkan Energy Intensity)
    efficiency_target = st.sidebar.slider("Peningkatan Efisiensi Energi (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0, help="Persentase penurunan intensitas/keborosan energi industri dan masyarakat pada tahun 2030.")

    st.sidebar.markdown("### 🍃 Target Transisi Energi (2030)")
    target_renewable = st.sidebar.slider("Target Energi Terbarukan (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    target_lowcarbon = st.sidebar.slider("Target Listrik Rendah Karbon (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

    # Backend Logic Model
    features = [
        'gdp_per_capita', 'Access to electricity (% of population)',
        'Renewable energy share in the total final energy consumption (%)',
        'Primary energy consumption per capita (kWh/person)',
        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
        'Low-carbon electricity (% electricity)', 'Years_Since_2000',
        'Prev_Year_Renewable_Share', '3_Year_Avg_GDP_Growth'
    ]

    base_data = df[(df['Entity'] == selected_country) & (df['Year'] == 2020)].iloc[0]
    actual_co2_2020 = base_data['CO2_per_capita_tons']
    base_intensity = base_data['Energy intensity level of primary energy (MJ/$2017 PPP GDP)']

    years_ahead = 10
    future_features = base_data[features].copy()
    future_features['Years_Since_2000'] = 30
    future_features['Access to electricity (% of population)'] = 100.0

    # Aplikasi Skala Ekonomi & Efisiensi
    future_features['gdp_per_capita'] *= ((1 + (gdp_growth/100)) ** years_ahead)
    future_features['3_Year_Avg_GDP_Growth'] = gdp_growth
    future_features['Primary energy consumption per capita (kWh/person)'] *= ((1 + (energy_growth/100)) ** years_ahead)
    
    # Aplikasi LOGIKA BARU: Semakin tinggi efisiensi energi (%), semakin kecil angka Energy Intensity-nya
    future_features['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'] = base_intensity * (1 - (efficiency_target / 100))

    # Aplikasi Transisi Energi
    future_features['Renewable energy share in the total final energy consumption (%)'] = target_renewable
    future_features['Prev_Year_Renewable_Share'] = target_renewable - 2.0
    future_features['Low-carbon electricity (% electricity)'] = target_lowcarbon

    pred_co2_2030 = rf_model.predict(future_features.values.reshape(1, -1))[0]
    delta_pct = ((pred_co2_2030 - actual_co2_2020) / actual_co2_2020) * 100

    # Display Metrik
    st.markdown(f"### Analisis Skenario Tahun 2030: **{selected_country}**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card" style="border-top: 5px solid #6c757d;">
                <div class="metric-label">Actual CO2 (2020)</div>
                <div class="metric-value">{actual_co2_2020:.2f} <span style="font-size:1rem;">Tons</span></div>
                <div style="color: #6c757d; font-size: 0.9rem; margin-top:10px;">Baseline per Kapita</div>
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
                <div class="metric-label">Efisiensi Energi (Target)</div>
                <div class="metric-value">{efficiency_target:.0f} <span style="font-size:1rem;">%</span></div>
                <div style="color: #6c757d; font-size: 0.9rem; margin-top:10px;">Penghematan Intensitas Energi</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.info(f"**💡 Insight Model:** Jika **{selected_country}** mencetak pertumbuhan ekonomi {gdp_growth}% dan konsumsi energi melonjak {energy_growth}% tiap tahunnya hingga 2030, NAMUN dibarengi dengan efisiensi energi industri sebesar {efficiency_target}%, emisi CO2 diproyeksikan berada di angka **{pred_co2_2030:.2f} Ton per kapita**.")

# ==========================================
# PANEL 2: MEET THE TEAM
# ==========================================
elif menu == "👥 Meet the Team":
    st.title("👨‍💻 Meet Our Team")
    st.markdown("Proyek **Visualisasi dan Pemodelan Data Energi Global** ini disusun oleh Grup [Nomor/Nama Grup].")
    st.divider()
    
    team1, team2, team3, team4 = st.columns(4)

    with team1:
        st.image("https://ui-avatars.com/api/?name=Darrell&background=2e7b32&color=fff&size=200", use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>Darrell</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d;'>Machine Learning & Deployment Lead</p>", unsafe_allow_html=True)

    with team2:
        st.image("https://ui-avatars.com/api/?name=Anggota+2&background=1976d2&color=fff&size=200", use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>[Nama Anggota 2]</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d;'>Data Prep & Cleaning Lead</p>", unsafe_allow_html=True)

    with team3:
        st.image("https://ui-avatars.com/api/?name=Anggota+3&background=d32f2f&color=fff&size=200", use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>[Nama Anggota 3]</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d;'>Data Visualization Lead</p>", unsafe_allow_html=True)
        
    with team4:
        st.image("https://ui-avatars.com/api/?name=Anggota+4&background=ffb300&color=fff&size=200", use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>[Nama Anggota 4]</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d;'>Research & Documentation</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<center><p style='color: #6c757d;'>© 2026 EcoSim Project - Universitas [Nama Kampus]. All rights reserved.</p></center>", unsafe_allow_html=True)