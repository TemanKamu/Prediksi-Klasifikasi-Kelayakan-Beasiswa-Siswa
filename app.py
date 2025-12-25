# pip install streamlit pandas scikit-learn
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Konfigurasi Halaman
st.set_page_config(page_title="Scholarship Dashboard", layout="wide")

# --- 1. LOAD DATA & MODEL TRAINING ---
@st.cache_resource
def train_model():
    df = pd.read_csv('Students Performance .csv')
    target_col = 'Scholarship'
    drop_cols = ['Student_ID', target_col]
    
    df_features = df.drop(columns=drop_cols)
    feature_names = df_features.columns.tolist()
    
    encoders = {}
    X = df_features.copy()
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[target_col].astype(str))
    encoders[target_col] = target_le
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders, feature_names

model, encoders, feature_names = train_model()

# --- 2. SIDEBAR FILTER ---
st.sidebar.header("⚙️ Filter Data")
st.sidebar.write("Gunakan filter ini setelah mengunggah file untuk membedah hasil.")

# --- 3. ANTARMUKA UTAMA ---
st.title("📊 Dashboard Analisis Beasiswa")
uploaded_file = st.file_uploader("Unggah file CSV (Dataset Siswa)", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    # Tombol Prediksi
    if st.sidebar.button('🚀 Jalankan Analisis'):
        try:
            # PROSES PREDIKSI
            X_input = input_df[feature_names].copy()
            for col in X_input.columns:
                if col in encoders and col != 'Scholarship':
                    le = encoders[col]
                    X_input[col] = X_input[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in le.classes_ else -1
                    )
            
            predictions = model.predict(X_input)
            input_df['PREDIKSI_BEASISWA'] = encoders['Scholarship'].inverse_transform(predictions)

            # --- BAGIAN 1: SUMMARY METRICS ---
            st.markdown("### 📌 Ringkasan Cepat")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Siswa", len(input_df))
            c2.metric("Beasiswa 100%", len(input_df[input_df['PREDIKSI_BEASISWA'] == '100%']))
            c3.metric("Beasiswa 75%", len(input_df[input_df['PREDIKSI_BEASISWA'] == '75%']))
            c4.metric("Tidak Dapat (None)", len(input_df[input_df['PREDIKSI_BEASISWA'] == 'None']))

            st.divider()

            # --- BAGIAN 2: VISUALISASI YANG LEBIH MUDAH ---
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🏆 Peringkat Penerima Beasiswa")
                # Menggunakan Bar Chart Horizontal (Lebih mudah dibaca daripada Pie)
                count_df = input_df['PREDIKSI_BEASISWA'].value_counts().reset_index()
                count_df.columns = ['Kategori', 'Jumlah Siswa']
                fig_bar = px.bar(count_df, x='Jumlah Siswa', y='Kategori', orientation='h',
                                 color='Kategori', text='Jumlah Siswa',
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption("Grafik ini menunjukkan kategori beasiswa mana yang paling banyak diberikan.")

            with col_right:
                st.subheader("🎯 Korelasi Grade vs Beasiswa")
                # Menggunakan Treemap untuk melihat komposisi
                fig_tree = px.treemap(input_df, path=['PREDIKSI_BEASISWA', 'Grade'], 
                                     color='PREDIKSI_BEASISWA',
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_tree, use_container_width=True)
                st.caption("Klik pada kotak beasiswa untuk melihat detail persebaran Grade di dalamnya.")

            st.divider()
            
            # --- BAGIAN 3: ANALISIS KEBIASAAN ---
            st.subheader("💡 Mengapa Mereka Mendapat Beasiswa?")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Rata-rata Jam Belajar per Minggu**")
                avg_study = input_df.groupby('PREDIKSI_BEASISWA')['Weekly_Study_Hours'].mean().reset_index()
                fig_line = px.line(avg_study, x='PREDIKSI_BEASISWA', y='Weekly_Study_Hours', markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
            
            with col_b:
                st.write("**Pengaruh Kehadiran di Kelas**")
                fig_attend = px.density_heatmap(input_df, x="Attendance", y="PREDIKSI_BEASISWA", 
                                                color_continuous_scale="Viridis")
                st.plotly_chart(fig_attend, use_container_width=True)

            # --- DATA TABLE ---
            with st.expander("Klik untuk melihat detail tabel data"):
                st.dataframe(input_df, use_container_width=True)
                csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Simpan Hasil ke Excel/CSV", csv, "hasil_analisis.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Selamat Datang! Silakan unggah file CSV di atas untuk memulai analisis.")