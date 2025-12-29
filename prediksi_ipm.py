import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===== INIT SESSION STATE =====
if "year" not in st.session_state:
    st.session_state.year = 2025
    st.session_state.predictions = []

# ===== LOAD DATA =====
df_ipm = pd.read_csv("ipm_2012_2024.csv")  # kolom: tahun, ipm

# ===== HEADER =====
st.header("Prediksi Indeks Pembangunan Manusia (IPM)")

# ===== PETUNJUK =====
with st.expander("📘 Petunjuk Penggunaan"):
    st.markdown("""
**Gunakan HLS dan UHH jika:**
- tujuan utama adalah memprediksi nilai IPM secara numerik seakurat mungkin
- model tidak digunakan untuk analisis kausal atau perumusan kebijakan

**Jangan gunakan HLS dan UHH jika:**
- tujuan analisis adalah memahami pengaruh kemiskinan dan ketimpangan
- hasil digunakan untuk dasar perumusan atau evaluasi kebijakan

⚠️ Gunakan koma sebagai desimal, dan tidak perlu ada pemisah ribuan
                
⚠️ Aplikasi ini melakukan **simulasi berbasis input pengguna**,  
bukan proyeksi otomatis berbasis time series.
""")

# ===== INPUT FEATURES =====
poverty = st.number_input(
    "Garis Kemiskinan (rupiah/bulan/kapita)",
    min_value=0.0
)
poor_pop = st.number_input(
    "Jumlah Penduduk Miskin (ribu)",
    min_value=0.0
)
gini = st.number_input(
    "Gini Rasio",
    min_value=0.0,
    max_value=1.0
)

hls = st.number_input(
    "Harapan Lama Sekolah/HLS (tahun) (opsional)",
    value=None,
    format="%.2f"
)
uhh = st.number_input(
    "Umur Harapan Hidup Saat Lahir/UHH (tahun) (opsional)",
    value=None,
    format="%.2f"
)

# ===== PREDICT BUTTON =====
if st.button("Tambah Tahun Prediksi"):
    use_full_model = hls is not None and uhh is not None

    if use_full_model:
        model = joblib.load("model1.joblib")
        X = pd.DataFrame([{
            "garis kemiskinan (rupiah/bulan/kapita)": poverty,
            "jumlah penduduk miskin (ribu)": poor_pop,
            "gini rasio": gini,
            "harapan lama sekolah (tahun)": hls,
            "Umur Harapan Hidup Saat Lahir (UHH) (tahun)": uhh
        }])
        mode = "full"
    else:
        model = joblib.load("model2.joblib")
        X = pd.DataFrame([{
            "garis kemiskinan (rupiah/bulan/kapita)": poverty,
            "jumlah penduduk miskin (ribu)": poor_pop,
            "gini rasio": gini
        }])
        mode = "structural"

    pred_ipm = model.predict(X)[0]

    st.session_state.predictions.append({
        "year": st.session_state.year,
        "ipm": pred_ipm,
        "mode": mode
    })

    st.success(
        f"Prediksi IPM tahun {st.session_state.year}: {pred_ipm:.2f}"
    )

    st.session_state.year += 1

# ===== RESET BUTTON (UX FIX) =====
if st.button(
    "Reset Simulasi",
    disabled=not st.session_state.predictions
):
    st.session_state.year = 2025
    st.session_state.predictions = []
    st.toast("Simulasi di-reset")
    st.rerun()

# ===== PLOT =====
fig, ax = plt.subplots()

# ---- IPM Historis ----
ax.plot(
    df_ipm["tahun"],
    df_ipm["ipm"],
    color="blue",
    marker="o",
    label="IPM Historis (2012–2024)"
)

# titik terakhir historis
timeline = [{
    "year": df_ipm["tahun"].iloc[-1],
    "ipm": df_ipm["ipm"].iloc[-1],
    "mode": "historical"
}]

# gabung prediksi ke timeline
if st.session_state.predictions:
    timeline += sorted(
        st.session_state.predictions,
        key=lambda x: x["year"]
    )

# ---- Plot segmen per tahun ----
for i in range(len(timeline) - 1):
    y1, y2 = timeline[i], timeline[i + 1]

    if y2["mode"] == "full":
        color = "green"
    elif y2["mode"] == "structural":
        color = "red"
    else:
        continue

    ax.plot(
        [y1["year"], y2["year"]],
        [y1["ipm"], y2["ipm"]],
        color=color,
        marker="o"
    )

# ---- Legend manual ----
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="blue", marker="o", label="IPM Historis (2012–2024)"),
    Line2D([0], [0], color="green", marker="o", label="Prediksi (dengan HLS & UHH)"),
    Line2D([0], [0], color="red", marker="o", label="Prediksi (tanpa HLS & UHH)")
]
ax.legend(handles=legend_elements)

ax.set_xlabel("Tahun")
ax.set_ylabel("IPM")
ax.set_title("Simulasi IPM Historis dan Prediksi")
ax.grid(True)

st.pyplot(fig)
