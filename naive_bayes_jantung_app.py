
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Prediksi Penyakit Jantung dengan Naive Bayes", layout="wide")

# Mapping untuk Terjemahan Nama Fitur (dari Inggris ke Indonesia)
FEATURE_MAP = {
    'age': 'Usia (tahun)',
    'sex': 'Jenis Kelamin (0=P/1=L)',
    'cp': 'Jenis Nyeri Dada (0-3)',
    'trestbps': 'Tekanan Darah Istirahat',
    'chol': 'Kolesterol Serum',
    'fbs': 'Gula Darah Puasa (>120)',
    'restecg': 'Hasil EKG Istirahat (0-2)',
    'thalach': 'Denyut Jantung Maksimum',
    'exang': 'Angina Akibat Olahraga (1=Ya)',
    'oldpeak': 'Depresi ST Saat Olahraga',
    'slope': 'Slope Segmen ST Puncak (0-2)',
    'ca': 'Pembuluh Darah Utama (0-3)',
    'thal': 'Hasil Tes Thallium (1-3)',
    'condition': 'Kondisi (Target)'  # Hanya untuk referensi
}

# Konstanta
CLASS_NAMES = ['0 (Tidak Sakit)', '1 (Sakit)']

# Session keys yang digunakan
SESSION_KEYS = [
    'model', 'feature_names', 'target_classes', 'uploaded_file',
    'accuracy', 'input_data', 'prediction', 'proba', 'page'
]

# --- Fungsi untuk Melatih Model (dengan caching) ---
@st.cache_resource
def get_model_info_from_file(file_like):
    """Melatih GaussianNB dari file CSV dan mengembalikan model, fitur, kelas, akurasi.
       Mengembalikan (None, None, None, None) jika gagal.
    """
    try:
        file_like.seek(0)
        df = pd.read_csv(file_like)
        if df.shape[1] < 2:
            return None, None, None, None

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Pastikan X numeric; jika ada kolom non-numeric, coba konversi
        X = X.apply(pd.to_numeric, errors='coerce')
        if X.isnull().any().any():
            # Jika masih ada NaN setelah coercion, gagal
            return None, None, None, None

        feature_names = X.columns.tolist()
        target_classes = sorted(pd.Series(y).unique())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, feature_names, target_classes, accuracy
    except Exception:
        return None, None, None, None

# --- Halaman: Home ---
def home_page():
    st.title('🏠 Selamat Datang di Aplikasi Prediksi Penyakit Jantung')
    st.markdown("---")
    st.header("Tujuan Aplikasi")
    st.markdown("""
        Aplikasi ini memprediksi risiko Penyakit Jantung berdasarkan fitur klinis
        menggunakan Gaussian Naive Bayes. Unggah dataset CSV dengan target di kolom terakhir.
    """)
    st.header("Cara Menggunakan")
    st.markdown("""
        1. Unggah file CSV (kolom target di kolom terakhir).  
        2. Buka halaman "Fitur Data" untuk melihat fitur.  
        3. Buka "Input & Algoritma" untuk memasukkan nilai pasien dan melakukan prediksi.  
        4. Lihat hasil di "Hasil Diagnosis".
    """)
    st.info("💡 Contoh dataset: kolom fitur numerik, kolom terakhir berisi label 0/1.")

# --- Halaman: Detail Fitur ---
def feature_page(feature_names):
    st.title('📊 Detail Fitur Data')
    st.markdown("---")
    if not feature_names:
        st.warning("Tidak ada fitur yang terdeteksi. Silakan unggah file CSV yang valid.")
        return

    st.header("Daftar Fitur yang Digunakan untuk Prediksi")
    translated = [FEATURE_MAP.get(f, f) for f in feature_names]
    df_feat = pd.DataFrame({
        "Nama Fitur (Original)": feature_names,
        "Nama Fitur (Terjemahan)": translated
    })
    st.dataframe(df_feat, use_container_width=True)

# --- Halaman: Input & Detail Algoritma ---
def input_and_model_detail_page(model, feature_names, target_classes):
    st.title('🔬 Input Pasien & Detail Algoritma')
    st.markdown("---")

    if model is None or not feature_names:
        st.warning("Model belum tersedia. Silakan unggah dataset yang valid terlebih dahulu.")
        return

    st.header("🎯 Input Fitur Pasien Baru")

    # Pastikan session input_data valid
    input_data = st.session_state.get('input_data')
    if input_data is None or not isinstance(input_data, np.ndarray) or input_data.size == 0:
        st.session_state.input_data = np.array([[0.0] * len(feature_names)])
        input_data = st.session_state.input_data

    # Ambil default means (jika tersedia), fallback ke 0.0
    try:
        # model.theta_ shape: (n_classes, n_features)
        default_means = pd.Series(model.theta_[0, :], index=feature_names)
    except Exception:
        default_means = pd.Series([0.0]*len(feature_names), index=feature_names)

    input_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        with col:
            # Aman ambil current_input
            try:
                current_input = float(input_data[0, i])
            except Exception:
                current_input = 0.0

            default_val = current_input if current_input != 0.0 else float(default_means.get(feature, 0.0))

            display_name = FEATURE_MAP.get(feature, feature)
            # Gunakan key terpisah per fitur agar Streamlit tidak bingung
            widget_key = f"input_{feature}"
            # Pastikan step dan min/max bisa diubah jika user ingin
            val = st.number_input(
                f"**{display_name}**",
                value=float(default_val),
                step=0.1,
                key=widget_key
            )
            input_values[feature] = float(val)

    # Tombol Prediksi
    if st.button('Hitung Prediksi & Pindah ke Hasil Diagnosis'):
        try:
            arr = np.array([list(input_values.values())], dtype=float)
            # Simpan di session
            st.session_state.input_data = arr
            # Prediksi
            prediction = model.predict(arr)[0]
            proba = None
            try:
                proba = model.predict_proba(arr)[0]
            except Exception:
                # Jika predict_proba tidak tersedia, buat proba dari class_count_
                try:
                    counts = model.class_count_
                    proba = counts / counts.sum()
                except Exception:
                    proba = np.array([0.0, 0.0])

            st.session_state.prediction = int(prediction)
            st.session_state.proba = proba
            st.session_state.page = "prediction"
            st.rerun()
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

    st.markdown("---")
    st.header("Detail Algoritma Gaussian Naive Bayes")
    tab_prior, tab_likelihood = st.tabs(["Probabilitas Awal", "Parameter Likelihood"])
    with tab_prior:
        st.subheader("Probabilitas Awal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ``")
        try:
            total_samples = model.class_count_.sum()
            prior_proba_values = model.class_count_ / total_samples
            prior_proba = pd.Series(prior_proba_values, index=[str(c) for c in model.classes_])
            prior_proba.index = CLASS_NAMES[: len(prior_proba)]
            st.dataframe(prior_proba.rename("Probabilitas Awal"), use_container_width=True)
        except Exception:
            st.warning("Tidak dapat menampilkan Probabilitas Awal.")

    with tab_likelihood:
        st.subheader("Parameter Likelihood ($\\mu$ dan $\\sigma^2$)")
        try:
            mean_df = pd.DataFrame(model.theta_.T, index=feature_names, columns=CLASS_NAMES[:model.theta_.shape[0]])
            mean_df.index = [FEATURE_MAP.get(f, f) for f in feature_names]
            st.markdown("##### Rata-rata ($\\mu$) per Fitur dan Kelas")
            st.dataframe(mean_df, use_container_width=True)
            variance_df = pd.DataFrame(model.var_.T, index=feature_names, columns=CLASS_NAMES[:model.var_.shape[0]])
            variance_df.index = [FEATURE_MAP.get(f, f) for f in feature_names]
            st.markdown("##### Variansi ($\\sigma^2$) per Fitur dan Kelas")
            st.dataframe(variance_df, use_container_width=True)
        except Exception:
            st.warning("Tidak dapat menampilkan parameter likelihood.")

# --- Halaman: Hasil Diagnosis ---
def prediction_page():
    st.title('✅ Hasil Diagnosis & Probabilitas')
    st.markdown("---")

    prediction = st.session_state.get('prediction')
    proba = st.session_state.get('proba')
    input_data_values = st.session_state.get('input_data')
    feature_names = st.session_state.get('feature_names')

    if prediction is None or proba is None:
        st.warning("Anda belum melakukan prediksi. Silakan kembali ke halaman 'Input & Algoritma' untuk mengisi input dan menekan tombol prediksi.")
        return

    st.header("Diagnosis Akhir")
    if int(prediction) == 1:
        st.error('Pasien **MUNGKIN TERKENA** Penyakit Jantung (Kelas 1)')
    else:
        st.success('Pasien **TIDAK TERKENA** Penyakit Jantung (Kelas 0)')

    st.markdown("---")
    st.header("Inti Prediksi")
    proba_arr = np.array(proba, dtype=float)
    # pastikan panjang sesuai
    labels = CLASS_NAMES[:proba_arr.shape[0]]
    proba_df = pd.DataFrame({
        'Kelas': labels,
        'Inti Prediksi': proba_arr
    })
    proba_df['Inti Prediksi'] = proba_df['Inti Prediksi'].map('{:.4f}'.format)
    st.dataframe(proba_df.set_index('Kelas'), use_container_width=True)

    st.info(f"Keputusan diagnosis adalah **Kelas {prediction}** karena memiliki probabilitas tertinggi.")
    st.markdown("---")

    if input_data_values is not None and feature_names is not None:
        st.header("Input Data Pasien yang Digunakan")
        try:
            input_series = pd.Series(input_data_values.flatten(), index=feature_names)
            input_series.index = [FEATURE_MAP.get(f, f) for f in feature_names]
            st.dataframe(input_series.rename("Nilai Input Pasien").to_frame(), use_container_width=True)
        except Exception:
            st.warning("Gagal menampilkan input pasien.")

# --- Fungsi Utama Aplikasi ---
def app():
    # Inisialisasi session_state dengan nilai default yang aman
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None

    # Sidebar: Uploader
    st.sidebar.header("📁 Unggah Data CSV")
    uploaded_file = st.sidebar.file_uploader(
        "Pilih File Data Penyakit Jantung (CSV)",
        type=["csv"],
        help="Pastikan kolom target ada di kolom terakhir.",
        key="file_uploader"
    )

    # Jika ada file baru yang diunggah (berbeda dari yang tersimpan)
    if uploaded_file is not None:
        prev = st.session_state.get('uploaded_file')
        try:
            same_file = (prev is not None and getattr(prev, "name", None) == getattr(uploaded_file, "name", None))
        except Exception:
            same_file = False

        if not same_file:
            # Proses file baru
            st.session_state.uploaded_file = uploaded_file
            model, feature_names, target_classes, accuracy = get_model_info_from_file(uploaded_file)
            st.session_state.model = model
            st.session_state.feature_names = feature_names
            st.session_state.target_classes = target_classes
            st.session_state.accuracy = accuracy

            if model is not None:
                st.sidebar.success(f"✅ Model dilatih. Akurasi Test: {accuracy:.2f}")
                # Inisialisasi input_data dengan ukuran yang benar
                st.session_state.input_data = np.array([[0.0] * len(feature_names)])
                st.session_state.page = "detail_fitur"
                st.rerun()
            else:
                st.sidebar.error("❌ Gagal memproses dataset. Pastikan semua kolom fitur numerik dan kolom target benar.")
                st.session_state.page = "home"
                st.session_state.model = None
                st.session_state.feature_names = None
                st.session_state.target_classes = None
                st.session_state.accuracy = None

    # Jika user menghapus file (uploaded_file is None)
    elif uploaded_file is None and st.session_state.get('uploaded_file') is not None and st.session_state.uploaded_file is not None:
        # user cleared uploader in UI -> reset app state
        st.session_state.uploaded_file = None
        st.session_state.model = None
        st.session_state.feature_names = None
        st.session_state.target_classes = None
        st.session_state.accuracy = None
        st.session_state.input_data = None
        st.session_state.prediction = None
        st.session_state.proba = None
        st.session_state.page = "home"
        st.rerun()

    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Navigasi")
    pages_map = {
        "home": "Home",
        "detail_fitur": "Fitur Data",
        "app": "Input & Algoritma",
        "prediction": "Hasil Diagnosis"
    }
    options = list(pages_map.values())

    # Disabled if model belum tersedia
    model_ready = st.session_state.get('model') is not None and st.session_state.get('feature_names') is not None

    # Default selected index logic
    current_page_key = st.session_state.get('page') or ("home" if not model_ready else "app")
    if current_page_key not in pages_map:
        current_page_key = "home" if not model_ready else "app"
        st.session_state.page = current_page_key

    default_index = options.index(pages_map[current_page_key]) if pages_map[current_page_key] in options else 0

    selected = st.sidebar.radio("Pilih Halaman", options, index=default_index, disabled=not model_ready, key="nav_radio")

    # Sync selection back to page key
    for k, v in pages_map.items():
        if v == selected:
            st.session_state.page = k
            break

    # Tampilkan halaman sesuai state
    page = st.session_state.get('page')
    if page == "home":
        home_page()
    elif page == "detail_fitur":
        feature_page(st.session_state.get('feature_names'))
    elif page == "app":
        input_and_model_detail_page(st.session_state.get('model'), st.session_state.get('feature_names'), st.session_state.get('target_classes'))
    elif page == "prediction":
        prediction_page()
    else:
        home_page()

if __name__ == "__main__":
    app()
