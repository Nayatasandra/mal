import joblib
import pandas as pd
from scipy.spatial.distance import cdist
import streamlit as st
from pyclustering.cluster.cure import cure
import numpy as np
from streamlit_option_menu import option_menu

def preprocess_data(data):
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data['Jenis_Kelamin'] = data['Jenis_Kelamin'].map({'Perempuan': 1, 'Laki-laki': 0})
    data['Penyakit_Kronis'] = data['Penyakit_Kronis'].map({
        'Ya': 1, 'Tidak': 0,
    })
    data['Stres'] = data['Stres'].map({'Ya': 1, 'Tidak': 0})
    data['Frekuensi_Makan'] = data['Frekuensi_Makan'].map({
        'Tiga kali atau lebih': 3,
        'Dua kali': 2,
        'Satu kali atau kurang': 1
    })
    data['Variasi_Makanan'] = data['Variasi_Makanan'].map({
        'Makanan bervariasi (misalnya daging, sayuran, buah)': 3,
        'Makanan terbatas pada beberapa jenis saja': 2,
        'Makanan sangat terbatas atau monoton': 1
    })
    data['Asupan_Protein'] = data['Asupan_Protein'].map({
        'Dua kali sehari atau lebih': 3,
        'Sekali sehari': 2,
        'Kurang dari sekali sehari atau tidak pernah': 1
    })
    data['Mobilitas'] = data['Mobilitas'].map({
        'Bergerak dengan bebas tanpa bantuan': 3,
        'Bergerak dengan bantuan': 2,
        'Tidak dapat bergerak tanpa bantuan orang lain': 1
    })
    data['Aktivitas_Sehari_hari'] = data['Aktivitas_Sehari_hari'].map({
        'Tidak ada kesulitan': 3,
        'Kesulitan ringan': 2,
        'Kesulitan berat': 1
    })
    data['Kesehatan_Mulut'] = data['Kesehatan_Mulut'].map({
        'Tidak ada masalah': 3,
        'Ada kesulitan ringan': 2,
        'Kesulitan berat': 1,
    })
    data['Masalah_Kognitif'] = data['Masalah_Kognitif'].map({
        'Tidak ada masalah daya ingat': 3,
        'Ada sedikit masalah daya ingat': 2,
        'Masalah daya ingat yang signifikan': 1
    })

    return data.drop(columns=['Nama'])

def load_model(filename='cure_model.pkl'):
    return joblib.load(filename)

def load_scaler(filename='scaler.pkl'):
    return joblib.load(filename)


#Website
st.set_page_config(page_title="Elderly Nutriion Care", layout="wide")

selected = option_menu(
        menu_title=None, 
        options=["Beranda", "Deteksi Malnutrisi", "Informasi", "Visualisasi"],
        icons=["house", "activity", "info", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
)

if selected == "Beranda":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Selamat datang di Elderly Nutrition Care!</h1>
            <p>Website ini dirancang untuk membantu mendeteksi malnutrisi pada lansia.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("home.svg")


elif selected == "Deteksi Malnutrisi":
    st.title("Deteksi Malnutrisi pada Lansia")
    
    nama = st.text_input("Nama")
    umur = st.number_input("Umur", value=0)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["--", "Laki-laki", "Perempuan"])
    berat_badan = st.number_input("Berat Badan (kg)", min_value=0)
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0)
    penurunan_berat_badan = st.number_input("Penurunan Berat Badan", value=0)
    frekuensi_makan = st.selectbox("Frekuensi Makan", ["--","Tiga kali atau lebih", "Dua kali", "Satu kali atau kurang"])
    variasi_makanan = st.selectbox("Variasi Makanan:", ["--", 'Makanan bervariasi (misalnya daging, sayuran, buah)', 'Makanan terbatas pada beberapa jenis saja', 'Makanan sangat terbatas atau monoton'])
    asupan_protein = st.selectbox("Asupan Protein:", ["--", 'Dua kali sehari atau lebih', 'Sekali sehari', 'Kurang dari sekali sehari atau tidak pernah'])
    mobilitas = st.selectbox("Mobilitas:", ["--", 'Bergerak dengan bebas tanpa bantuan', 'Bergerak dengan bantuan', 'Tidak dapat bergerak tanpa bantuan orang lain'])
    aktivitas_sehari_hari = st.selectbox("Aktivitas Sehari-hari:", ["--", 'Tidak ada kesulitan', 'Kesulitan ringan', 'Kesulitan berat'])
    kesehatan_mulut = st.selectbox("Kesehatan Mulut:", ["--", 'Tidak ada masalah', 'Ada kesulitan ringan', 'Kesulitan berat'])
    stres = st.selectbox("Stres", ["--", "Ya", "Tidak"])
    penyakit_kronis = st.selectbox("Penyakit Kronis:", ["--", 'Ya', 'Tidak'])
    masalah_kognitif = st.selectbox("Masalah Kognitif", ["--", 'Tidak ada masalah daya ingat', 'Ada sedikit masalah daya ingat', 'Masalah daya ingat yang signifikan'])

    if st.button("Deteksi Kluster"):
        if not nama:
            st.warning("Nama tidak boleh kosong.")
        elif not umur:
            st.warning("Umur tidak boleh kosong.")
        elif umur < 60:
            st.warning("Lansia adalah individu yang berusia 60 tahun keatas.")
        elif jenis_kelamin == "--":
            st.warning("Pilih jenis kelamin.")
        elif not berat_badan:
            st.warning("Berat badan tidak boleh kosong.")
        elif not tinggi_badan:
            st.warning("Tinggi badan tidak boleh kosong.")
        elif penurunan_berat_badan == "--":
            st.warning("Pilih status penurunan berat badan.")
        elif frekuensi_makan == "--":
            st.warning("Frekuensi makan tidak boleh kosong.")
        elif variasi_makanan == "--":
            st.warning("Pilih variasi makanan.")
        elif asupan_protein == "--":
            st.warning("Pilih asupan protein.")
        elif mobilitas == "--":
            st.warning("Pilih status mobilitas.")
        elif aktivitas_sehari_hari == "--":
            st.warning("Pilih aktivitas sehari-hari.")
        elif kesehatan_mulut == "--":
            st.warning("Pilih status kesehatan mulut.")
        elif stres == "--":
            st.warning("Pilih status stres.")
        elif penyakit_kronis == "--":
            st.warning("Pilih status penyakit kronis.")
        elif masalah_kognitif == "--":
            st.warning("Pilih status masalah kognitif.")
        else:
            data_baru = pd.DataFrame({
                'Nama': [nama],
                'Umur': [umur],
                'Jenis_Kelamin': [jenis_kelamin],
                'Berat_Badan': [berat_badan],
                'Tinggi_Badan': [tinggi_badan],
                'Penurunan_Berat_Badan': [penurunan_berat_badan],
                'Frekuensi_Makan': [frekuensi_makan],
                'Variasi_Makanan': [variasi_makanan],
                'Asupan_Protein': [asupan_protein],
                'Mobilitas': [mobilitas],
                'Aktivitas_Sehari_hari': [aktivitas_sehari_hari],
                'Kesehatan_Mulut': [kesehatan_mulut],
                'Penyakit_Kronis': ['Tidak'],
                'Stres': [stres],
                'Masalah_Kognitif': [masalah_kognitif]
            })

        scaler = load_scaler()
        rep_points = load_model()

        data_baru_preprocessed = preprocess_data(data_baru)

        data_baru_scaled = scaler.transform(data_baru_preprocessed)

        def detect_new_data(new_data_scaled, rep_points):
            distances = []
            for reps in rep_points:
                dist = np.linalg.norm(new_data_scaled - reps, axis=1)
                distances.append(np.min(dist))
            return np.argmin(distances) + 1

        cluster_detected = detect_new_data(data_baru_scaled, rep_points)

        # st.write(f"Data baru masuk ke dalam klaster: {cluster_detected}")
        
        def intervensi(cluster):
            if cluster == 2:
                return (
                    "### Klaster Malnutrisi Berat:\n"
                    "**Intervensi:**\n"
                    "- Pendekatan medis yang intensif, termasuk evaluasi komprehensif oleh dokter, ahli gizi, dan tim medis terkait.\n"
                    "- Pemberian nutrisi melalui metode enteral atau parenteral sesuai kebutuhan, untuk memastikan asupan nutrisi yang tepat bagi pasien.\n"
                    "- Terapi fisik dan intervensi mobilitas yang bertahap guna membantu meningkatkan kekuatan fisik.\n"
                    "- Konseling psikologis untuk mengelola stres, depresi, atau kecemasan yang dapat memperburuk kondisi malnutrisi.\n"
                )
            elif cluster == 3:
                return (
                    "### Klaster Malnutrisi Ringan:\n"
                    "**Intervensi:**\n"
                    "- Pengaturan pola makan yang lebih seimbang dan bervariasi untuk mencegah malnutrisi lebih lanjut.\n"
                    "- Kegiatan fisik ringan yang teratur untuk menjaga kesehatan fisik dan mobilitas.\n"
                    "- Konseling psikologis ringan, jika diperlukan, untuk membantu manajemen stres.\n"
                    "- Pengawasan berkala terhadap status gizi untuk memastikan stabilitas kondisi.\n"
                )
            elif cluster == 1:
                return (
                    "### Klaster Cukup Gizi:\n"
                    "**Intervensi:**\n"
                    "- Pemeliharaan gizi dengan pola makan yang seimbang dan aktivitas fisik teratur.\n"
                    "- Konsultasi berkala dengan ahli gizi untuk memastikan asupan nutrisi tetap optimal.\n"
                    "- Pemantauan kondisi kesehatan secara teratur untuk mendeteksi perubahan status gizi atau kondisi fisik.\n"
                )
            else:
                return "Klaster tidak diketahui."

        st.write(intervensi(cluster_detected))
        
elif selected == "Informasi":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Informasi</h1>
            <p>Berdasarkan pedoman World Health Organization (WHO), malnutrisi terbagi menjadi tiga kelompok utama:</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="border: 2px solid #28A745; border-radius: 10px; padding: 20px; height: 400px;">
            <h3 style="color: #28A745;">Malnutrisi Ringan</h3>
            <p><strong>Intervensi:</strong></p>
            <ul>
                <li>Pengaturan pola makan yang lebih seimbang dan bervariasi.</li>
                <li>Kegiatan fisik ringan yang teratur.</li>
                <li>Konseling psikologis ringan, jika diperlukan.</li>
                <li>Pengawasan berkala terhadap status gizi.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="border: 2px solid #FFC107; border-radius: 10px; padding: 20px; height: 400px;">
            <h3 style="color: #FFC107;">Malnutrisi Sedang</h3>
            <p><strong>Intervensi:</strong></p>
            <ul>
                <li>Konsultasi ahli gizi untuk merancang rencana makan yang lebih kaya nutrisi.</li>
                <li>Program rehabilitasi fisik yang ringan.</li>
                <li>Pemberian suplemen gizi, terutama protein dan mikronutrien.</li>
                <li>Pengawasan berkala oleh tenaga medis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="border: 2px solid #DC3545; border-radius: 10px; padding: 20px; height: 400px;">
            <h3 style="color: #DC3545;">Malnutrisi Berat</h3>
            <p><strong>Intervensi:</strong></p>
            <ul>
                <li>Pendekatan medis segera, dengan evaluasi menyeluruh oleh dokter dan ahli gizi.</li>
                <li>Pemberian makanan atau nutrisi enteral/parenteral, jika diperlukan.</li>
                <li>Terapi fisik untuk meningkatkan kekuatan fisik.</li>
                <li>Konseling psikologis untuk mengurangi stres dan kecemasan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
elif selected == "Visualisasi":
    st.markdown(
        """
        ### Visualisasi Hasil Clustering
        Gambar di bawah menunjukkan visualisasi hasil clustering dengan PCA dari hasil penerapan model CURE untuk mengelompokkan status malnutrisi pada lansia dengan data sebanyak 517 baris.. 
        Titik-titik pada visualisasi menggambarkan satu titik data. 
        Pada gambar terdapat 3 kelompok data, masing-masing kelompok ditandai dengan warna titik yang berbeda:
        - **Titik Ungu**: Nutrisi Baik
        - **Titik Hijau**: Malnutrisi Berat
        - **Titik Kuning**: Malnutrisi Ringan
        """
    )
    
    col1, col2, col3 = st.columns([1, 3, 1])  
    
    with col2:
        st.image("visualisasi.png", caption="Hasil Clustering", use_column_width=True)
    
    
footer = """
<style>
    footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 50px;  /* Atur tinggi footer sesuai kebutuhan */
        background-color: #f1f1f1; /* Warna latar belakang footer */
        text-align: center;
        padding: 10px;
        z-index: 1000; /* Pastikan footer berada di atas konten lainnya */
    }
</style>
<footer>
    © 2024 Deteksi Malnutrisi pada Lansia
</footer>
"""
st.markdown(footer, unsafe_allow_html=True)
