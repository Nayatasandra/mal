import joblib
import pandas as pd
from scipy.spatial.distance import cdist
import streamlit as st
from pyclustering.cluster.cure import cure
import numpy as np

# Fungsi untuk memproses data baru
def preprocess_data(data):
    data['Jenis_Kelamin'] = data['Jenis_Kelamin'].map({'Perempuan': 1, 'Laki-laki': 0})
    data['Penyakit_Kronis'] = data['Penyakit_Kronis'].map({'Ya': 1, 'Tidak': 0})
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
        'Tidak ada masalah': 4,
        'Ada kesulitan ringan': 3,
        'Kesulitan berat': 2,
        'Kesulitan berat dalam mengunyah atau makan': 1
    })
    
    data['Masalah_Kognitif'] = data['Masalah_Kognitif'].map({
        'Tidak ada masalah daya ingat': 3,
        'Ada sedikit masalah daya ingat': 2,
        'Masalah daya ingat yang signifikan': 1
    })

    # Seleksi hanya kolom numerik
    data = data.drop(columns=['Nama'])

    return data

# Fungsi untuk memprediksi kluster untuk data baru
def predict_cure(centroids, data_baru):
    distances = cdist([data_baru], centroids, metric='euclidean')
    return np.argmin(distances)

# Load the representative points of the clusters
def load_model(filename='cure_model.pkl'):
    return joblib.load(filename)

# Load the fitted scaler
def load_scaler(filename='scaler.pkl'):
    return joblib.load(filename)

# Fungsi untuk memberikan intervensi berdasarkan kluster
def intervensi(cluster):
    if cluster == 2:
        return ("Klaster 1 - Malnutrisi Berat:\n"
                "Lansia di klaster ini memiliki berat badan rendah, mengalami penurunan berat badan lebih dari 3 kg dalam tiga bulan terakhir, "
                "asupan protein sangat rendah, serta masalah kesehatan kronis dan kognitif.\n"
                "Intervensi: Lansia memerlukan program gizi khusus dengan fokus pada peningkatan berat badan dan asupan makanan yang lebih bergizi, "
                "serta intervensi medis untuk masalah kronis.")
    elif cluster == 1:
        return ("Klaster 2 - Malnutrisi Ringan:\n"
                "Lansia dalam kelompok ini menunjukkan sedikit penurunan berat badan, dengan frekuensi makan dua kali sehari, namun tidak memiliki masalah kesehatan yang parah.\n"
                "Intervensi: Pemantauan berkala dan saran gizi untuk meningkatkan variasi makanan dan asupan protein.")
    elif cluster == 0:
        return ("Klaster 3 - Cukup Gizi:\n"
                "Lansia dalam klaster ini memiliki berat badan stabil, tidak ada penurunan berat badan yang signifikan, dan asupan makanan mencukupi.\n"
                "Intervensi: Lansia hanya perlu pemantauan reguler dan memastikan bahwa asupan gizi tetap terjaga.")
    else:
        return "Klaster tidak diketahui."

# Judul aplikasi
st.title("Prediksi Kluster Lansia")

# Input data dari pengguna
nama = st.text_input("Nama:")
umur = st.number_input("Umur:", min_value=0, max_value=150, value=75)
jenis_kelamin = st.selectbox("Jenis Kelamin:", ['Laki-laki', 'Perempuan'])
li_la = st.number_input("LiLA:", value=30)
berat_badan = st.number_input("Berat Badan (kg):", value=50)
tinggi_badan = st.number_input("Tinggi Badan (cm):", value=160)
penurunan_berat_badan = st.number_input("Penurunan Berat Badan:", value=0)
frekuensi_makan = st.selectbox("Frekuensi Makan:", ['Satu kali atau kurang', 'Dua kali', 'Tiga kali atau lebih'])
variasi_makanan = st.selectbox("Variasi Makanan:", ['Makanan bervariasi (misalnya daging, sayuran, buah)', 'Makanan terbatas pada beberapa jenis saja', 'Makanan sangat terbatas atau monoton'])
asupan_protein = st.selectbox("Asupan Protein:", ['Dua kali sehari atau lebih', 'Sekali sehari', 'Kurang dari sekali sehari atau tidak pernah'])
mobilitas = st.selectbox("Mobilitas:", ['Bergerak dengan bebas tanpa bantuan', 'Bergerak dengan bantuan', 'Tidak dapat bergerak tanpa bantuan orang lain'])
aktivitas_sehari_hari = st.selectbox("Aktivitas Sehari-hari:", ['Tidak ada kesulitan', 'Kesulitan ringan', 'Kesulitan berat'])
kesehatan_mulut = st.selectbox("Kesehatan Mulut:", ['Tidak ada masalah', 'Ada kesulitan ringan', 'Kesulitan berat', 'Kesulitan berat dalam mengunyah atau makan'])
stres = st.selectbox("Stres:", ['Ya', 'Tidak'])
masalah_kognitif = st.selectbox("Masalah Kognitif:", ['Tidak ada masalah daya ingat', 'Ada sedikit masalah daya ingat', 'Masalah daya ingat yang signifikan'])

# Tombol untuk memproses prediksi
if st.button("Prediksi Kluster"):
    # Siapkan data baru
    data_baru = pd.DataFrame({
        'Nama': [nama],
        'Umur': [umur],
        'Jenis_Kelamin': [jenis_kelamin],
        'LiLA': [li_la],
        'Berat_Badan': [berat_badan],
        'Tinggi_Badan': [tinggi_badan],
        'Penurunan_Berat_Badan': [penurunan_berat_badan],
        'Frekuensi_Makan': [frekuensi_makan],
        'Variasi_Makanan': [variasi_makanan],
        'Asupan_Protein': [asupan_protein],
        'Mobilitas': [mobilitas],
        'Aktivitas_Sehari_hari': [aktivitas_sehari_hari],
        'Kesehatan_Mulut': [kesehatan_mulut],
        'Penyakit_Kronis': ['Tidak'],  # Misalnya, Anda dapat mengubah ini sesuai input pengguna
        'Stres': [stres],
        'Masalah_Kognitif': [masalah_kognitif]
    })

    scaler = load_scaler()
    rep_points = load_model()

    # Preprocess the new data
    data_baru_preprocessed = preprocess_data(data_baru)

    # Scale the new data
    data_baru_scaled = scaler.transform(data_baru_preprocessed)

    # Classify the new data
    def classify_new_data(new_data_scaled, clusters):
        # Calculate distances to the representative points of the clusters
        distances = []
        for cluster in clusters:
            # Calculate the distance to each representative point in the cluster
            dist = np.linalg.norm(new_data_scaled - cluster.repPoints, axis=1)
            distances.append(np.min(dist))  # Take the minimum distance to a representative point
        return np.argmin(distances) + 1  # Return the cluster label (1-based)

    # Perform classification
    cluster_predicted = classify_new_data(data_baru_scaled, clusters)

    st.write(f"Data baru masuk ke dalam klaster: ")

    # Fungsi untuk memberikan intervensi berdasarkan kluster
    def intervensi(cluster):
        if cluster == 2:
            return ("Klaster 1 - Malnutrisi Berat:\n"
                    "Lansia di klaster ini memiliki berat badan rendah, mengalami penurunan berat badan lebih dari 3 kg dalam tiga bulan terakhir, "
                    "asupan protein sangat rendah, serta masalah kesehatan kronis dan kognitif.\n"
                    "Intervensi: Lansia memerlukan program gizi khusus dengan fokus pada peningkatan berat badan dan asupan makanan yang lebih bergizi, "
                    "serta intervensi medis untuk masalah kronis.")
        elif cluster == 1:
            return ("Klaster 2 - Malnutrisi Ringan:\n"
                    "Lansia dalam kelompok ini menunjukkan sedikit penurunan berat badan, dengan frekuensi makan dua kali sehari, namun tidak memiliki masalah kesehatan yang parah.\n"
                    "Intervensi: Pemantauan berkala dan saran gizi untuk meningkatkan variasi makanan dan asupan protein.")
        elif cluster == 0:
            return ("Klaster 3 - Cukup Gizi:\n"
                    "Lansia dalam klaster ini memiliki berat badan stabil, tidak ada penurunan berat badan yang signifikan, dan asupan makanan mencukupi.\n"
                    "Intervensi: Lansia hanya perlu pemantauan reguler dan memastikan bahwa asupan gizi tetap terjaga.")
        else:
            return "Klaster tidak diketahui."

    # Tampilkan intervensi
    st.write(intervensi(cluster_predicted))
