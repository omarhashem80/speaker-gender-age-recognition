# import os
# import librosa
# import numpy as np
# import pandas as pd
# import parselmouth
# from parselmouth.praat import call
#
#
# # base = "D:/College/Third Year/Second Term/Pattern/Project/data/"
#
#
# i = 0
# # =============================
# # 📥 Audio Loading & Trimming
# # =============================
# def load_audio(file_path, sr=16000):
#     y, _ = librosa.load(file_path, sr=sr)
#     y_trimmed, _ = librosa.effects.trim(y, top_db=20)
#     return y_trimmed, sr
#
#
# def aggregate_features(features, axis=1):
#     return np.concatenate([np.mean(features, axis=axis), np.std(features, axis=axis)])
#
#
# # =============================
# # 🎵 MFCCs + Deltas
# # =============================
# def extract_mfccs(y, sr, n_mfcc=13):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     return aggregate_features(np.vstack([mfcc, delta, delta2]))
#
#
# # =============================
# # 📏 Pitch (f0 stats)
# # =============================
# def extract_pitch_stats(y, sr):
#     f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
#     f0 = f0[~np.isnan(f0)]
#     return np.array([
#         np.mean(f0) if f0.size > 0 else 0,
#         np.std(f0) if f0.size > 0 else 0,
#         np.min(f0) if f0.size > 0 else 0,
#         np.max(f0) if f0.size > 0 else 0,
#     ])
#
#
# # =============================
# # 🎼 Formants (mean of F1–F3)
# # =============================
# def extract_formants_stats(file_path):
#     snd = parselmouth.Sound(file_path)
#     formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
#     f1, f2, f3 = [], [], []
#     for t in np.arange(0, snd.duration, 0.01):
#         f1.append(formant.get_value_at_time(1, t))
#         f2.append(formant.get_value_at_time(2, t))
#         f3.append(formant.get_value_at_time(3, t))
#     return np.array([
#         np.nanmean(f1) if len(f1) else 0,
#         np.nanmean(f2) if len(f2) else 0,
#         np.nanmean(f3) if len(f3) else 0,
#     ])
#
#
# # =============================
# # 📈 Spectral Features
# # =============================
# def extract_spectral_features(y, sr):
#     centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     combined = np.vstack([centroid, bandwidth, contrast, rolloff])
#     return aggregate_features(combined)
#
#
# # =============================
# # 🔄 Voice Quality (Jitter, Shimmer, HNR)
# # =============================
# def extract_voice_quality(file_path, sr=16000, min_voiced_frames=10):
#     y, _ = librosa.load(file_path, sr=sr)
#     y_trimmed, _ = librosa.effects.trim(y, top_db=20)
#     snd = parselmouth.Sound(y_trimmed, sr)
#
#     f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=50, fmax=500, sr=sr)
#     num_voiced = np.count_nonzero(voiced_flag)
#
#     if num_voiced < min_voiced_frames:
#         return np.array([0.0, 0.0, 0.0])
#
#     point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
#     harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
#
#     try:
#         jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
#     except:
#         jitter = 0.0
#
#     try:
#         shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
#     except:
#         shimmer = 0.0
#
#     try:
#         hnr = call(harmonicity, "Get mean", 0, 0)
#     except:
#         hnr = 0.0
#
#     return np.array([jitter, shimmer, hnr])
#
#
# # =============================
# # 🔢 ZCR (Zero-Crossing Rate)
# # =============================
# def extract_zcr(y):
#     zcr = librosa.feature.zero_crossing_rate(y)
#     return np.array([np.mean(zcr)])
#
# # =============================
# # 🧠 Batch Processing: All Features Combined for a Batch
# # =============================
# def extract_features(file_path):
#     y, sr = load_audio(file_path)
#
#     mfccs = extract_mfccs(y, sr)
#     pitch = extract_pitch_stats(y, sr)
#     formants = extract_formants_stats(file_path)
#     spectral = extract_spectral_features(y, sr)
#     voice_quality = extract_voice_quality(file_path, sr)
#     zcr = extract_zcr(y)
#
#     features = np.hstack([mfccs, pitch, formants, spectral, voice_quality, zcr])
#     return features
#
#
# # =============================
# # 📊 Load CSV, Process Features & Save
# # =============================
# def process_csv(df, output_path):
#     global i
#     features_list = []
#     labels = []
#
#     for index, row in df.iterrows():
#         file_path = row['path']
#         label = row['label']
#         print(i)
#         i += 1
#         features = extract_features(file_path)
#         features_list.append(features)
#         labels.append(label)
#
#     # Combine features and labels
#     features_array = np.array(features_list)
#     labels_array = np.array(labels)
#
#     # Save to new CSV
#     result_df = pd.DataFrame(features_array)
#     result_df['label'] = labels_array
#     result_df.to_csv(output_path, index=False)
#
# # =============================
# # 🧑‍💻 Example Usage
# # =============================
#
#
# def formater(row):
#     if row.endswith('.mp3'):
#         base = row[:-4]  # Remove the last 4 characters: '.mp3'
#         return f"D:/College/Third Year/Second Term/Pattern/Project/data/{base}_preprocessed.mp3"
#
#
# if __name__ == "__main__":
#     input_csv = "../../../filtered_data_cleaned.csv"  # Path to your input CSV file with paths and labels
#     df = pd.read_csv(input_csv)
#     df['path'] = df['path'].apply(formater)
#     output_csv = "D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/output_features3k.csv"   # Path to save the output CSV with features and labels
#     process_csv(df[2000:3000], output_csv)

import os
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

i = 0

# =============================
# 📥 Audio Loading & Trimming
# =============================
def load_audio(file_path, sr=16000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if y.size == 0:
            return None, None
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if y_trimmed.size == 0:
            return None, None
        return y_trimmed, sr
    except Exception as e:
        print(f"[ERROR] Loading audio failed for {file_path}: {e}")
        return None, None

def aggregate_features(features, axis=1):
    return np.concatenate([np.mean(features, axis=axis), np.std(features, axis=axis)])

# =============================
# 🎵 MFCCs + Deltas
# =============================
def extract_mfccs(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return aggregate_features(np.vstack([mfcc, delta, delta2]))

# =============================
# 📏 Pitch (f0 stats)
# =============================
def extract_pitch_stats(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return np.array([
        np.mean(f0) if f0.size > 0 else 0,
        np.std(f0) if f0.size > 0 else 0,
        np.min(f0) if f0.size > 0 else 0,
        np.max(f0) if f0.size > 0 else 0,
    ])

# =============================
# 🎼 Formants (mean of F1–F3)
# =============================
def extract_formants_stats(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
        f1, f2, f3 = [], [], []
        for t in np.arange(0, snd.duration, 0.01):
            f1.append(formant.get_value_at_time(1, t))
            f2.append(formant.get_value_at_time(2, t))
            f3.append(formant.get_value_at_time(3, t))
        return np.array([
            np.nanmean(f1) if len(f1) else 0,
            np.nanmean(f2) if len(f2) else 0,
            np.nanmean(f3) if len(f3) else 0,
        ])
    except Exception as e:
        print(f"[ERROR] Formant extraction failed for {file_path}: {e}")
        return np.array([0.0, 0.0, 0.0])

# =============================
# 📈 Spectral Features
# =============================
def extract_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    combined = np.vstack([centroid, bandwidth, contrast, rolloff])
    return aggregate_features(combined)

# =============================
# 🔄 Voice Quality (Jitter, Shimmer, HNR)
# =============================
def extract_voice_quality(file_path, sr=16000, min_voiced_frames=10):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if y_trimmed.size == 0:
            return np.array([0.0, 0.0, 0.0])

        snd = parselmouth.Sound(y_trimmed, sr)

        f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=50, fmax=500, sr=sr)
        num_voiced = np.count_nonzero(voiced_flag)

        if num_voiced < min_voiced_frames:
            return np.array([0.0, 0.0, 0.0])

        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return np.array([jitter, shimmer, hnr])
    except Exception as e:
        print(f"[ERROR] Voice quality extraction failed for {file_path}: {e}")
        return np.array([0.0, 0.0, 0.0])

# =============================
# 🔢 ZCR (Zero-Crossing Rate)
# =============================
def extract_zcr(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.array([np.mean(zcr)])

# =============================
# 🧠 Batch Processing: All Features Combined for a Batch
# =============================
def extract_features(file_path):
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        print(f"[SKIPPED] Invalid or empty audio: {file_path}")
        return None

    try:
        mfccs = extract_mfccs(y, sr)
        pitch = extract_pitch_stats(y, sr)
        formants = extract_formants_stats(file_path)
        spectral = extract_spectral_features(y, sr)
        voice_quality = extract_voice_quality(file_path, sr)
        zcr = extract_zcr(y)

        features = np.hstack([mfccs, pitch, formants, spectral, voice_quality, zcr])
        return features
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {file_path}: {e}")
        return None

# =============================
# 📊 Load CSV, Process Features & Save
# =============================
def process_csv(df, output_path):
    global i
    features_list = []
    labels = []

    for index, row in df.iterrows():
        file_path = row['path']
        label = row['label']
        print(i)
        i += 1

        features = extract_features(file_path)
        if features is None:
            continue  # Skip invalid or failed files

        features_list.append(features)
        labels.append(label)

    if features_list:
        features_array = np.array(features_list)
        labels_array = np.array(labels)

        result_df = pd.DataFrame(features_array)
        result_df['label'] = labels_array
        result_df.to_csv(output_path, index=False)
        print(f"✅ Features saved to {output_path}")
    else:
        print("⚠️ No valid features extracted. Nothing saved.")

# =============================
# 🧑‍💻 Example Usage
# =============================
def formater(row):
    if row.endswith('.mp3'):
        base = row[:-4]  # Remove the last 4 characters: '.mp3'
        return f"D:/College/Third Year/Second Term/Pattern/Project/data/{base}_preprocessed.mp3"

if __name__ == "__main__":
    input_csv = "../../../filtered_data_cleaned.csv"  # Path to your input CSV file with paths and labels
    df = pd.read_csv(input_csv)
    df['path'] = df['path'].apply(formater)
    output_csv = "D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/output_features4k.csv"
    process_csv(df[3000:4000], output_csv)
