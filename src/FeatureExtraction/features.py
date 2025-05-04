import librosa
import librosa.effects
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from itertools import groupby
import warnings

warnings.filterwarnings("ignore")


# === Audio Loading & Trimming ===
def load_audio(file_path, sr=18000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        if y.size == 0:
            return None, None
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if y_trimmed.size == 0:
            return None, None
        return y_trimmed, sr
    except Exception:
        return None, None


def aggregate_features(features, axis=1):
    return np.concatenate([np.mean(features, axis=axis), np.std(features, axis=axis)])


# === Feature Extractors ===


# =============================
# 🎵 MFCCs + Deltas
# =============================
def extract_mfccs(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return aggregate_features(np.vstack([mfcc, delta, delta2]))


# =============================
# 📏 Pitch (f0 stats)
# =============================
def extract_pitch_stats(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, hop_length=512)
    f0 = f0[~np.isnan(f0)]
    return np.array(
        [
            np.mean(f0) if f0.size > 0 else 0,
            np.std(f0) if f0.size > 0 else 0,
            np.min(f0) if f0.size > 0 else 0,
            np.max(f0) if f0.size > 0 else 0,
        ]
    )


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
        return np.array(
            [
                np.nanmean(f1) if len(f1) else 0,
                np.nanmean(f2) if len(f2) else 0,
                np.nanmean(f3) if len(f3) else 0,
            ]
        )
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
        shimmer = call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
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


def extract_spectral_contrast(y, sr, n_bands=6, fmin=100.0):  # Lowered fmin to 100 Hz
    """Extracts spectral contrast with adjusted fmin."""
    if y is None or sr is None:
        return np.zeros(2 * (n_bands + 1))  # Mean/Std for n_bands + overall

    try:
        # Check Nyquist constraint for chosen fmin and n_bands
        nyquist = sr / 2.0
        highest_band_top = fmin * (2**n_bands)
        if highest_band_top > nyquist:
            print(
                f" Highest spectral contrast band ({highest_band_top:.1f} Hz) exceeds Nyquist ({nyquist:.1f} Hz). Consider reducing n_bands or fmin."
            )
            # Option: Adjust n_bands dynamically, or return zeros, or proceed with caution
            # For simplicity, return zeros here, but adjustment might be better
            return np.zeros(2 * (n_bands + 1))

        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_bands=n_bands, fmin=fmin
        )
        # Aggregate mean and std dev
        return aggregate_features(contrast)
    except Exception as e:
        print(f" Spectral contrast extraction failed: {e}")
        return np.zeros(2 * (n_bands + 1))


def extract_chroma_cqt(y, sr, n_chroma=12, fmin_note="C2"):

    try:
        fmin_hz = librosa.note_to_hz(fmin_note)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, n_chroma=n_chroma, fmin=fmin_hz, bins_per_octave=36
        )
        return aggregate_features(chroma)
    except Exception as e:
        print(f" Chroma CQT extraction failed: {e}")
        return np.zeros(2 * n_chroma)


def extract_tonnetz(y, sr, fmin_note="C2"):

    try:
        fmin_hz = librosa.note_to_hz(fmin_note)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr, fmin=fmin_hz)
        return aggregate_features(tonnetz)
    except Exception as e:
        print(f" Tonnetz extraction failed: {e}")
        return np.zeros(2 * 6)


# === Feature Extraction Wrapper ===
def extract_features_from_path(args):
    file_path, label = args
    print(file_path, label)
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        return None
    try:
        features = np.hstack(
            [
                extract_mfccs(y, sr),
                extract_pitch_stats(y, sr),
                extract_formants_stats(file_path),
                extract_spectral_features(y, sr),
                extract_spectral_contrast(y, sr),
                extract_chroma_cqt(y, sr),
                extract_tonnetz(y, sr),
                extract_voice_quality(y, sr),
                extract_zcr(y),
            ]
        )
        return (features, label)
    except Exception:
        return None


# === Parallel Processing ===
def process_csv(df, output_prefix, isTrain=False, start_i=0):
    inputs = [(row.path, row.label) for row in df.itertuples(index=False)]
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(extract_features_from_path, inputs), total=len(inputs))
        )

    features_list = []
    labels = []
    for result in results:
        if result:
            features, label = result
            features_list.append(features)
            labels.append(label)

    output_path = f"{output_prefix}_{start_i}_{start_i + len(features_list) - 1}.csv"
    return save_chunk(features_list, labels, output_path, isTrain)


def save_chunk(features_list, labels, output_path, isTrain) -> pd.DataFrame:
    df = pd.DataFrame(features_list)
    if isTrain:
        df["label"] = labels
    df.to_csv(output_path, index=False)
    print(f"Saved {len(features_list)} entries to {output_path}")
    return df


# === Path Formatter ===
def formater(data_dir, row):

    if row.endswith(".mp3"):
        base = row[:-4]
        return f"data\\preprocessed\\{base}_preprocessed.mp3"
    elif row.endswith(".wav"):
        base = row[:-4]
        return f"data\\preprocessed\\{base}_preprocessed.wav"


# === Main Execution ===
# if __name__ == "__main__":
#     start_time = time.time()

#     input_csv = "filtered_data_cleaned.csv"
#     df = pd.read_csv(input_csv)
#     df["path"] = df["path"].apply(formater)
#     output_csv_prefix = "output_features.csv"
#     features_df = process_csv(df[:5], output_csv_prefix)
#     print(features_df.head())
#     elapsed_time = time.time() - start_time
#     print(f"Total execution time: {elapsed_time:.2f} seconds")
