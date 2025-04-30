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
def extract_mfccs(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return aggregate_features(np.vstack([mfcc, delta, delta2]))


def extract_pitch_stats(y, sr):
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, hop_length=512)
    f0 = f0[~np.isnan(f0)]
    voiced_fraction = np.mean(voiced_flag) if voiced_flag.size > 0 else 0
    hop_time = 512 / sr
    voiced_durations = [sum(1 for _ in group) * hop_time for key, group in groupby(voiced_flag) if key]
    avg_voiced_duration = np.mean(voiced_durations) if voiced_durations else 0
    return np.array([
        np.mean(f0) if f0.size > 0 else 0,
        np.std(f0) if f0.size > 0 else 0,
        np.min(f0) if f0.size > 0 else 0,
        np.max(f0) if f0.size > 0 else 0,
        voiced_fraction,
        avg_voiced_duration
    ])


def extract_formants_stats(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
        times = np.arange(0, snd.duration, 0.01)
        formant_means = []
        bandwidth_means = []
        for i in range(1, 4):
            fv = [formant.get_value_at_time(i, t) for t in times]
            bv = [formant.get_bandwidth_at_time(i, t) for t in times]
            formant_means.append(np.nanmean(fv))
            bandwidth_means.append(np.nanmean(bv))
        return np.hstack([np.nan_to_num(formant_means), np.nan_to_num(bandwidth_means)])
    except Exception:
        return np.zeros(6)


def extract_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return aggregate_features(np.vstack([centroid, bandwidth, contrast, rolloff]))


def extract_voice_quality(y, sr):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if y_trimmed.size == 0:
            return np.zeros(3)
        snd = parselmouth.Sound(y_trimmed, sr)
        f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=50, fmax=500, sr=sr)
        if np.count_nonzero(voiced_flag) < 10:
            return np.zeros(3)
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return np.array([jitter, shimmer, hnr])
    except Exception:
        return np.zeros(3)


def extract_zcr(y):
    return np.array([np.mean(librosa.feature.zero_crossing_rate(y))])


def extract_spectral_contrast(y, sr, n_bands=6, fmin=100.0):
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=fmin)
        return aggregate_features(contrast)
    except Exception:
        return np.zeros(2 * (n_bands + 1))


def extract_chroma_cqt(y, sr, n_chroma=12, fmin_note="C2"):
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma,
                                            fmin=librosa.note_to_hz(fmin_note), bins_per_octave=36)
        return aggregate_features(chroma)
    except Exception:
        return np.zeros(2 * n_chroma)


def extract_tonnetz(y, sr, fmin_note="C2"):
    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr, fmin=librosa.note_to_hz(fmin_note))
        return aggregate_features(tonnetz)
    except Exception:
        return np.zeros(12)


def extract_energy_stats(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_range = np.max(rms) - np.min(rms) if rms.size > 0 else 0
    return np.array([np.mean(rms), np.std(rms), rms_range])


def extract_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_power = S ** 2
        S_power /= np.sum(S_power, axis=0, keepdims=True) + 1e-10
        entropy = -np.sum(S_power * np.log2(S_power + 1e-10), axis=0)
        return np.array([np.mean(entropy), np.std(entropy)])
    except Exception:
        return np.zeros(2)


# === Feature Extraction Wrapper ===
def extract_features_from_path(args):
    file_path, label = args
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        return None
    try:
        features = np.hstack([
            extract_mfccs(y, sr),
            extract_pitch_stats(y, sr),
            extract_formants_stats(file_path),
            extract_spectral_features(y, sr),
            extract_spectral_contrast(y, sr),
            extract_chroma_cqt(y, sr),
            extract_tonnetz(y, sr),
            extract_voice_quality(y, sr),
            extract_zcr(y),
            extract_energy_stats(y, sr),
            extract_spectral_entropy(y, sr)
        ])
        return (features, label)
    except Exception:
        return None


# === Parallel Processing ===
def process_csv(df, output_prefix, start_i=0):
    inputs = [(row.path, row.label) for row in df.itertuples(index=False)]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_features_from_path, inputs), total=len(inputs)))

    features_list = []
    labels = []
    for result in results:
        if result:
            features, label = result
            features_list.append(features)
            labels.append(label)

    output_path = f"{output_prefix}_{start_i}_{start_i + len(features_list) - 1}.csv"
    save_chunk(features_list, labels, output_path)


def save_chunk(features_list, labels, output_path):
    df = pd.DataFrame(features_list)
    df['label'] = labels
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(features_list)} entries to {output_path}")


# === Path Formatter ===
def formater(row):
    if row.endswith('.mp3'):
        base = row[:-4]
        return f"D:/College/Third Year/Second Term/Pattern/Project/data/{base}_preprocessed.mp3"


# === Main Execution ===
if __name__ == "__main__":
    start_time = time.time()

    input_csv = "../../../filtered_data_cleaned.csv"
    df = pd.read_csv(input_csv)
    df['path'] = df['path'].apply(formater)
    output_csv_prefix = "D:/College/Third Year/Second Term/Pattern/Project/speaker-gender-age-recognition/output_features160_end"
    process_csv(df[:5], output_csv_prefix)

    elapsed_time = time.time() - start_time
    print(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")