# %% [markdown]
# ## Step 2: Feature Extraction (with Jitter & Shimmer)
#
# This step takes the preprocessed audio files from Step 1, extracts various acoustic features using `librosa` and `parselmouth-praat`, and saves them into compressed `.npz` files. It includes caching and multiprocessing.
#
# **Features Extracted:**
# - **Librosa (Frame-wise):**
#   - Mel-Frequency Cepstral Coefficients (MFCCs)
#   - Root Mean Square Energy (RMS)
#   - Spectral Centroid
#   - Spectral Bandwidth
#   - Spectral Contrast
#   - Spectral Flatness
#   - Spectral Rolloff
#   - Pitch (Fundamental Frequency - F0 using PYIN)
#   - Zero-Crossing Rate (ZCR)
# - **Parselmouth/Praat (Scalar per file):**
#   - Pitch (Praat's algorithm - Median F0)
#   - Jitter (local, local_absolute, rap, ppq5)
#   - Shimmer (local, local_db, apq3, apq5, apq11)

# %%
import os
import numpy as np
import librosa  # pip install librosa
import parselmouth  # pip install praat-parselmouth
from parselmouth.praat import call  # To call Praat functions
from multiprocessing import Pool, cpu_count
import traceback  # For detailed error logging
import warnings  # To suppress specific warnings if needed

# Suppress UserWarnings from librosa related to audioread or other backend issues if they occur
# warnings.filterwarnings('ignore', category=UserWarning)
# Suppress RuntimeWarnings often encountered in pitch/spectral calculations with silence
warnings.filterwarnings("ignore", category=RuntimeWarning)


# %%
# --- Configuration ---
PREPROCESSED_FOLDER = "./data/preprocessed/"  # Input for Step 2 (Output of Step 1)
FEATURE_FOLDER = "./data/features/"  # Output for Step 2

# Librosa Feature Extraction Parameters
SR = 16000  # Target Sample Rate for librosa features
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
N_MELS = 128  # Number of Mel bands
N_MFCC = 20  # Number of MFCCs
FMIN_PITCH_LIBROSA = librosa.note_to_hz("C2")  # Min frequency for librosa pitch
FMAX_PITCH_LIBROSA = librosa.note_to_hz("C7")  # Max frequency for librosa pitch

# Praat Pitch Parameters (adjust as needed)
PRAAT_PITCH_FLOOR = 75.0  # Pitch floor (Hz) for Praat analysis
PRAAT_PITCH_CEILING = 600.0  # Pitch ceiling (Hz) for Praat analysis

# --- Ensure Output Directory Exists ---
# Create the directory if it doesn't exist when you run the script
# os.makedirs(FEATURE_FOLDER, exist_ok=True)


# %%
def extract_praat_features(sound, pitch_floor=75.0, pitch_ceiling=600.0):
    """
    Extracts scalar Jitter, Shimmer, and median F0 using Parselmouth/Praat.

    Args:
        sound (parselmouth.Sound): Parselmouth sound object.
        pitch_floor (float): Minimum pitch frequency for analysis.
        pitch_ceiling (float): Maximum pitch frequency for analysis.

    Returns:
        dict: Dictionary containing scalar Praat features. Returns NaNs on failure.
    """
    praat_features = {
        "f0_median_praat": np.nan,
        "jitter_local": np.nan,
        "jitter_local_abs": np.nan,
        "jitter_rap": np.nan,
        "jitter_ppq5": np.nan,
        "shimmer_local": np.nan,
        "shimmer_local_db": np.nan,
        "shimmer_apq3": np.nan,
        "shimmer_apq5": np.nan,
        "shimmer_apq11": np.nan,
    }
    try:
        # --- Pitch ---
        # Time step: 0.0 = auto CQT_DURATION
        pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
        praat_features["f0_median_praat"] = call(
            pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"
        )  # Get median pitch

        # --- PointProcess for Jitter/Shimmer ---
        # Create PointProcess from pitch track (needed for voice report)
        # Use recommended time range for analysis (e.g., 0.01s to avoid edge effects)
        # Max number of candidates = 15 can improve accuracy
        point_process = call(pitch, "To PointProcess (cc)", 15)  # Add max_candidates

        # --- Voice Report (Jitter, Shimmer) ---
        # Define time range for analysis (adjust start/end if needed, 0, 0 uses full range)
        # Silence threshold, voicing threshold from Praat defaults
        # Octave cost, octave jump cost, voiced/unvoiced cost from Praat defaults
        jitter_shimmer_report = call(
            [sound, point_process, pitch],
            "Voice report",
            0.0,
            0.0,
            pitch_floor,
            pitch_ceiling,
            1.3,
            1.6,
            0.03,
            0.45,
        )

        # --- Extract values from the report string ---
        # Note: Praat report format might vary slightly. Check output if errors occur.
        # Parsing the string report is fragile; be careful with indices/splitting.
        lines = jitter_shimmer_report.strip().split("\n")
        values = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = (
                    parts[1].strip().split(" ")[0]
                )  # Take the numeric part before units
                try:
                    # Handle Praat's '--undefined--' output
                    values[key] = (
                        float(val_str) if val_str != "--undefined--" else np.nan
                    )
                except ValueError:
                    values[key] = np.nan  # Assign NaN if conversion fails

        # Map parsed values to feature names
        praat_features["jitter_local"] = values.get("Jitter (local)", np.nan)
        praat_features["jitter_local_abs"] = values.get(
            "Jitter (local, absolute)", np.nan
        )
        praat_features["jitter_rap"] = values.get("Jitter (rap)", np.nan)
        praat_features["jitter_ppq5"] = values.get("Jitter (ppq5)", np.nan)

        praat_features["shimmer_local"] = values.get("Shimmer (local)", np.nan)
        praat_features["shimmer_local_db"] = values.get("Shimmer (local, dB)", np.nan)
        praat_features["shimmer_apq3"] = values.get("Shimmer (apq3)", np.nan)
        praat_features["shimmer_apq5"] = values.get("Shimmer (apq5)", np.nan)
        praat_features["shimmer_apq11"] = values.get("Shimmer (apq11)", np.nan)

        # Convert percentages to absolute values if needed (Praat often reports % for local Jitter/Shimmer)
        if not np.isnan(praat_features["jitter_local"]):
            praat_features["jitter_local"] /= 100.0
        if not np.isnan(praat_features["shimmer_local"]):
            praat_features["shimmer_local"] /= 100.0

    except parselmouth.PraatError as e:
        filename = sound.name if hasattr(sound, "name") else "Unknown"
        print(
            f"⚠️ PraatError processing {filename}: {e}. Storing NaNs for Praat features."
        )
        # Keep NaNs assigned initially
    except Exception as e:
        filename = sound.name if hasattr(sound, "name") else "Unknown"
        print(f"❌ Unexpected error during Praat processing for {filename}: {e}")
        # Keep NaNs assigned initially

    return praat_features


def extract_librosa_features(y, sr):
    """
    Extracts frame-wise features using Librosa.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sample rate.

    Returns:
        dict: Dictionary containing frame-wise librosa features. Returns None if fails.
    """
    librosa_features = {}
    try:
        # Ensure audio is not empty
        if len(y) == 0:
            print(
                "⚠️ Warning: Librosa received empty audio array. Skipping librosa features."
            )
            return None

        # Extract Librosa Features (same as before)
        librosa_features["mfcc"] = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC
        )
        librosa_features["rms"] = librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )[0]
        librosa_features["spec_cent"] = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        librosa_features["spec_bw"] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        librosa_features["spec_contrast"] = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_flatness"] = librosa.feature.spectral_flatness(
            y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        librosa_features["spec_rolloff"] = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        librosa_features["zcr"] = librosa.feature.zero_crossing_rate(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )[0]

        # Librosa Pitch (PYIN)
        f0_librosa, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=FMIN_PITCH_LIBROSA,
            fmax=FMAX_PITCH_LIBROSA,
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        librosa_features["pitch_librosa"] = np.nan_to_num(f0_librosa, nan=0.0)

        # --- Consistency Check for frame-wise features ---
        try:
            num_frames = librosa_features["mfcc"].shape[1]
        except IndexError:
            print(
                "⚠️ Warning: Could not determine number of frames from MFCCs. Skipping consistency check."
            )
            return librosa_features

        for key, value in librosa_features.items():
            if value is None:
                continue  # Skip if feature extraction failed earlier
            # Ensure value is a numpy array before checking ndim
            if not isinstance(value, np.ndarray):
                continue

            if value.ndim == 1:
                current_len = len(value)
                if current_len != num_frames:
                    pad_width = num_frames - current_len
                    if pad_width > 0:
                        librosa_features[key] = np.pad(
                            value, (0, pad_width), mode="edge"
                        )
                    else:
                        librosa_features[key] = value[:num_frames]
            elif value.ndim == 2:
                current_len = value.shape[1]
                if current_len != num_frames:
                    pad_width = num_frames - current_len
                    if pad_width > 0:
                        librosa_features[key] = np.pad(
                            value, ((0, 0), (0, pad_width)), mode="edge"
                        )
                    else:
                        librosa_features[key] = value[:, :num_frames]

        return librosa_features

    except Exception as e:
        print(f"❌ Error during Librosa feature extraction: {e}")
        # traceback.print_exc() # Uncomment locally for debugging
        return None


# %%
def extract_features(audio_path):
    """
    Loads an audio file and extracts features using both Librosa and Parselmouth.

    Args:
        audio_path (str): Path to the preprocessed audio file.

    Returns:
        dict: A dictionary containing feature names as keys and numpy arrays/scalars
              as values. Returns None if loading or critical processing fails.
    """
    all_features = {}
    basename = os.path.basename(audio_path)

    try:
        # --- Load with Librosa (for librosa features) ---
        print(f"Attempting to load with Librosa: {basename}")
        y, sr = librosa.load(audio_path, sr=SR, duration=None)
        print(f"Successfully loaded with Librosa: {basename}")

        # --- Extract Librosa Features ---
        librosa_features = extract_librosa_features(y, sr)
        if librosa_features:
            all_features.update(librosa_features)
            print(f"✅ Extracted Librosa features for {basename}")
        else:
            print(f"⚠️ Failed to extract Librosa features for {basename}")
            # Decide if you want to continue without librosa features or return None
            # return None # Option: Fail entirely if librosa part fails

        # --- Load with Parselmouth (for Praat features) ---
        print(f"Attempting to load with Parselmouth: {basename}")
        sound = parselmouth.Sound(audio_path)
        sound.name = basename  # Assign name for better error messages
        print(f"Successfully loaded with Parselmouth: {basename}")

        # --- Extract Praat Features ---
        praat_features = extract_praat_features(
            sound, PRAAT_PITCH_FLOOR, PRAAT_PITCH_CEILING
        )
        all_features.update(praat_features)  # Add Praat features (even if NaNs)
        if not all(np.isnan(v) for v in praat_features.values()):
            print(f"✅ Extracted Praat features (Jitter/Shimmer/F0) for {basename}")
        else:
            print(f"⚠️ Failed to extract valid Praat features for {basename}")

        # Check if any features were successfully extracted
        if not all_features:
            print(f"🛑 No features could be extracted for {basename}")
            return None

        return all_features

    except FileNotFoundError:
        print(f"❌ Error: File not found at {audio_path}")
        return None
    except parselmouth.PraatError as e:
        print(f"❌ PraatError loading/processing {basename}: {e}")
        return None  # Fail if fundamental Praat loading fails
    except Exception as e:
        print(f"❌ Unexpected Error processing {basename}: {e}")
        # traceback.print_exc() # Uncomment locally for detailed errors
        return None


# %%
def save_features(features, output_path):
    """Saves the extracted features dictionary to a compressed .npz file."""
    try:
        np.savez_compressed(output_path, **features)
        # print(f"💾 Saved features to {output_path}")
    except Exception as e:
        print(f"❌ Error saving features to {output_path}: {e}")


# %%
def process_feature_wrapper(preprocessed_filename):
    """Wrapper function for multiprocessing."""
    input_path = os.path.join(PREPROCESSED_FOLDER, preprocessed_filename)
    base_name = os.path.splitext(preprocessed_filename)[0]
    original_name = base_name.replace("_preprocessed", "")  # Adjust if needed
    output_filename = f"{original_name}_features.npz"
    output_path = os.path.join(FEATURE_FOLDER, output_filename)

    # Caching Check
    if os.path.exists(output_path):
        print(
            f"⏩ Skipping {preprocessed_filename} (features already exist at {output_path})"
        )
        return

    print(f"\nProcessing {preprocessed_filename} for feature extraction...")
    features = extract_features(input_path)

    if features is not None and features:
        save_features(features, output_path)
    else:
        print(
            f"🛑 Feature extraction failed or returned no features for {preprocessed_filename}. Skipping save."
        )


# %%
def extract_features_for_all(input_folder, output_folder):
    """Processes all compatible audio files using multiprocessing."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Ensured output directory exists: {output_folder}")
    except OSError as e:
        print(f"❌ Error creating output directory {output_folder}: {e}")
        return

    try:
        print(f"Looking for preprocessed files in: {input_folder}")
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input directory not found: {input_folder}")

        preprocessed_files = [
            f
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
            and f.lower().endswith((".wav", ".mp3", ".flac"))
        ]
        print(f"Found {len(preprocessed_files)} potential files.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error listing files in {input_folder}: {e}")
        return

    if not preprocessed_files:
        print(f"ℹ️ No compatible audio files found in {input_folder} to process.")
        return

    print(f"Found {len(preprocessed_files)} preprocessed files to process.")
    num_processes = max(1, cpu_count() // 2)
    print(f"🚀 Starting feature extraction with {num_processes} processes...")

    try:
        with Pool(processes=num_processes) as pool:
            pool.map(process_feature_wrapper, preprocessed_files)
    except Exception as e:
        print(f"❌ Error during multiprocessing: {e}")

    print("\n✅ Feature extraction process completed.")


# %%
# --- Main Execution Guard ---
if __name__ == "__main__":
    print("📊 Starting Step 2: Feature Extraction (including Jitter/Shimmer)...")

    if not os.path.isdir(PREPROCESSED_FOLDER):
        print(f"❌ Error: Preprocessed input folder not found: {PREPROCESSED_FOLDER}")
        print("Please ensure Step 1 has run and generated files in that location.")
    else:
        # Ensure FEATURE_FOLDER exists before multiprocessing starts
        if not os.path.isdir(FEATURE_FOLDER):
            try:
                os.makedirs(FEATURE_FOLDER, exist_ok=True)
                print(f"Created feature output directory: {FEATURE_FOLDER}")
            except OSError as e:
                print(
                    f"❌ Failed to create feature output directory {FEATURE_FOLDER}: {e}"
                )
                # Decide if you want to exit if the output dir can't be created
                # exit() # Or handle differently

        # Only proceed if input exists and output (or can be created)
        if os.path.isdir(FEATURE_FOLDER):
            extract_features_for_all(PREPROCESSED_FOLDER, FEATURE_FOLDER)

    print("🏁 Step 2 Finished.")
