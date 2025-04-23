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

import os
import time
import numpy as np
import librosa  # pip install librosa
import parselmouth  # pip install praat-parselmouth
from parselmouth.praat import call  # To call Praat functions
from multiprocessing import Pool, cpu_count, Manager, Value
import traceback  # For detailed error logging
import warnings  # To suppress specific warnings if needed
import gc  # For garbage collection
from datetime import datetime, timedelta
from tqdm import tqdm  # For progress bar visualization
import sys  # For console printing

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# --- Configuration ---
# Consider moving these to a config file for easier management
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

# Praat Pitch Parameters
PRAAT_PITCH_FLOOR = 75.0  # Pitch floor (Hz) for Praat analysis
PRAAT_PITCH_CEILING = 600.0  # Pitch ceiling (Hz) for Praat analysis

# Processing parameters
MAX_WORKERS = max(1, cpu_count() - 1)  # Leave one core free
PRINT_INTERVAL = 5  # Print progress update every X seconds


class TimeTracker:
    """Class to track processing time and provide estimates"""
    
    def __init__(self, total_files):
        self.start_time = time.time()
        self.total_files = total_files
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self.file_times = []  # Store processing times for better estimates
        self.last_print_time = time.time()
        
    def update(self, status, processing_time=None):
        """Update tracker with a new file status"""
        if status == 'completed':
            self.completed += 1
            if processing_time:
                self.file_times.append(processing_time)
        elif status == 'skipped':
            self.skipped += 1
        elif status == 'failed':
            self.failed += 1
        
    def get_progress_str(self):
        """Get a formatted progress string with ETA"""
        elapsed = time.time() - self.start_time
        total_processed = self.completed + self.skipped + self.failed
        progress = total_processed / self.total_files if self.total_files > 0 else 0
        
        # Calculate ETA based on average of recent processing times
        if len(self.file_times) > 0 and self.completed > 0:
            # Use up to the last 10 files for a more accurate recent average
            recent_times = self.file_times[-min(10, len(self.file_times)):]
            avg_time_per_file = sum(recent_times) / len(recent_times)
            files_remaining = self.total_files - total_processed
            eta_seconds = avg_time_per_file * files_remaining
        else:
            # Fall back to simple estimation if we don't have timing data
            if progress > 0 and elapsed > 0:
                eta_seconds = (elapsed / progress) - elapsed
            else:
                eta_seconds = 0
        
        # Format ETA string
        if eta_seconds > 0:
            eta_td = timedelta(seconds=int(eta_seconds))
            if eta_td.days > 0:
                eta_str = f"{eta_td.days}d {eta_td.seconds//3600}h {(eta_td.seconds//60)%60}m"
            elif eta_td.seconds > 3600:
                eta_str = f"{eta_td.seconds//3600}h {(eta_td.seconds//60)%60}m {eta_td.seconds%60}s"
            elif eta_td.seconds > 60:
                eta_str = f"{eta_td.seconds//60}m {eta_td.seconds%60}s"
            else:
                eta_str = f"{eta_td.seconds}s"
        else:
            eta_str = "calculating..."
            
        # Format elapsed time string
        elapsed_td = timedelta(seconds=int(elapsed))
        if elapsed_td.days > 0:
            elapsed_str = f"{elapsed_td.days}d {elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m"
        elif elapsed_td.seconds > 3600:
            elapsed_str = f"{elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m {elapsed_td.seconds%60}s"
        elif elapsed_td.seconds > 60:
            elapsed_str = f"{elapsed_td.seconds//60}m {elapsed_td.seconds%60}s"
        else:
            elapsed_str = f"{elapsed_td.seconds}s"
        
        # Build the progress string
        progress_str = (f"[{total_processed}/{self.total_files}] {progress:.1%} "
                        f"(✅{self.completed} ⏩{self.skipped} ❌{self.failed}) "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                        
        # If we have enough data, add throughput
        if elapsed > 60 and self.completed > 0:  # Minimum 1 minute elapsed
            files_per_minute = (self.completed / elapsed) * 60
            progress_str += f" | Rate: {files_per_minute:.1f} files/min"
            
        return progress_str
    
    def should_print_update(self):
        """Check if we should print a progress update"""
        current_time = time.time()
        if current_time - self.last_print_time >= PRINT_INTERVAL:
            self.last_print_time = current_time
            return True
        return False
        
    def get_final_stats(self):
        """Get final statistics string"""
        elapsed = time.time() - self.start_time
        elapsed_td = timedelta(seconds=int(elapsed))
        
        # Format elapsed time
        if elapsed_td.days > 0:
            elapsed_str = f"{elapsed_td.days}d {elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m"
        elif elapsed_td.seconds > 3600:
            elapsed_str = f"{elapsed_td.seconds//3600}h {(elapsed_td.seconds//60)%60}m {elapsed_td.seconds%60}s"
        elif elapsed_td.seconds > 60:
            elapsed_str = f"{elapsed_td.seconds//60}m {elapsed_td.seconds%60}s"
        else:
            elapsed_str = f"{elapsed_td.seconds}s"
            
        # Calculate average time per file
        if self.completed > 0:
            avg_time = sum(self.file_times) / len(self.file_times)
            avg_time_str = f"{avg_time:.2f}s"
        else:
            avg_time_str = "N/A"
            
        # Build the stats string
        stats = [
            f"Total time: {elapsed_str}",
            f"Completed:  {self.completed} files",
            f"Skipped:    {self.skipped} files",
            f"Failed:     {self.failed} files",
            f"Total:      {self.total_files} files",
            f"Avg. time:  {avg_time_str} per file"
        ]
        
        if elapsed > 60 and self.completed > 0:
            files_per_minute = (self.completed / elapsed) * 60
            stats.append(f"Throughput:  {files_per_minute:.1f} files/min")
            
        return "\n".join(stats)


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
    
    # Skip processing for very short sounds
    if sound.get_total_duration() < 0.1:  # Skip if less than 100ms
        filename = sound.name if hasattr(sound, "name") else "Unknown"
        print(f"⚠️ Sound too short for Praat analysis: {filename}")
        return praat_features
        
    try:
        # --- Pitch ---
        # Time step: 0.0 = auto
        pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
        
        # Check if pitch object has any valid candidates
        num_frames = call(pitch, "Get number of frames")
        if num_frames <= 0:
            filename = sound.name if hasattr(sound, "name") else "Unknown"
            print(f"⚠️ No valid pitch frames found in {filename}")
            return praat_features
            
        praat_features["f0_median_praat"] = call(
            pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"
        )  # Get median pitch

        # --- PointProcess for Jitter/Shimmer ---
        point_process = call(pitch, "To PointProcess (cc)", 15)
        
        # Check if enough points for analysis
        num_points = call(point_process, "Get number of points")
        if num_points < 5:  # Minimum needed for meaningful jitter/shimmer
            filename = sound.name if hasattr(sound, "name") else "Unknown"
            print(f"⚠️ Too few points ({num_points}) for jitter/shimmer analysis in {filename}")
            return praat_features

        # --- Voice Report (Jitter, Shimmer) ---
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
        lines = jitter_shimmer_report.strip().split("\n")
        values = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = parts[1].strip().split(" ")[0]  # Take the numeric part before units
                try:
                    # Handle Praat's '--undefined--' output
                    values[key] = float(val_str) if val_str != "--undefined--" else np.nan
                except ValueError:
                    values[key] = np.nan  # Assign NaN if conversion fails

        # Map parsed values to feature names
        feature_mappings = {
            "Jitter (local)": "jitter_local",
            "Jitter (local, absolute)": "jitter_local_abs",
            "Jitter (rap)": "jitter_rap",
            "Jitter (ppq5)": "jitter_ppq5",
            "Shimmer (local)": "shimmer_local",
            "Shimmer (local, dB)": "shimmer_local_db",
            "Shimmer (apq3)": "shimmer_apq3",
            "Shimmer (apq5)": "shimmer_apq5",
            "Shimmer (apq11)": "shimmer_apq11",
        }
        
        for praat_key, feature_key in feature_mappings.items():
            if praat_key in values:
                praat_features[feature_key] = values[praat_key]

        # Convert percentages to absolute values if needed
        if not np.isnan(praat_features["jitter_local"]):
            praat_features["jitter_local"] /= 100.0
        if not np.isnan(praat_features["shimmer_local"]):
            praat_features["shimmer_local"] /= 100.0

        # Validate features are in reasonable ranges
        for key in ["jitter_local", "jitter_rap", "jitter_ppq5"]:
            if not np.isnan(praat_features[key]) and praat_features[key] > 0.1:  # Jitter > 10% is suspicious
                praat_features[key] = np.nan
                
        for key in ["shimmer_local"]:
            if not np.isnan(praat_features[key]) and praat_features[key] > 0.3:  # Shimmer > 30% is suspicious
                praat_features[key] = np.nan

    except parselmouth.PraatError as e:
        filename = sound.name if hasattr(sound, "name") else "Unknown"
        print(f"⚠️ PraatError processing {filename}: {e}. Storing NaNs for Praat features.")
    except Exception as e:
        filename = sound.name if hasattr(sound, "name") else "Unknown"
        print(f"❌ Unexpected error during Praat processing for {filename}: {e}")

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
        # Ensure audio is not empty or too short
        if len(y) < sr / 10:  # Less than 100ms
            print("⚠️ Warning: Audio too short for meaningful features")
            return None

        # Extract Librosa Features
        librosa_features["mfcc"] = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC
        )
        librosa_features["rms"] = librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_cent"] = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_bw"] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_contrast"] = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_flatness"] = librosa.feature.spectral_flatness(
            y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["spec_rolloff"] = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        librosa_features["zcr"] = librosa.feature.zero_crossing_rate(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )

        # Librosa Pitch (PYIN) - This can be computationally expensive
        f0_librosa, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=FMIN_PITCH_LIBROSA,
            fmax=FMAX_PITCH_LIBROSA,
            sr=sr,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        librosa_features["pitch_librosa"] = np.nan_to_num(f0_librosa, nan=0.0)
        
        # Store voiced/unvoiced information
        librosa_features["voiced_flag"] = voiced_flag

        # --- Consistency Check for frame-wise features ---
        try:
            # Find maximum number of frames across all features
            max_frames = max(
                val.shape[1] if val.ndim > 1 else len(val)
                for val in librosa_features.values()
                if isinstance(val, np.ndarray)
            )
            
            # Standardize all features to this frame count
            for key, value in librosa_features.items():
                if not isinstance(value, np.ndarray):
                    continue
                    
                if value.ndim == 1:
                    current_len = len(value)
                    if current_len != max_frames:
                        if current_len < max_frames:
                            # Pad shorter arrays
                            librosa_features[key] = np.pad(
                                value, (0, max_frames - current_len), mode="edge"
                            )
                        else:
                            # Truncate longer arrays
                            librosa_features[key] = value[:max_frames]
                            
                elif value.ndim == 2:
                    current_len = value.shape[1]
                    if current_len != max_frames:
                        if current_len < max_frames:
                            # Pad shorter arrays
                            librosa_features[key] = np.pad(
                                value, ((0, 0), (0, max_frames - current_len)), mode="edge"
                            )
                        else:
                            # Truncate longer arrays
                            librosa_features[key] = value[:, :max_frames]

        except ValueError as e:
            print(f"⚠️ Warning during frame consistency check: {e}")
        
        # Basic validation - check for NaNs or infinite values
        has_issues = False
        for key, value in librosa_features.items():
            if isinstance(value, np.ndarray) and (np.isnan(value).any() or np.isinf(value).any()):
                print(f"⚠️ Found NaN or Inf values in {key}, replacing with zeros")
                librosa_features[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                has_issues = True
                
        if has_issues:
            print("⚠️ Fixed some numerical issues in features")

        return librosa_features

    except MemoryError:
        print("❌ Memory error during Librosa feature extraction - file may be too large")
        return None
    except Exception as e:
        print(f"❌ Error during Librosa feature extraction: {e}")
        return None


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
        try:
            y, sr = librosa.load(audio_path, sr=SR, duration=None)
            
            # Check signal validity
            if len(y) == 0:
                print(f"⚠️ Empty audio data in {basename}")
                return None
                
            if np.isnan(y).any() or np.isinf(y).any():
                print(f"⚠️ Found NaN or Inf values in audio data for {basename}")
                y = np.nan_to_num(y)
                
            # Extract Librosa Features
            librosa_features = extract_librosa_features(y, sr)
            if librosa_features:
                all_features.update(librosa_features)
                print(f"✅ Extracted Librosa features for {basename}")
            else:
                print(f"⚠️ Failed to extract Librosa features for {basename}")
        except Exception as e:
            print(f"❌ Error with Librosa processing for {basename}: {e}")
        
        # --- Load with Parselmouth (for Praat features) ---
        try:
            sound = parselmouth.Sound(audio_path)
            sound.name = basename  # Assign name for better error messages
            
            # Extract Praat Features
            praat_features = extract_praat_features(
                sound, PRAAT_PITCH_FLOOR, PRAAT_PITCH_CEILING
            )
            all_features.update(praat_features)  # Add Praat features (even if NaNs)
            
            if not all(np.isnan(v) for v in praat_features.values()):
                print(f"✅ Extracted Praat features for {basename}")
            else:
                print(f"⚠️ Failed to extract valid Praat features for {basename}")
                
            # Clean up parselmouth object (may help with memory)
            del sound
            
        except parselmouth.PraatError as e:
            print(f"❌ PraatError loading/processing {basename}: {e}")
        except Exception as e:
            print(f"❌ Error with Praat processing for {basename}: {e}")

        # Check if any features were successfully extracted
        if not all_features:
            print(f"🛑 No features could be extracted for {basename}")
            return None

        # Force garbage collection to free memory
        gc.collect()
        
        return all_features

    except FileNotFoundError:
        print(f"❌ Error: File not found at {audio_path}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error processing {basename}: {e}")
        return None


def save_features(features, output_path):
    """Saves the extracted features dictionary to a compressed .npz file."""
    try:
        # Check feature validity before saving
        for key, value in features.items():
            if isinstance(value, np.ndarray) and (np.isnan(value).any() or np.isinf(value).any()):
                print(f"⚠️ Warning: Feature '{key}' contains NaN or Inf values - replacing with zeros")
                features[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.savez_compressed(output_path, **features)
        return True
    except Exception as e:
        print(f"❌ Error saving features to {output_path}: {e}")
        return False


def process_feature_wrapper(args):
    """Wrapper function for multiprocessing."""
    preprocessed_filename, time_tracker, progress_lock = args
    
    input_path = os.path.join(PREPROCESSED_FOLDER, preprocessed_filename)
    base_name = os.path.splitext(preprocessed_filename)[0]
    original_name = base_name.replace("_preprocessed", "")  # Adjust if needed
    output_filename = f"{original_name}_features.npz"
    output_path = os.path.join(FEATURE_FOLDER, output_filename)

    # Caching Check
    if os.path.exists(output_path):
        with progress_lock:
            time_tracker.update('skipped')
            # Only print progress update periodically to avoid terminal spam
            if time_tracker.should_print_update():
                print(f"\r⏩ {time_tracker.get_progress_str()}", end="", flush=True)
        return

    # Process file
    file_start_time = time.time()
    
    # Extract features
    features = extract_features(input_path)
    processing_time = time.time() - file_start_time

    # Handle results
    with progress_lock:
        if features is not None and features:
            if save_features(features, output_path):
                time_tracker.update('completed', processing_time)
                print(f"\n✅ Processed {preprocessed_filename} in {processing_time:.1f}s")
            else:
                time_tracker.update('failed')
                print(f"\n❌ Failed to save features for {preprocessed_filename}")
        else:
            time_tracker.update('failed')
            print(f"\n🛑 Feature extraction failed for {preprocessed_filename}")
            
        # Print progress update periodically
        if time_tracker.should_print_update():
            print(f"\r📊 {time_tracker.get_progress_str()}", end="", flush=True)


def extract_features_for_all(input_folder, output_folder):
    """Processes all compatible audio files using multiprocessing with comprehensive time tracking."""
    # Ensure output directory exists
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Ensured output directory exists: {output_folder}")
    except OSError as e:
        print(f"❌ Error creating output directory {output_folder}: {e}")
        return

    # Find all input files
    try:
        print(f"Looking for preprocessed files in: {input_folder}")
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input directory not found: {input_folder}")

        # Get all audio files
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        preprocessed_files = [
            f for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f)) and 
            f.lower().endswith(audio_extensions)
        ]
        
        # Validate files before processing
        valid_files = []
        for f in preprocessed_files:
            full_path = os.path.join(input_folder, f)
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                print(f"⚠️ Skipping zero-sized file: {f}")
                continue
            valid_files.append(f)
            
        preprocessed_files = valid_files
        print(f"Found {len(preprocessed_files)} valid audio files.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error listing files in {input_folder}: {e}")
        return

    if not preprocessed_files:
        print(f"ℹ️ No compatible audio files found in {input_folder} to process.")
        return

    # Setup multiprocessing and time tracking
    manager = Manager()
    progress_lock = manager.Lock()
    time_tracker = TimeTracker(len(preprocessed_files))
    
    # Create process arguments with progress tracking info
    process_args = [(filename, time_tracker, progress_lock) for filename in preprocessed_files]
    
    # Determine optimal number of workers
    num_processes = min(MAX_WORKERS, len(preprocessed_files))
    print(f"🚀 Starting feature extraction with {num_processes} processes...")
    print(f"📊 Advanced progress tracking enabled - will show completion rate and ETA")
    print(f"🔄 Processing {len(preprocessed_files)} files...")

    try:
        # Process files with multiprocessing
        with Pool(processes=num_processes) as pool:
            pool.map(process_feature_wrapper, process_args)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Partial results may be available.")
    except Exception as e:
        print(f"\n\n❌ Error during multiprocessing: {e}")
        traceback.print_exc()

    # Final stats
    print(f"\n\n📊 Feature Extraction Statistics:")
    print("="*50)
    print(time_tracker.get_final_stats())
    print("="*50)
    
    # Validate results
    processed_features = [f for f in os.listdir(output_folder) if f.endswith("_features.npz")]
    success_rate = len(processed_features) / len(preprocessed_files) if preprocessed_files else 0
    print(f"\nFound {len(processed_features)} feature files in output directory")
    print(f"Overall success rate: {success_rate:.1%}")
    
    print(f"Feature extraction completed. Check {output_folder} for results.")


if __name__ == "__main__":
    # Run the feature extraction process
    extract_features_for_all(PREPROCESSED_FOLDER, FEATURE_FOLDER)
    print("Feature extraction script completed.")
