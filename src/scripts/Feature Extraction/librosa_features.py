from config import *


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