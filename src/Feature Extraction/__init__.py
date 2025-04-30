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