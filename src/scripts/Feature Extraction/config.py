# import os
# import time
# import numpy as np
# import librosa  # pip install librosa
# import parselmouth  # pip install praat-parselmouth
# from parselmouth.praat import call  # To call Praat functions
# from multiprocessing import Pool, cpu_count, Manager, Value
# import traceback  # For detailed error logging
# import warnings  # To suppress specific warnings if needed
# import gc  # For garbage collection
# from datetime import datetime, timedelta
# from tqdm import tqdm  # For progress bar visualization
# import sys  # For console printing
#
# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
#
# # --- Configuration ---
# # Consider moving these to a config file for easier management
# PREPROCESSED_FOLDER = "../data"  # Input for Step 2 (Output of Step 1)
# FEATURE_FOLDER = ".features/"  # Output for Step 2
#
# # Librosa Feature Extraction Parameters
# SR = 16000  # Target Sample Rate for librosa features
# N_FFT = 2048  # FFT window size
# HOP_LENGTH = 512  # Hop length for STFT
# N_MELS = 128  # Number of Mel bands
# N_MFCC = 20  # Number of MFCCs
# FMIN_PITCH_LIBROSA = librosa.note_to_hz("C2")  # Min frequency for librosa pitch
# FMAX_PITCH_LIBROSA = librosa.note_to_hz("C7")  # Max frequency for librosa pitch
#
# # Praat Pitch Parameters
# PRAAT_PITCH_FLOOR = 75.0  # Pitch floor (Hz) for Praat analysis
# PRAAT_PITCH_CEILING = 600.0  # Pitch ceiling (Hz) for Praat analysis
#
# # Processing parameters
# MAX_WORKERS = max(1, cpu_count() - 1)  # Leave one core free
# PRINT_INTERVAL = 5  # Print progress update every X seconds