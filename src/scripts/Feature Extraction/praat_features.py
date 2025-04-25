# from config import *
#
#
# def extract_praat_features(sound, pitch_floor=75.0, pitch_ceiling=600.0):
#     """
#     Extracts scalar Jitter, Shimmer, and median F0 using Parselmouth/Praat.
#
#     Args:
#         sound (parselmouth.Sound): Parselmouth sound object.
#         pitch_floor (float): Minimum pitch frequency for analysis.
#         pitch_ceiling (float): Maximum pitch frequency for analysis.
#
#     Returns:
#         dict: Dictionary containing scalar Praat features. Returns NaNs on failure.
#     """
#     praat_features = {
#         "f0_median_praat": np.nan,
#         "jitter_local": np.nan,
#         "jitter_local_abs": np.nan,
#         "jitter_rap": np.nan,
#         "jitter_ppq5": np.nan,
#         "shimmer_local": np.nan,
#         "shimmer_local_db": np.nan,
#         "shimmer_apq3": np.nan,
#         "shimmer_apq5": np.nan,
#         "shimmer_apq11": np.nan,
#     }
#
#     # Skip processing for very short sounds
#     if sound.get_total_duration() < 0.1:  # Skip if less than 100ms
#         filename = sound.name if hasattr(sound, "name") else "Unknown"
#         print(f"⚠️ Sound too short for Praat analysis: {filename}")
#         return praat_features
#
#     try:
#         # --- Pitch ---
#         # Time step: 0.0 = auto
#         pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
#
#         # Check if pitch object has any valid candidates
#         num_frames = call(pitch, "Get number of frames")
#         if num_frames <= 0:
#             filename = sound.name if hasattr(sound, "name") else "Unknown"
#             print(f"⚠️ No valid pitch frames found in {filename}")
#             return praat_features
#
#         praat_features["f0_median_praat"] = call(
#             pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"
#         )  # Get median pitch
#
#         # --- PointProcess for Jitter/Shimmer ---
#         point_process = call(pitch, "To PointProcess (cc)", 15)
#
#         # Check if enough points for analysis
#         num_points = call(point_process, "Get number of points")
#         if num_points < 5:  # Minimum needed for meaningful jitter/shimmer
#             filename = sound.name if hasattr(sound, "name") else "Unknown"
#             print(f"⚠️ Too few points ({num_points}) for jitter/shimmer analysis in {filename}")
#             return praat_features
#
#         # --- Voice Report (Jitter, Shimmer) ---
#         jitter_shimmer_report = call(
#             [sound, point_process, pitch],
#             "Voice report",
#             0.0,
#             0.0,
#             pitch_floor,
#             pitch_ceiling,
#             1.3,
#             1.6,
#             0.03,
#             0.45,
#         )
#
#         # --- Extract values from the report string ---
#         lines = jitter_shimmer_report.strip().split("\n")
#         values = {}
#         for line in lines:
#             parts = line.split(":")
#             if len(parts) == 2:
#                 key = parts[0].strip()
#                 val_str = parts[1].strip().split(" ")[0]  # Take the numeric part before units
#                 try:
#                     # Handle Praat's '--undefined--' output
#                     values[key] = float(val_str) if val_str != "--undefined--" else np.nan
#                 except ValueError:
#                     values[key] = np.nan  # Assign NaN if conversion fails
#
#         # Map parsed values to feature names
#         feature_mappings = {
#             "Jitter (local)": "jitter_local",
#             "Jitter (local, absolute)": "jitter_local_abs",
#             "Jitter (rap)": "jitter_rap",
#             "Jitter (ppq5)": "jitter_ppq5",
#             "Shimmer (local)": "shimmer_local",
#             "Shimmer (local, dB)": "shimmer_local_db",
#             "Shimmer (apq3)": "shimmer_apq3",
#             "Shimmer (apq5)": "shimmer_apq5",
#             "Shimmer (apq11)": "shimmer_apq11",
#         }
#
#         for praat_key, feature_key in feature_mappings.items():
#             if praat_key in values:
#                 praat_features[feature_key] = values[praat_key]
#
#         # Convert percentages to absolute values if needed
#         if not np.isnan(praat_features["jitter_local"]):
#             praat_features["jitter_local"] /= 100.0
#         if not np.isnan(praat_features["shimmer_local"]):
#             praat_features["shimmer_local"] /= 100.0
#
#         # Validate features are in reasonable ranges
#         for key in ["jitter_local", "jitter_rap", "jitter_ppq5"]:
#             if not np.isnan(praat_features[key]) and praat_features[key] > 0.1:  # Jitter > 10% is suspicious
#                 praat_features[key] = np.nan
#
#         for key in ["shimmer_local"]:
#             if not np.isnan(praat_features[key]) and praat_features[key] > 0.3:  # Shimmer > 30% is suspicious
#                 praat_features[key] = np.nan
#
#     except parselmouth.PraatError as e:
#         filename = sound.name if hasattr(sound, "name") else "Unknown"
#         print(f"⚠️ PraatError processing {filename}: {e}. Storing NaNs for Praat features.")
#     except Exception as e:
#         filename = sound.name if hasattr(sound, "name") else "Unknown"
#         print(f"❌ Unexpected error during Praat processing for {filename}: {e}")
#
#     return praat_features