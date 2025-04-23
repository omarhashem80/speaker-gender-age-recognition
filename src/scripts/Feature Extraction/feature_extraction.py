from praat_features import *
from librosa_features import extract_librosa_features
from time_tracker import TimeTracker


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