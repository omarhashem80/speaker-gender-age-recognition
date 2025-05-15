import os
import noisereduce as nr
import numpy as np
import webrtcvad
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import split_on_silence
from multiprocessing import Pool, cpu_count
import pandas as pd

INPUT_FOLDER = "data\\raw"
OUTPUT_FOLDER = "data\\processed"


def load_audio(file_path):
    """Load an audio file using pydub"""
    return AudioSegment.from_file(file_path)


def reduce_noise(audio_segment):
    """Apply noise reduction using noisereduce"""
    samples = np.array(audio_segment.get_array_of_samples())
    reduced_samples = nr.reduce_noise(y=samples, sr=audio_segment.frame_rate)
    reduced_audio = AudioSegment(
        reduced_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels,
    )
    return reduced_audio


def remove_silence(
    audio_segment, min_silence_len=500, silence_thresh=None, keep_silence=200
):
    """Remove long silences from the audio"""
    if silence_thresh is None:
        silence_thresh = audio_segment.dBFS - 16  # Auto calculate if not provided

    chunks = split_on_silence(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    if not chunks:
        print(
            "Warning: No chunks found after silence removal. Returning original audio."
        )
        return audio_segment

    cleaned_audio = AudioSegment.empty()
    for chunk in chunks:
        cleaned_audio += chunk

    return cleaned_audio


def frame_audio(samples, sample_rate, frame_duration_ms=30, hop_duration_ms=10):
    """
    Split audio into frames with optional overlap (hop)

    Args:
        samples: numpy array of audio samples
        sample_rate: audio sample rate in Hz
        frame_duration_ms: frame length in milliseconds
        hop_duration_ms: hop length in milliseconds

    Returns:
        List of (frame, start_sample, end_sample) tuples
    """
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    hop_length = int(sample_rate * hop_duration_ms / 1000)

    frames = []
    for i in range(0, len(samples) - frame_length + 1, hop_length):
        frame = samples[i : i + frame_length]
        frames.append((frame, i, i + frame_length))

    return frames


def apply_webrtc_vad(
    audio_segment, aggressiveness=2, frame_duration_ms=30, hop_duration_ms=10
):
    """
    Enhanced VAD processing with proper framing and overlap
    Returns a new AudioSegment with only VAD-approved speech segments
    """
    # Convert to VAD-compatible format
    audio = audio_segment.set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)

    # Frame the audio with overlap
    frames = frame_audio(samples, sample_rate, frame_duration_ms, hop_duration_ms)

    # Process frames and mark speech regions
    speech_mask = np.zeros(len(samples), dtype=bool)
    for frame, start, end in frames:
        frame_bytes = frame.astype(np.int16).tobytes()
        if vad.is_speech(frame_bytes, sample_rate):
            speech_mask[start:end] = True

    # Apply hang-over to prevent clipping of speech ends
    speech_mask = smooth_speech_mask(speech_mask, sample_rate, min_speech_gap_ms=200)

    # Extract only speech segments
    speech_samples = samples[speech_mask]

    if len(speech_samples) == 0:
        print("Warning: No speech detected by VAD. Returning empty segment.")
        return AudioSegment.silent(duration=0, frame_rate=sample_rate)

    return AudioSegment(
        speech_samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )


def smooth_speech_mask(mask, sample_rate, min_speech_gap_ms=200):
    """
    Apply temporal smoothing to speech mask to:
    1. Merge nearby speech segments
    2. Remove very short speech segments
    3. Prevent abrupt cuts
    """
    min_gap_samples = int(sample_rate * min_speech_gap_ms / 1000)

    # Find transitions between speech and non-speech
    changes = np.diff(mask.astype(int))
    speech_starts = np.where(changes == 1)[0]
    speech_ends = np.where(changes == -1)[0]

    # Handle edge cases
    if mask[0]:
        speech_starts = np.insert(speech_starts, 0, 0)
    if mask[-1]:
        speech_ends = np.append(speech_ends, len(mask) - 1)

    # Merge nearby segments
    new_mask = np.copy(mask)
    for i in range(len(speech_ends) - 1):
        gap = speech_starts[i + 1] - speech_ends[i]
        if gap < min_gap_samples:
            new_mask[speech_ends[i] : speech_starts[i + 1]] = True

    return new_mask


def normalize_audio(audio_segment):
    """Normalize the audio to boost volume properly"""
    return normalize(audio_segment)


def save_audio(audio_segment, output_path):
    """Save the audio to the specified path"""
    audio_segment.export(output_path, format="mp3")


def process_audio_file(input_path, output_path):
    """Full pipeline: Load -> Denoise -> Silence Removal -> VAD -> Normalize -> Save"""
    print(f"\nProcessing {input_path}...")

    # 1. Load original audio
    audio = load_audio(input_path)
    print(f"Loaded | Duration: {len(audio) / 1000:.2f}s | dBFS: {audio.dBFS:.2f}")

    # 2. Noise reduction
    denoised_audio = reduce_noise(audio)
    print(
        f"Denoised | Duration: {len(denoised_audio) / 1000:.2f}s | dBFS: {denoised_audio.dBFS:.2f}"
    )

    # 3. Silence removal (coarse)
    silence_removed_audio = remove_silence(denoised_audio)
    print(
        f"Silence removed | Duration: {len(silence_removed_audio) / 1000:.2f}s | dBFS: {silence_removed_audio.dBFS:.2f}"
    )

    # 4. WebRTC VAD with proper framing
    vad_processed_audio = apply_webrtc_vad(
        silence_removed_audio,
        aggressiveness=2,
        frame_duration_ms=30,
        hop_duration_ms=10,
    )
    print(
        f"VAD processed | Duration: {len(vad_processed_audio) / 1000:.2f}s | dBFS: {vad_processed_audio.dBFS:.2f}"
    )

    # 5. Normalization
    if len(vad_processed_audio) > 0:
        normalized_audio = normalize_audio(vad_processed_audio)
        print(
            f"Normalized | Duration: {len(normalized_audio) / 1000:.2f}s | dBFS: {normalized_audio.dBFS:.2f}"
        )
        save_audio(normalized_audio, output_path)
    else:
        print("No audio remaining after processing. Skipping save.")

    print(f"Saved processed file to {output_path}")


def process_wrapper(args):
    filename, input_folder, output_folder = args
    extension = filename.split(".")[-1]

    if extension in ["mp3", "wav"]:
        input_path = os.path.join(input_folder, filename)
        output_filename = (
            filename.replace(".mp3", "_preprocessed.mp3")
            if filename.endswith(".mp3")
            else filename.replace(".wav", "_preprocessed.wav")
        )
        output_path = os.path.join(output_folder, output_filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already preprocessed)")
            return output_path

        process_audio_file(input_path, output_path)
        return output_path


def process_all_files(input_folder, output_folder) -> pd.DataFrame:
    """Process all mp3 files in the input folder using maximum concurrency"""
    os.makedirs(output_folder, exist_ok=True)

    audio_files = [
        f for f in os.listdir(input_folder) if f.endswith(".mp3") or f.endswith(".wav")
    ]
    args = [(filename, input_folder, output_folder) for filename in audio_files]
    max_processes = max(2, cpu_count() - 2)

    with Pool(processes=max_processes) as pool:
        output_paths = pool.map(process_wrapper, args)
    return pd.DataFrame({"path": output_paths})


if __name__ == "__main__":
    print("🔊 Starting audio preprocessing...")
    process_all_files(INPUT_FOLDER, OUTPUT_FOLDER)
