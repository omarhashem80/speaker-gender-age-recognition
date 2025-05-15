# 🎙️ Speaker Gender & Age Recognition

A machine learning system that analyzes voice recordings to predict a speaker’s **gender** and **age group**.
It processes audio files, extracts detailed acoustic features, and classifies each speaker into one of four predefined categories with a focus on accuracy and robustness.

---

## 🧠 Classification Task

The system predicts one of the following labels for each input audio file:

| Label | Category         |
| :---: | ---------------- |
|   0   | Male, Twenties   |
|   1   | Female, Twenties |
|   2   | Male, Fifties    |
|   3   | Female, Fifties  |

---

## 🧼 Preprocessing Pipeline

Every audio file is passed through a multi-step cleaning and preparation process before feature extraction:

### 🔧 Steps

1. **Loading**
   Reads audio in `.mp3` or `.wav` formats using `pydub`.

2. **Noise Reduction**
   Removes background noise using spectral subtraction via `noisereduce`.

3. **Silence Removal**
   Trims long silences based on dB thresholds using `split_on_silence`.

4. **Voice Activity Detection (VAD)**
   Keeps only voiced segments using `webrtcvad` with overlapping frames and smoothing.

5. **Normalization**
   Boosts volume to standardize loudness using `pydub.effects.normalize`.

🗂️ Output files are saved to the `data/processed/` folder with `_preprocessed` suffixes.

---

## 🎛️ Feature Extraction

The system uses a diverse set of handcrafted features designed for speech analysis:

| Feature Type          | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| **MFCCs + Deltas**    | 20 coefficients with Δ and Δ² for dynamic speech features     |
| **Pitch Statistics**  | Mean, Std, Min, Max of fundamental frequency (F₀)             |
| **Formants**          | Mean of F1, F2, F3 using Praat-based analysis (`parselmouth`) |
| **Spectral Features** | Centroid, Bandwidth, Contrast, Rolloff                        |
| **Spectral Contrast** | Multi-band contrast with adjustable frequency boundaries      |
| **Chroma CQT**        | Harmonic pitch class features from constant-Q transform       |
| **Tonnetz**           | Tonal centroid features representing harmonic relations       |
| **Voice Quality**     | Jitter, Shimmer, Harmonics-to-Noise Ratio (HNR)               |
| **ZCR**               | Zero-Crossing Rate (temporal noisiness indicator)             |

All features are aggregated using **mean** and **standard deviation** for consistency.

---

## 🧮 Modeling Approach

The classifier is based on **XGBoost**, a robust gradient-boosted decision tree algorithm.

### Key Techniques:

* **Feature Selection**: `SelectKBest` with ANOVA F-score
* **Cross-Validation**: 5-fold stratified sampling
* **Hyperparameter Tuning**:

  * Grid search with cross-validation
  * Optuna for advanced optimization
* **Sample Weighting**: Class balance handled via `compute_sample_weight`

📦 Final model is saved as `classifier.joblib` and automatically loaded during inference.

---

## 🚀 How to Run Locally with Docker

Follow these steps to test the system locally:

### 1. Clone the Repository

```bash
git clone https://github.com/omarhashem80/speaker-gender-age-recognition.git
cd speaker-gender-age-recognition
```

### 2. Prepare Your Data

Place your test audio files inside the `data/` folder at the root of the project.

### 3. Build the Docker Image

```bash
docker build -t nn_p .
```

### 4. Run the Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -it nn_p
```

> This mounts your local `data` and `output` folders inside the container for input and output respectively.

---

## 📤 Output Format

The system writes two files to the `output/` directory:

| File          | Description                                              |
| ------------- | -------------------------------------------------------- |
| `results.txt` | One predicted label (0–3) per line for each audio file   |
| `time.txt`    | Processing time in seconds for each audio file (rounded) |

> ⚠️ The order of predictions matches the numerical order of filenames.
> No extra text, headers, or IDs should be included.

---

## 🗂️ Project Structure

```
.
├── data/                     # Input test audio files
├── output/                   # Model predictions and timings
├── src/
│   ├── DataCleaning/         # Audio noise reduction & silence trimming
│   ├── FeatureExtraction/    # Feature computation modules
│   ├── Modeling/             # Model training and hyperparameter tuning
│   ├── Performance/          # Evaluation metrics and timing
│   ├── Deployment/           # cloud deployment client
│   └── Preprocess/           # Full audio preprocessing pipeline
│   ├── main.py               # Complete pipeline entry point
│   └── main.ipynb            # Jupyter notebook version for dev
├── infer.py                  # Inference script for batch prediction
├── external_infer.py         # External API deployment test client
├── Dockerfile                # Runtime environment specification
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (you’re here)
```

---

## ☁️ Deployment

The model is deployed on  **Hugging Face Spaces** and can be accessed via HTTP API

```python
response = requests.post("https://omarhashem80-age-gender-classifier.hf.space/predict", json={"data": feature_dict})
```

---

## 🔐 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---
