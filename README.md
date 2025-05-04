Here’s your updated README with everything related to Docker Hub removed:

---

# Speaker Gender and Age Recognition

A Dockerized deep learning project for predicting speaker gender and age from audio files.

## 🚀 Getting Started

Follow the steps below to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/speaker-gender-age-recognition.git
cd speaker-gender-age-recognition
```

### 2. Prepare Your Data

Place your test audio files inside the `data` folder in the project root directory.

### 3. Build the Docker Image

```bash
docker build -t speaker-gender-age-recognition .
```

### 4. Run the Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -it speaker-gender-age-recognition
```

> ✅ This will mount your local `data` and `output` directories into the container.

### 5. Check the Results

After the container runs, check the `output` folder. It will contain:

* `results.txt` — predicted gender and age for each input.
* `time.txt` — time taken for processing.

---

## 🗃 Directory Structure

```
.
├── data/            # Input audio files
├── output/          # Output files from the container
└── README.md
```

---