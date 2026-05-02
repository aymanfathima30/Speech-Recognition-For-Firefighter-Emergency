# 🔥 Speech Recognition System for Firefighters

**Group Project** | Python · PyTorch · scikit-learn · NLP · CNN

A speech recognition pipeline engineered for **high-noise emergency environments**, where communication accuracy can be the difference between life and death. Built specifically for fire service radio transmissions — cutting through SCBA breathing apparatus noise, radio static, and stress-distorted speech under physical exertion.

---

## About the Project

### The Problem

Standard commercial speech recognition systems — including cloud-based ASR services — are trained on clean, studio-quality audio. They fail significantly in fire emergency environments due to:

- **SCBA (Self-Contained Breathing Apparatus) noise** — the constant hiss of breathing equipment overlaid on every transmission
- **Radio static and transmission artefacts** — clipping, dropout, and compression noise from analogue radios
- **Stressed and distorted speech** — firefighters operating under extreme physical exertion speak faster, louder, and less clearly than normal
- **Time-critical context** — a misrecognised "mayday" or missed "structural collapse" call has direct life-safety consequences

Existing solutions either require expensive specialist hardware or rely on human dispatchers to manually triage all incoming radio traffic, creating bottlenecks during large-scale incidents.

### Our Solution

We built a purpose-designed pipeline with two key components addressing these challenges directly:

**1. Robust preprocessing** — rather than passing raw audio to a recogniser, transmissions first go through spectral subtraction noise reduction, peak normalisation, and silence trimming. This cleans the audio to a standard the CNN can reliably process regardless of radio quality.

**2. NLP urgency detection** — transcribed text is not just logged; it is semantically analysed and classified into four urgency levels using a weighted combination of domain keyword matching and TF-IDF cosine similarity. This means even if the transcription is imperfect, the urgency score is still reliable because it draws on multiple signal types.

### Technology Choices

| Technology | Why it was chosen |
|---|---|
| **CNN on mel spectrograms** | Treats audio as a 2D image problem — convolutional filters learn spectro-temporal patterns that are inherently robust to noise and pitch variation |
| **CTC decoding** | Handles variable-length transcriptions without needing forced alignment between audio frames and output characters |
| **TF-IDF + cosine similarity** | Lightweight semantic matching — no large language model required, runs in real time on standard hardware |
| **scikit-learn SVM** | Fast, interpretable classifier for the urgency levels with good performance on small, domain-specific datasets |
| **Python + PyTorch** | Rapid prototyping, extensive audio/ML ecosystem, and broad deployment compatibility |

### Academic Context

This project was developed as part of a group university project exploring the application of NLP and deep learning to safety-critical communication systems. The urgency detection module and preprocessing pipeline were co-designed and evaluated against a custom dataset of simulated fire service radio transmissions.

---

## Use Cases

| Scenario | How the system helps |
|---|---|
| **Mayday call in a burning building** | Detects `"firefighter down"` or `"mayday"` instantly, alerts the Rapid Intervention Team and Incident Commander within seconds |
| **Noisy fire ground radio traffic** | Preprocessing strips SCBA noise and static before transcription — significantly improving accuracy over off-the-shelf ASR systems |
| **Dispatcher triage under heavy comms** | Urgency classifier auto-prioritises transmissions so dispatchers focus on Level 3 CRITICAL calls first |
| **Post-incident review** | Every transmission is timestamped and logged with urgency scores for debriefs and investigations |
| **Training exercises** | Can be run on recorded exercise audio to evaluate communication clarity and response protocols |

---

## System Architecture

```
Raw Audio Input (radio transmission .wav)
        │
        ▼
┌──────────────────────────────────────┐
│          AudioPreprocessor           │
│  • Spectral subtraction denoising    │
│  • Peak normalisation to [-1, 1]     │
│  • Silence trimming                  │
│  • Log-Mel spectrogram (128 × T)     │
└──────────────┬───────────────────────┘
               │  Mel Spectrogram
               ▼
┌──────────────────────────────────────┐
│          CNN Speech Model            │
│  • Conv Block 1 — 32 filters (3×3)   │
│  • Conv Block 2 — 64 filters (3×3)   │
│  • Conv Block 3 — 128 filters (3×3)  │
│  • Global Avg Pool → Dense 256       │
│  • CTC output → transcription text   │
└──────────────┬───────────────────────┘
               │  Transcribed Text
               ▼
┌──────────────────────────────────────┐
│          Urgency Detector (NLP)      │
│  • Stage 1: keyword matching (0.6×)  │
│  • Stage 2: TF-IDF cosine sim (0.4×) │
│  • Output: Level 0–3 urgency score   │
└──────────────┬───────────────────────┘
               │  Urgency Level
               ▼
┌──────────────────────────────────────┐
│            Dispatcher                │
│  Routes alert to the correct units:  │
│  RIT · EMS · Sector Commander · IC   │
└──────────────────────────────────────┘
```

---

## Demo — Sample Output

**CRITICAL transmission (Level 3):**
```
$ python main.py --audio samples/mayday_call.wav

Processing: samples/mayday_call.wav
✔ Preprocessing complete
✔ Transcription : "mayday mayday firefighter trapped no air" (confidence: 0.91)
✔ Urgency       : CRITICAL (Level 3)

=== TRANSMISSION REPORT ===
Text     : mayday mayday firefighter trapped no air
Urgency  : CRITICAL (Level 3)
Action   : IMMEDIATE MAYDAY RESPONSE — Alert all units and incident commander
Notified : Incident Commander, Rapid Intervention Team, EMS
Time     : 2024-03-15T14:32:07Z
```

**ROUTINE transmission (Level 0):**
```
$ python main.py --audio samples/status_check.wav

✔ Transcription : "all units standing by scene secure" (confidence: 0.88)
✔ Urgency       : ROUTINE (Level 0)

Action   : ROUTINE LOG — Record transmission in incident log
Notified : (none)
```

---

## CNN — Why Convolutional Layers?

The CNN processes audio as **log-mel spectrograms** — 2D images where the x-axis is time and y-axis is frequency. This lets convolutional filters learn spectro-temporal patterns the same way image CNNs learn visual edges and shapes:

| Layer | Filters | What it learns |
|---|---|---|
| Conv Block 1 | 32 | Low-level edges — onsets, transients |
| Conv Block 2 | 64 | Phoneme-level frequency patterns |
| Conv Block 3 | 128 | Word-level spectro-temporal signatures |

MaxPooling after each block provides **spatial invariance**, keeping the model robust to pitch and tempo shifts caused by stress or equipment distortion. The final **CTC head** handles variable-length output without forced alignment between audio frames and characters.

---

## Urgency Detection — Semantic Analysis

The `UrgencyDetector` classifies transcribed text into four levels using a **two-stage approach**:

**Stage 1 — Domain keyword matching** (60% weight)

| Level | Label | Example triggers |
|---|---|---|
| 3 | CRITICAL | `"mayday"`, `"firefighter down"`, `"structural collapse"`, `"no egress"` |
| 2 | HIGH | `"flashover"`, `"victim located"`, `"rapid fire spread"`, `"second alarm"` |
| 1 | MODERATE | `"water supply needed"`, `"equipment malfunction"`, `"staging update"` |
| 0 | ROUTINE | `"all clear"`, `"standing by"`, `"scene secure"`, `"en route"` |

**Stage 2 — TF-IDF cosine semantic similarity** (40% weight)

Each urgency level has an anchor sentence. Input text is vectorised with bigram TF-IDF and compared against each anchor via cosine distance — catching paraphrased urgency that keyword lists alone would miss.

```
final_score = 0.6 × normalised_keyword_score + 0.4 × cosine_similarity
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Process a single audio file

```bash
python main.py --audio samples/transmission_001.wav
```

### 3. Batch process a folder of recordings

```bash
python main.py --batch samples/
```

### 4. Use a specific trained model checkpoint

```bash
python main.py --audio samples/transmission_001.wav --model models/cnn_speech_v2.pt
```

### 5. Run urgency detection on text directly (skip ASR)

```bash
python main.py --text "mayday firefighter trapped no air second floor"
```

### 6. Run the test suite

```bash
pytest tests/
```

---

## Project Structure

```
speech-recognition-firefighters/
├── main.py                     # Entry point + CLI
├── pipeline/
│   ├── preprocessor.py         # Denoising, normalisation, mel spectrogram
│   ├── cnn_model.py            # CNN architecture + CTC decoding
│   ├── urgency_detector.py     # NLP urgency classification
│   └── dispatcher.py          # Alert routing by urgency level
├── models/
│   └── cnn_speech_v2.pt        # Trained model weights
├── samples/                    # Example .wav transmissions for testing
├── tests/
│   └── test_urgency.py
├── requirements.txt
└── README.md
```

---

## Contributors

Group project — co-developed the audio preprocessing pipeline, CNN architecture, and NLP urgency detection module.
