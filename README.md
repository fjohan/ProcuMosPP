# Prosodic Cues and Modeling Strategies in Swedish Prominence Prediction

## Quickstart (Inference First)

If you only want to run inference with the included example files, use this section.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas librosa scikit-learn scipy matplotlib transformers tqdm
```

Notes:

- On first run, the Hugging Face model `KBLab/wav2vec2-large-voxrex-swedish` may be downloaded automatically.
- This project has only been tested on Linux.
- Inference works on CPU, but is significantly faster with a GPU-enabled PyTorch install.

### 2. Run inference with provided example data

Using word-level timestamps (recommended):

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --csv example_data/seg_006.csv \
  --out_csv example_data/seg_006_pred.csv \
  --praat
```

Another example:

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_023.wav \
  --csv example_data/seg_023.csv \
  --out_csv example_data/seg_023_pred.csv
```

### 3. Output files

- Prediction CSV, e.g. `example_data/seg_006_pred.csv`
- With `--praat`:
  - `<prefix>_pred.TextGrid`
  - `<prefix>_prom.Sound`

### 4. Sliding-window mode (no CSV)

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --interval 0.4 \
  --overlap 0.1 \
  --out_csv example_data/seg_006_windows_pred.csv
```

See "6.5 Caveat: Fixed-Interval Inference and Silence" before using this mode on long recordings.

Tip (Praat workflow):

- Open the original `.wav`, the produced `*_prom.Sound`, and the produced `*_pred.TextGrid` in Praat.
- Resample `*_prom.Sound` with sampling frequency `16000 Hz` and precision `1`.
- Combine the original wav and the resampled prominence sound into stereo.
- View the stereo sound together with the TextGrid.
- Optionally mute channel 2 while inspecting alignment.

## 1. Introduction
This study investigates the prediction of word-level prominence in Swedish news speech. The goal was to predict continuous prominence ratings (scale 0–2) derived from mass crowdsourcing (20+ raters per file). We compared two pre-trained Wav2Vec 2.0 backbones—one generic and one language-specific—across three levels of architectural complexity to determine the optimal configuration for small-data prosody modeling.

## 2. Methodology

### 2.1 Dataset
The dataset consists of approximately 130 audio files (total duration ~20 minutes) featuring 5 speakers (3 male, 2 female) reading news in a homogenous Swedish dialect. Labels are mean per-word prominence ratings. Training was performed using Leave-One-Speaker-Out (LOSO) cross-validation to ensure speaker independence.

### 2.2 Backbone Models
We compared two pre-trained feature extractors:
1.  **W2V2-Base:** `facebook/wav2vec2-base-960h` (Generic/English, 768-dim). A standard baseline.
2.  **VoxRex-Large:** `KBLab/wav2vec2-large-voxrex-swedish` (Swedish-specific, 1024-dim). Trained specifically on Swedish corpora (SR, SVT, audiobooks).

### 2.3 Experimental Configurations
We evaluated three incremental configurations:

*   **Config 1: Bare (Baseline)**
    *   **Pooling:** Simple Mean Pooling over the word's duration.
    *   **Loss:** Standard Mean Squared Error (MSE).
    *   **Input:** Frozen W2V2 embeddings + Log Duration.
*   **Config 2: AWM (Architectural Enhancements)**
    *   **A (Attention):** Learned Attention Pooling to focus on the syllabic nucleus rather than averaging silence/consonants.
    *   **W (Weighted Loss):** Custom loss function weighing high-prominence targets ($y>1.0$) 5x higher than non-prominent ones to combat "regression to the mean."
    *   **M (Max Pooling):** Hybrid pooling concatenating the *Max* activation with the *Attention* vector to capture peak intensity.
*   **Config 3: PiSh (Pitch Shapes & Scalars)**
    *   **Includes all AWM features.**
    *   **Explicit Prosody:** Injection of 7 scalar features per word into the LSTM:
        *   *Pitch Shape:* 2nd-degree polynomial coefficients (Curvature, Slope, Height) + Residual Error to capture rises/falls/peaks.
        *   *Stats:* Log Duration, RMS Mean, RMS Max, Spectral Centroid.

## 3. Results
Results are reported as the mean Pearson Correlation ($r$) and Mean Squared Error (MSE) across 3 random seeds (30–35 epochs).

| Model Backbone | Configuration | Correlation ($r$) | MSE | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **VoxRex (Swedish)** | **Bare** | 0.7288 | 0.0394 | Strong baseline due to language fit. |
| | **AWM** | **0.7957** | 0.0331 | **Major improvement (+0.07)**. Attention unlocks the model's potential. |
| | **PiSh** | **0.7987** | **0.0311** | Minimal $r$ gain, but **lowest MSE**. Reduced variance. |
| **W2V2 (Generic)** | **Bare** | 0.6877 | 0.0438 | Struggles with Swedish prosody alignment. |
| | **AWM** | 0.7046 | 0.0440 | Moderate improvement. |
| | **PiSh** | 0.7238 | 0.0416 | **Significant gain**. Explicit features compensate for lack of language knowledge. |

## 4. Discussion

### 4.1 Language Specificity dominates
The Swedish-specific **VoxRex** model consistently outperformed the generic W2V2 model by a margin of $r \approx 0.04 - 0.07$. This confirms that for prosody tasks on small datasets, using a backbone pre-trained on the target language is the single most effective design choice. VoxRex likely encodes Swedish tonal word accents (Accent I/II) implicitly, whereas W2V2 does not.

### 4.2 Architecture unlocks representations (The "AWM" Jump)
For VoxRex, the jump from **Bare** (0.72) to **AWM** (0.79) is dramatic.
*   *Mean pooling* acts as a low-pass filter, smoothing out the sharp peaks that characterize prominence.
*   *Attention* and *Max Pooling* allow the model to pinpoint the stressed vowel and peak energy, which aligns better with human perception of prominence.
*   *Weighted Loss* successfully forced the model to predict values $>1.0$, reducing the "safe bet" under-prediction seen in early baselines.

### 4.3 Explicit Features: Calibration vs. Detection
Adding **Pitch Shapes (PiSh)** had different effects on the two models:
*   **For VoxRex:** The correlation saturated at $\sim0.80$ (likely near the ceiling of human inter-rater agreement). Adding pitch shapes didn't change the *ranking* ($r$) much, but it significantly improved the *magnitude* (MSE). This suggests VoxRex already knew *where* the prominence was, but the explicit scalars helped it calibrate exactly *how high* the rating should be.
*   **For W2V2:** Adding pitch shapes improved both $r$ and MSE. Since the generic model lacks deep knowledge of Swedish prosodic structure, providing explicit contour information (rises/falls) acted as a crucial crutch, helping it close the gap.

## 5. Conclusion
We have established a robust pipeline for Swedish prominence prediction. The optimal configuration uses a **language-specific transformer (VoxRex)** combined with **Attention/Max pooling** and **Weighted Loss**. While explicit **Pitch Shape** features offer diminishing returns for correlation on the best model, they provide the most stable and accurate amplitude predictions (lowest MSE), making them valuable for high-precision applications.

## 6. Running the Code

This repository contains two runnable scripts:

- `prompred_train.py`: training and LOSO evaluation
- `prompred_infer.py`: inference on new audio using a saved checkpoint

### 6.1 Environment Setup

This project requires PyTorch and audio/ML dependencies. In an environment without `torch`, both scripts will fail at import time.

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas librosa scikit-learn scipy matplotlib transformers tqdm
```

### 6.2 Expected Training Data Layout

`prompred_train.py` expects a `data/` directory in the project root.  
Inside `data/`, each speaker gets a subfolder. Each `.wav` must have a matching `.csv` with the same basename.

Expected structure:

```text
data/
  spk1/
    file001.wav
    file001.csv
    file002.wav
    file002.csv
  spk2/
    ...
  spk3/
    ...
```

Expected training CSV format (no header):

```text
start,end,word,rating
0.12,0.41,det,0.35
0.41,0.88,huset,1.42
```

Notes:

- `start` and `end` are in seconds.
- `rating` is the target prominence value (continuous).
- If `data/` does not exist or contains no valid pairs, training exits with `No data found.`

### 6.3 Train Script (`prompred_train.py`)

Run LOSO evaluation (default mode):

```bash
python prompred_train.py --mode loso
```

Run LOSO with a single seed:

```bash
python prompred_train.py --mode loso --seed 42
```

Train final model on all speakers and save checkpoint(s) to `models/`:

```bash
python prompred_train.py --mode all
```

Single-seed full training:

```bash
python prompred_train.py --mode all --seed 142857
```

Outputs:

- feature cache in `cache/` (auto-created)
- plots in `plots/` (during LOSO)
- model checkpoint(s) in `models/`, e.g. `models/prom_model_full_seed142857.pt`

### 6.4 Inference Script (`prompred_infer.py`)

Required inputs:

- `--checkpoint`: trained `.pt` checkpoint (for example from `models/`)
- `--wav`: input wav file

Optional:

- `--csv`: segments/tokens file
- `--interval`, `--overlap`: sliding window settings if no CSV is provided
- `--praat`: write Praat outputs

Included example files:

- `example_data/seg_006.wav`
- `example_data/seg_006.csv`
- `example_data/seg_023.wav`
- `example_data/seg_023.csv`

Example with word-level CSV (recommended):

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --csv example_data/seg_006.csv \
  --out_csv example_data/seg_006_pred.csv \
  --praat
```

Another CSV-based example:

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_023.wav \
  --csv example_data/seg_023.csv \
  --out_csv example_data/seg_023_pred.csv
```

Example with automatic sliding windows (no CSV):

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --interval 0.4 \
  --overlap 0.1 \
  --out_csv example_data/seg_006_windows_pred.csv
```

Inference CSV accepted formats:

- Header: `start,end,word,rating`
- Header: `start,end,word` (rating optional)
- Header: `start_time,end_time,word[,rating]`
- No header with 3 or 4 columns in the same order

Inference outputs:

- prediction CSV (default name: `<wavbase>_pred.csv`)
- if `--praat` is used:
  - `<prefix>_pred.TextGrid`
  - `<prefix>_prom.Sound`

### 6.5 Caveat: Fixed-Interval Inference and Silence

`⚠ NB: Inference on Fixed Intervals and Non-Speech Regions`

When running inference using fixed-length sliding windows (that is, without word timestamps), the model may produce unexpectedly high prominence values during pauses or non-speech regions.

This occurs because:

- The model was trained exclusively on segments containing speech (word-level units).
- Feature normalization (for example, log duration and RMS energy) can cause silent intervals to resemble low-energy speech rather than true silence.
- The model has never learned an explicit non-speech class.

As a result, silence, background noise, or filled pauses may receive non-zero or even high prominence predictions when using interval-based inference.

Recommendations:

- Prefer inference with word-level timestamps when available.
- If using sliding windows, consider:
  - Adding a simple energy threshold to suppress predictions during silence.
  - Running a Voice Activity Detection (VAD) step before inference.
  - Post-processing the prominence curve to zero out low-energy regions.

This limitation does not affect inference when using word-aligned CSV input.
