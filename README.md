# Prosodic Cues and Modeling Strategies in Swedish Prominence Prediction

## Contents

This repo contains a model for prominence prediction and scripts to (a) perform inference and (b) train such a model. The model is based on five speakers and has been evaluated using a Leave-One-Speaker-Out (LOSO) cross-validation paradigm (see "2. Methodology"), but the model provided here is trained on all five speakers.

## Quickstart (Inference First)

To run inference with the included example files, use this section.

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

### 3. Output files

- Prediction CSV, e.g. `example_data/seg_006_pred.csv`
- With `--praat`:
  - `<prefix>_pred.TextGrid`
  - `<prefix>_prom.Sound`

Tip (Praat workflow):

- Open the original `.wav`, the produced `*_prom.Sound`, and the produced `*_pred.TextGrid` in Praat.
- Resample `*_prom.Sound` with a sampling frequency of `16000 Hz` and precision `1`.
- Combine the original `.wav` and the resampled prominence sound into stereo.
- View the stereo sound together with the TextGrid.
- Optionally mute channel 2 while inspecting results.

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

## Hugging Face Space App

This repo includes a CPU-first Gradio app entrypoint for Hugging Face Spaces:

```bash
python app.py
```

The app supports:

- uploading a sound file
- optionally uploading a CSV or Praat TextGrid with word timings
- inference with word segments or fixed sliding windows
- minimal RMS-based silence suppression for sliding-window mode
- output as a prediction table, enriched CSV download, prominence curve, and word timeline

The app runs prominence prediction only. It does not run Whisper or produce transcripts.

For TextGrid uploads, the app uses an `IntervalTier` named `word` or `words` when present. If no such tier exists, it uses the first non-empty `IntervalTier`.

By default, the app loads:

```text
models/prom_model_full_seed142857.pt
```

For a Space where the ProcuMosPP checkpoint is stored in a separate Hugging Face model repo, set:

```bash
PROCUMOSPP_MODEL_REPO=your-name/your-procumospp-model-repo
PROCUMOSPP_MODEL_FILENAME=prom_model_full_seed142857.pt
```

or set `PROCUMOSPP_CHECKPOINT` to a local checkpoint path. The Wav2Vec2 backbone is loaded from `KBLab/wav2vec2-large-voxrex-swedish`.

CPU Spaces can be slow on first run because the Wav2Vec2 backbone is large. The app limits uploads to 60 seconds by default; change this with:

```bash
PROCUMOSPP_MAX_AUDIO_SECONDS=120
```

## 1. Introduction
This repo is the result of a study that investigates word-level prominence prediction in Swedish news speech. The goal was to predict continuous prominence ratings (scale 0-2) derived from mass crowdsourcing (20+ raters per file). We compared two pre-trained Wav2Vec 2.0 backbones, one generic and one language-specific, across three levels of architectural complexity to determine the optimal configuration for small-data prosody modeling.

## 2. Methodology

### 2.1 Dataset
The dataset consists of approximately 130 audio files (total duration ~20 minutes) featuring 5 speakers (3 male, 2 female) reading news in a homogeneous Swedish dialect. Labels are mean per-word prominence ratings. Training was performed using Leave-One-Speaker-Out (LOSO) cross-validation to ensure speaker independence.

### 2.2 Backbone Models
We compared two pre-trained feature extractors:
1.  **W2V2-Base:** `facebook/wav2vec2-base-960h` (Generic/English, 768-dim). A standard baseline.
2.  **VoxRex-Large:** `KBLab/wav2vec2-large-voxrex-swedish` (Swedish-specific, 1024-dim). Trained specifically on Swedish corpora (SR, SVT, audiobooks).

### 2.3 Experimental Configurations
We evaluated three incremental SSL-backed configurations, plus a PiSh-only scalar ablation:

*   **Config 1: Bare (Baseline)**
    *   **Pooling:** Simple Mean Pooling over the word's duration.
    *   **Loss:** Standard Mean Squared Error (MSE).
    *   **Input:** Frozen W2V2 embeddings + Log Duration.
*   **Config 2: AWM (Architectural Enhancements)**
    *   **A (Attention):** Learned attention pooling that can focus on the most informative frames within each word, instead of uniformly averaging all frames.
    *   **W (Weighted Loss):** Target-dependent weighted MSE, $w = 1 + 2y$ (for $y \ge 0$), which upweights higher-prominence targets (up to ~5x at $y=2$) to combat "regression to the mean."
    *   **M (Max Pooling):** Hybrid pooling concatenating the *Max* activation with the *Attention* vector to capture peak intensity.
*   **Config 3: PiSh (Pitch Shapes & Scalars)**
    *   **Includes all AWM features.**
    *   **Explicit Prosody:** Adding 8 scalar features per word:
        *   *Pitch Shape:* 2nd-degree polynomial coefficients (Curvature, Slope, Height) + Residual Error to capture rises/falls/peaks.
        *   *Stats:* Log Duration, RMS Mean, RMS Max, Spectral Centroid.
*   **Scalar Ablation: PiSh-only (`--no_ssl`)**
    *   **Input:** The same 8 explicit PiSh scalar features, without Wav2Vec/VoxRex frame embeddings.
    *   **Architecture:** Scalar sequence model (`PiSh scalars -> BiLSTM -> output head`), bypassing SSL frame extraction, attention pooling, and max pooling.
    *   **Purpose:** Tests how much prominence can be predicted from explicit prosodic cues alone.

## 3. Results
Results are reported as the mean Pearson Correlation ($r$) and Mean Squared Error (MSE) across 5 random seeds (30–35 epochs).

| Model Backbone | Configuration | Correlation ($r$) | MSE | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **VoxRex (Swedish)** | **Bare** | 0.7288 | 0.0394 | Strong baseline due to language fit. |
| | **AWM** | **0.7957** | 0.0331 | **Major improvement (+0.07)**. Attention unlocks the model's potential. |
| | **PiSh** | **0.7987** | **0.0311** | Minimal $r$ gain, but **lowest MSE**. Reduced variance. |
| **W2V2 (Generic)** | **Bare** | 0.6877 | 0.0438 | No language-specific knowledge. |
| | **AWM** | 0.7046 | 0.0440 | Moderate improvement. |
| | **PiSh** | 0.7238 | 0.0416 | **Significant gain**. Explicit features compensate for lack of language knowledge. |
| **No SSL backbone** | **PiSh-only (`--no_ssl`)** | 0.6622 | 0.0525 | Explicit prosodic features alone are informative, but substantially below SSL-backed models. |

## 4. Discussion

### 4.1 Language Specificity dominates
The Swedish-specific **VoxRex** model consistently outperformed the generic W2V2 model by a margin of $r \approx 0.04 - 0.07$. This confirms that for prosody tasks on small datasets, using a backbone pre-trained on the target language is the single most effective design choice. VoxRex likely encodes Swedish tonal word accents (Accent I/II) implicitly, whereas W2V2 does not.

### 4.2 Architecture unlocks representations (The "AWM" Jump)
For VoxRex, the jump from **Bare** (0.72) to **AWM** (0.79) is dramatic.
*   *Mean pooling* acts as a low-pass filter, smoothing out the sharp peaks that characterize prominence.
*   *Attention* and *Max Pooling* let the model emphasize informative and high-activation frames within each word, which better preserves prominence-related cues than mean pooling alone.
*   *Weighted Loss* increased the penalty for underestimating higher-prominence targets, reducing the "safe bet" under-prediction seen in early baselines.

### 4.3 Explicit Features: Calibration vs. Detection
Adding **Pitch Shapes (PiSh)** had different effects on the two models:
*   **For VoxRex:** The correlation saturated at $\sim0.80$ (likely near the ceiling of human inter-rater agreement). Adding pitch shapes didn't change the *ranking* ($r$) much, but it significantly improved the *magnitude* (MSE). This suggests VoxRex already knew *where* the prominence was, but the explicit scalars helped it calibrate exactly *how high* the rating should be.
*   **For W2V2:** Adding pitch shapes improved both $r$ and MSE. Since the generic model lacks deep knowledge of Swedish prosodic structure, providing explicit contour information (rises/falls) acted as a crucial crutch, helping it close the gap.

## 5. Conclusion
We have established a robust pipeline for Swedish prominence prediction. The optimal configuration uses a **language-specific transformer (VoxRex)** combined with **Attention/Max pooling** and **Weighted Loss**. While explicit **Pitch Shape** features offer diminishing returns for correlation on the best model, they provide the most stable and accurate amplitude predictions (lowest MSE), making them valuable for future exploration.

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
Inside `data/`, each speaker gets a subfolder. Each `.wav` must have a matching `.csv` with the same basename. The `.csv` must be headerless.

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

Expected training CSV format (no header, but columns are start, end, word, rating):

```text
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

Run a PiSh-only scalar baseline without Wav2Vec/VoxRex embeddings:

```bash
python prompred_train.py --mode loso --no_ssl
```

This keeps the explicit prosody scalar features and sequence model, but bypasses SSL frame extraction, attention pooling, and max pooling.

Run LOSO with a single seed:

```bash
python prompred_train.py --mode loso --seed 42
```

Run LOSO as a classifier instead of continuous regression:

```bash
# Binary: class 0 = no prominence, class 1 = prominent
python prompred_train.py --mode loso --target binary

# Ternary: class 0 = no prominence, class 1 = maybe prominent, class 2 = prominent
python prompred_train.py --mode loso --target ternary
```

By default, class boundaries are learned from the training labels with 1D k-means:

```bash
python prompred_train.py --mode loso --target ternary --class_boundary_method kmeans
```

A Gaussian mixture model can also be used to learn probabilistic 1D rating clusters:

```bash
python prompred_train.py --mode loso --target ternary --class_boundary_method gmm
```

In LOSO mode, those boundaries are fit separately inside each fold using only the training speakers, then applied to the held-out speaker. This avoids leaking held-out speaker label distributions into the class definition. For fixed conventional boundaries, use rounding-style thresholds instead:

```bash
python prompred_train.py --mode loso --target ternary --class_boundary_method round
```

`round` uses threshold `0.5` for binary classification and thresholds `0.5,1.5` for ternary classification.

Mean-rating thresholds can also be optimized inside each LOSO training fold:

```bash
python prompred_train.py --mode loso --target ternary --class_boundary_method opt_macro_f1
python prompred_train.py --mode loso --target ternary --class_boundary_method opt_balanced_acc
```

These optimized thresholds are fit from the training speakers only in each fold.

If per-rater labels are available, classification can use those instead of deriving classes from the mean rating. The expected file format is:

```text
========== Participant: 1 Date: ... ==========
file_id; sentence text; 0 0 1 0 2 ...
```

`file_id` must match the basename of a `.csv`/`.wav` pair under `data/spkN/`. The script aligns the rater sentence tokens to the timed CSV words, so it can handle untimed words that appear in the rater sentence but not in the CSV.

Use majority-vote hard labels:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_majority \
  --rater_file per-rater2.txt
```

Use weighted-majority hard labels where rating `2` counts double:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_weighted_majority \
  --rater_file per-rater2.txt
```

Use rater-distribution soft labels:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_soft \
  --rater_file per-rater2.txt
```

For binary per-rater training, ratings `1` and `2` are grouped as the prominent class. For soft-label training, the target is the empirical rater distribution per word, for example `[P(0), P(1), P(2)]` in ternary mode.

For `rater_soft`, hard class diagnostics and inference classes are derived from predicted/observed rater proportions instead of plain argmax. Defaults:

- Binary: class `1` if `P(rating > 0) >= 0.25`
- Ternary: class `2` if `P(rating = 2) >= 0.20`, else class `1` if `P(rating > 0) >= 0.25`, else class `0`

These can be changed:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_soft \
  --rater_file per-rater2.txt \
  --soft_ternary_prom_threshold 0.30 \
  --soft_ternary_strong_threshold 0.15
```

The `rater_soft` proportion thresholds can also be optimized from the training speakers in each LOSO fold:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_soft \
  --rater_file per-rater2.txt \
  --soft_threshold_method opt_macro_f1
```

Use `--soft_threshold_method opt_balanced_acc` to optimize balanced accuracy instead. The optimization target is agreement with the fold-local rater-majority labels, while the model can still train on soft rater distributions.

Categorical training also supports fold-local balanced loss weights and an ordinal head:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_soft \
  --rater_file per-rater2.txt \
  --class_loss_weighting balanced
```

`balanced` computes class weights inside each LOSO training fold only, so held-out speaker labels are not used for weighting. With `rater_soft`, weights are computed from summed class probabilities rather than majority labels.

The ordinal head replaces a flat 3-way softmax with cumulative binary decisions:

```bash
python prompred_train.py \
  --mode loso \
  --target ternary \
  --class_label_source rater_soft \
  --rater_file per-rater2.txt \
  --class_head ordinal \
  --class_loss_weighting balanced
```

For ternary classification, the ordinal head predicts `P(rating > 0)` and `P(rating > 1)`. For binary classification, it predicts `P(rating > 0)`.

Train final model on all speakers and save checkpoint(s) to `models/`:

```bash
python prompred_train.py --mode all
```

Single-seed full training:

```bash
python prompred_train.py --mode all --seed 142857
```

Train deployable classifier checkpoints:

```bash
python prompred_train.py --mode all --target binary --seed 142857
python prompred_train.py --mode all --target ternary --seed 142857
```

Classifier checkpoints are saved with the target and boundary method in the filename, for example `models/prom_model_full_binary_kmeans_seed142857.pt`. The learned class thresholds are stored in the checkpoint and reused by `prompred_infer.py`.
For per-rater classifier checkpoints, the label source is used in the filename, for example `models/prom_model_full_ternary_rater_soft_seed142857.pt`.

Outputs:

- feature cache in `cache/` (auto-created)
- plots in `plots/` (during LOSO)
- model checkpoint(s) in `models/`, e.g. `models/prom_model_full_seed142857.pt`

### 6.4 Inference Script (`prompred_infer.py`)

Required inputs:

- `--checkpoint`: trained `.pt` checkpoint (for example from `models/`)
- either `--wav` (single-file mode) or `--input_dir` (batch directory mode)

Optional:

- `--csv`: segments/tokens file
- `--interval`, `--overlap`: sliding window settings if no CSV is provided
- `--input_dir`: recursively find `.wav` + matching `.csv` basename pairs
- `--inplace`: directory mode only; overwrite matched CSVs with `start,end,word,predicted_rating`
- `--no_header`: write output CSV without a header row
- `--suppress_silence`: apply energy-based masking of non-speech regions
- `--rms_db_thresh`: absolute RMS threshold in dB for `--suppress_silence` (optional)
- `--rms_db_percentile`: adaptive RMS threshold percentile when `--rms_db_thresh` is not set
- `--use_vad`: apply optional librosa-based VAD mask
- `--vad_top_db`: VAD sensitivity for `--use_vad`
- `--silence_zero_thresh`: zero segment prediction if speech-mask ratio is below this threshold
- `--smooth_ms`: moving-average smoothing (ms) for the output prominence curve
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
Example with automatic sliding windows (no CSV):

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --interval 0.4 \
  --overlap 0.1 \
  --out_csv example_data/seg_006_windows_pred.csv
```

Example batch inference over a directory tree with in-place CSV without header overwrite:

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --input_dir example_data \
  --inplace \
  --no_header
```

Notes:

- In directory mode, the script descends recursively and processes files where `<name>.wav` and `<name>.csv` both exist.
- In directory mode with `--inplace`, each matched CSV is overwritten with: `start,end,word,predicted_rating`.
- Sliding-window inference (`--interval` / `--overlap`) remains available in single-file mode (`--wav`) when no `--csv` is provided.

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

When the checkpoint is a classifier checkpoint, inference writes class outputs instead of a continuous `pred` value:

- `pred_class`: numeric class id
- `pred_label`: readable class label
- `pred_prob`: probability of the predicted class
- `prob_<class>` columns: per-class probabilities
- `obs_class`: observed class, when the input CSV contains ratings

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

Mitigation example (single-file sliding windows):

```bash
python prompred_infer.py \
  --checkpoint models/prom_model_full_seed142857.pt \
  --wav example_data/seg_006.wav \
  --interval 0.4 \
  --overlap 0.1 \
  --suppress_silence \
  --use_vad \
  --smooth_ms 120 \
  --out_csv example_data/seg_006_windows_mitigated_pred.csv
```

Suggested presets:

- Conservative (minimal suppression):
```bash
python prompred_infer.py --checkpoint models/prom_model_full_seed142857.pt --wav example_data/seg_006.wav --interval 0.4 --overlap 0.1 --suppress_silence --rms_db_percentile 10 --silence_zero_thresh 0.10 --smooth_ms 60 --out_csv example_data/seg_006_windows_cons_pred.csv
```

- Balanced (recommended starting point):
```bash
python prompred_infer.py --checkpoint models/prom_model_full_seed142857.pt --wav example_data/seg_006.wav --interval 0.4 --overlap 0.1 --suppress_silence --use_vad --rms_db_percentile 20 --vad_top_db 35 --silence_zero_thresh 0.20 --smooth_ms 120 --out_csv example_data/seg_006_windows_balanced_pred.csv
```

- Aggressive (strong silence rejection):
```bash
python prompred_infer.py --checkpoint models/prom_model_full_seed142857.pt --wav example_data/seg_006.wav --interval 0.4 --overlap 0.1 --suppress_silence --use_vad --rms_db_percentile 35 --vad_top_db 25 --silence_zero_thresh 0.35 --smooth_ms 180 --out_csv example_data/seg_006_windows_aggressive_pred.csv
```

This limitation does not affect inference when using word-aligned CSV input.
