# Prosoduc Cues and Modeling Strategies in Swedish Prominence Prediction

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

