import os
import glob
import random
import warnings
import csv
import re
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/tmp", "numba_cache"))
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
from scipy.stats import pearsonr
from scipy.interpolate import interp1d # <--- NEW IMPORT
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_ROOT = 'data'
CACHE_DIR = 'cache'

W2V_MODEL_NAME = "KBLab/wav2vec2-large-voxrex-swedish" 
#W2V_MODEL_NAME = "facebook/wav2vec2-base-960h"

# --- FEATURE FLAGS ---
USE_SSL = True          # Use Wav2Vec/VoxRex frame embeddings
USE_RAW_PITCH = False    # Frame-by-frame pitch (keep False if using Shape)
USE_SCALARS = True       # Keep True
USE_PITCH_SHAPE = True   # <--- NEW: Use Polynomial Shape instead of basic stats

# --- ARCHITECTURE ---
USE_ATTENTION = True
USE_MAX_POOLING = True
USE_WEIGHTED_LOSS = True

BATCH_SIZE = 8
HIDDEN_DIM = 64       
NUM_LAYERS = 1
DROPOUT = 0.3 
LEARNING_RATE = 0.001
EPOCHS = 35
MAX_FRAMES_PER_WORD = 50 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#SEEDS_TO_TEST = [42, 100, 2025]
SEEDS_TO_TEST = [42, 100, 555, 1234, 2025]

TARGET_MODE = "regression"
CLASS_BOUNDARY_METHOD = "kmeans"
CLASS_LABEL_SOURCE = "mean"
RATER_COUNTS_BY_FILE = None
CLASS_HEAD = "softmax"
CLASS_LOSS_WEIGHTING = "none"
SOFT_BINARY_PROM_THRESHOLD = 0.25
SOFT_TERNARY_PROM_THRESHOLD = 0.25
SOFT_TERNARY_STRONG_THRESHOLD = 0.20
SOFT_THRESHOLD_METHOD = "fixed"
THRESHOLD_MIN_COUNT_PER_CLASS = 1

print(f"Running on: {DEVICE}")
print(f"Model: {W2V_MODEL_NAME if USE_SSL else 'PiSh-only (no SSL backbone)'}")
print("-" * 30)
print(f"FEATS: SSL={USE_SSL}, Shape={USE_PITCH_SHAPE}, RawPitch={USE_RAW_PITCH}")
print(f"ARCH:  Attn={USE_ATTENTION}, Max={USE_MAX_POOLING}, Weighted={USE_WEIGHTED_LOSS}")
print("-" * 30)

# ==========================================
# 2. SETUP
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

processor = None
w2v_model = None

def load_w2v_model():
    global processor, w2v_model
    if not USE_SSL:
        return
    if w2v_model is None:
        print(f"Loading Wav2Vec2 Model...")
        processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
        w2v_model = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME).to(DEVICE)
        w2v_model.eval()

from transformers import AutoConfig
W2V_DIM = 0
FRAME_DIM = 1

def configure_feature_dims():
    global W2V_DIM, FRAME_DIM
    if USE_SSL:
        config = AutoConfig.from_pretrained(W2V_MODEL_NAME)
        W2V_DIM = config.hidden_size
        FRAME_DIM = W2V_DIM + 1 if USE_RAW_PITCH else W2V_DIM
    else:
        W2V_DIM = 0
        FRAME_DIM = 1

# Dimension Logic
# [LogDur, RMS_Mean, RMS_Max, Cent_Mean] = 4 Base Scalars
# If PITCH_SHAPE: [a, b, c, error] = 4 Shape Scalars
# If Basic Stats: [Mean, SD, Range] = 3 Stats
if USE_SCALARS:
    SCALAR_DIM = 4 + (4 if USE_PITCH_SHAPE else 3)
else:
    SCALAR_DIM = 1

# ==========================================
# 3. FEATURE EXTRACTION (With Shape)
# ==========================================

def get_pitch_shape_coeffs(f0_seq):
    """
     Fits a quadratic curve (ax^2 + bx + c) to the pitch contour.
     Returns [a, b, c, residual_error]
    """
    # 1. Handle Silence / NaNs
    # If word is mostly unvoiced, return zeros
    valid_mask = ~np.isnan(f0_seq)
    if np.sum(valid_mask) < 3: # Need at least 3 points for quadratic fit
        return [0.0, 0.0, 0.0, 0.0]
    
    # 2. Interpolate over gaps (stylization)
    indices = np.arange(len(f0_seq))
    valid_indices = indices[valid_mask]
    valid_values = f0_seq[valid_mask]
    
    # Create interpolator
    interp_func = interp1d(valid_indices, valid_values, kind='linear', 
                           fill_value="extrapolate")
    f0_interp = interp_func(indices)
    
    # 3. Normalize Time axis to [-1, 1] for stable polyfit
    # This ensures 'slope' means the same thing for long and short words
    t_norm = np.linspace(-1, 1, len(f0_interp))
    
    # 4. Polyfit (Degree 2) -> ax^2 + bx + c
    # coeffs returned as [a, b, c] (highest power first)
    coeffs, residuals, _, _, _ = np.polyfit(t_norm, f0_interp, 2, full=True)
    
    # Residual error (how complex/wiggly was the curve?)
    mse_error = (residuals[0] / len(f0_interp)) if len(residuals) > 0 else 0.0
    
    return [coeffs[0], coeffs[1], coeffs[2], mse_error]

def extract_features(wav_path, csv_path):
    load_w2v_model()
    y, sr = librosa.load(wav_path, sr=16000)
    
    # W2V2
    if USE_SSL:
        inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = w2v_model(**inputs)
        w2v_frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    else:
        w2v_frames = None
    
    # Acoustic Extraction
    HOP = 320 
    f0_raw, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, frame_length=1024, hop_length=HOP)
    
    # Note: We keep NaNs in f0_raw for Shape interpolation logic
    f0_frames = np.nan_to_num(f0_raw).reshape(-1, 1) 
    
    rms_raw = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    cent_raw = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=HOP)[0]
    
    if USE_SSL:
        min_len = min(len(w2v_frames), len(f0_frames), len(rms_raw), len(cent_raw))
    else:
        min_len = min(len(f0_frames), len(rms_raw), len(cent_raw))
    
    MODEL_FPS = 50 
    df = pd.read_csv(csv_path, header=None, names=['start', 'end', 'word', 'rating'])
    
    out_frames = []
    out_scalars = []
    out_labels = []
    
    for _, row in df.iterrows():
        start_idx = int(row['start'] * MODEL_FPS)
        end_idx = int(row['end'] * MODEL_FPS)
        
        if start_idx >= min_len: continue 
        if end_idx > min_len: end_idx = min_len
        if start_idx >= end_idx:
            start_idx = min(start_idx, min_len - 1)
            end_idx = start_idx + 1

        # --- Frames ---
        if USE_SSL:
            curr_w2v = w2v_frames[start_idx:end_idx][:min_len]
        else:
            curr_w2v = np.zeros((end_idx - start_idx, FRAME_DIM), dtype=np.float32)

        if USE_RAW_PITCH and USE_SSL:
            curr_frames = np.concatenate([curr_w2v, f0_frames[start_idx:end_idx]], axis=1)
        else:
            curr_frames = curr_w2v

        if len(curr_frames) > MAX_FRAMES_PER_WORD:
            s = (len(curr_frames) - MAX_FRAMES_PER_WORD) // 2
            curr_frames = curr_frames[s : s + MAX_FRAMES_PER_WORD]
        else:
            pad = MAX_FRAMES_PER_WORD - len(curr_frames)
            curr_frames = np.pad(curr_frames, ((0, pad), (0,0)), mode='constant')

        # --- Scalars ---
        dur = row['end'] - row['start']
        log_dur = np.log(dur + 1e-6)
        
        rms_seg = rms_raw[start_idx:end_idx]
        cent_seg = cent_raw[start_idx:end_idx]
        
        rms_mean = np.mean(rms_seg) if len(rms_seg)>0 else 0
        rms_max = np.max(rms_seg) if len(rms_seg)>0 else 0
        cent_mean = np.mean(cent_seg) if len(cent_seg)>0 else 0
        
        # PITCH FEATURES
        f0_seg = f0_raw[start_idx:end_idx]
        
        if USE_PITCH_SHAPE:
            # Extract [Curvature, Slope, Height, Error]
            pitch_feats = get_pitch_shape_coeffs(f0_seg)
        else:
            # Old Logic
            if len(f0_seg) == 0 or np.all(np.isnan(f0_seg)):
                pitch_feats = [0, 0, 0]
            else:
                pitch_feats = [np.nanmean(f0_seg), np.nanstd(f0_seg), 
                               np.nanmax(f0_seg) - np.nanmin(f0_seg)]
            
        # Combine
        scalar_vec = [log_dur, rms_mean, rms_max, cent_mean] + pitch_feats
        
        out_frames.append(curr_frames)
        out_scalars.append(scalar_vec)
        out_labels.append(row['rating'])
        
    return (np.array(out_frames, dtype=np.float32), 
            np.array(out_scalars, dtype=np.float32), 
            np.array(out_labels, dtype=np.float32))

# ==========================================
# CACHING
# ==========================================
def get_cache_filename():
    safe_model = W2V_MODEL_NAME.replace("/", "_").replace("-", "_")
    ssl_name = safe_model if USE_SSL else "no_ssl"
    return f"feats_{ssl_name}_shape{USE_PITCH_SHAPE}_raw{USE_RAW_PITCH}_scalars{USE_SCALARS}.pt"

def precompute_data(root_dir):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, get_cache_filename())
    
    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        return torch.load(cache_path,weights_only=False)

    print(f"Extracting features...")
    data_map = {}
    spk_dirs = glob.glob(os.path.join(root_dir, '*'))
    
    for spk_path in spk_dirs:
        if not os.path.isdir(spk_path): continue
        spk_name = os.path.basename(spk_path)
        data_map[spk_name] = []
        wav_files = glob.glob(os.path.join(spk_path, '*.wav'))
        for wav in tqdm(wav_files, desc=f"Parsing {spk_name}", leave=False):
            base = os.path.splitext(wav)[0]
            csv = base + '.csv'
            if os.path.exists(csv):
                frames, scalars, labs = extract_features(wav, csv)
                if len(labs) > 0:
                    data_map[spk_name].append((frames, scalars, labs, wav, csv))
    
    print(f"Saving cache: {cache_path}")
    torch.save(data_map, cache_path)
    return data_map

# ==========================================
# 4. DATASET & COLLATOR
# ==========================================
class ProminenceDataset(Dataset):
    def __init__(self, data_list, scalar_scaler=None, frame_scaler=None, training=True, label_transform=None):
        self.data = data_list 
        self.label_transform = label_transform
        # SCALAR_DIM is calculated dynamically at top of script
        self.scalar_slice = SCALAR_DIM 
        
        if training:
            all_scalars = np.concatenate([d[1] for d in self.data], axis=0)
            # Safety slice in case cache has more dims
            all_scalars = all_scalars[:, :self.scalar_slice]
            self.scalar_scaler = StandardScaler().fit(all_scalars)
            
            if USE_RAW_PITCH:
                all_frames = [d[0] for d in self.data]
                dim = all_frames[0].shape[-1] 
                flat = np.concatenate(all_frames, axis=0).reshape(-1, dim)
                self.frame_scaler = StandardScaler().fit(flat[:, -1].reshape(-1, 1))
            else:
                self.frame_scaler = None
        else:
            self.scalar_scaler = scalar_scaler
            self.frame_scaler = frame_scaler

        self.processed_data = []
        for frames, scalars, labs, w_path, c_path in self.data:
            scalars_sliced = scalars[:, :self.scalar_slice]
            norm_scalars = self.scalar_scaler.transform(scalars_sliced)
            
            norm_frames = frames.copy()
            if USE_RAW_PITCH and self.frame_scaler:
                B, T, F = norm_frames.shape
                norm_frames[:, :, -1] = self.frame_scaler.transform(norm_frames[:, :, -1].reshape(-1, 1)).reshape(B, T)
                
            if self.label_transform is not None:
                labs = self.label_transform(labs, c_path)

            self.processed_data.append((norm_frames, norm_scalars, labs, w_path, c_path))

    def __len__(self): return len(self.processed_data)
    def __getitem__(self, idx):
        return (torch.tensor(self.processed_data[idx][0]), 
                torch.tensor(self.processed_data[idx][1]),
                torch.tensor(self.processed_data[idx][2]),
                self.processed_data[idx][3],
                self.processed_data[idx][4])

def pad_collate(batch):
    (frames, scalars, labels, wavs, csvs) = zip(*batch)
    x_lens = torch.tensor([len(f) for f in frames])
    frames_pad = pad_sequence(frames, batch_first=True, padding_value=0)
    scalars_pad = pad_sequence(scalars, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=-1)
    return frames_pad, scalars_pad, labels_pad, x_lens, wavs, csvs

# ==========================================
# 5. MODEL
# ==========================================
class ConfigurablePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.use_attn = USE_ATTENTION
        self.use_max = USE_MAX_POOLING
        if self.use_attn:
            self.W = nn.Linear(input_dim, 128)
            self.v = nn.Linear(128, 1)
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = []
        if self.use_attn:
            u = self.tanh(self.W(x))
            scores = self.v(u)
            weights = self.softmax(scores)
            outputs.append(torch.sum(weights * x, dim=1))
        else:
            outputs.append(torch.mean(x, dim=1))
        if self.use_max:
            outputs.append(torch.max(x, dim=1)[0])
        return torch.cat(outputs, dim=1)

class ProminencePredictor(nn.Module):
    def __init__(self, frame_dim, scalar_dim, hidden_dim=64, output_dim=1, target_mode="regression"):
        super().__init__()
        self.target_mode = target_mode
        self.use_ssl = USE_SSL
        if self.use_ssl:
            self.pooling = ConfigurablePooling(frame_dim)
            pooling_multiplier = 1 + (1 if USE_MAX_POOLING else 0)
            lstm_input_dim = (frame_dim * pooling_multiplier) + scalar_dim
        else:
            self.pooling = None
            lstm_input_dim = scalar_dim
        
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, 
                            num_layers=NUM_LAYERS, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=DROPOUT if NUM_LAYERS > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.activation = nn.Sigmoid() 

    def forward(self, frames, scalars, lengths):
        if self.use_ssl:
            batch_size, seq_len, num_frames, feat_dim = frames.shape
            flat_frames = frames.view(-1, num_frames, feat_dim)
            pooled_feats = self.pooling(flat_frames)
            word_embeddings = pooled_feats.view(batch_size, seq_len, -1)
            lstm_input = torch.cat([word_embeddings, scalars], dim=2)
        else:
            lstm_input = scalars
        packed_x = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        logits = self.fc(out)
        if self.target_mode == "regression":
            return (self.activation(logits) * 3.0).squeeze(-1)
        return logits

# ==========================================
# 5.1 CLASS TARGETS
# ==========================================
def _collect_labels(data_list):
    if not data_list:
        return np.array([], dtype=np.float32)
    return np.concatenate([d[2] for d in data_list], axis=0).astype(np.float32)

def _normalize_token(token):
    token = str(token).strip().lower()
    token = token.replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"^[^\wåäöéüÅÄÖÉÜ]+|[^\wåäöéüÅÄÖÉÜ-]+$", "", token)

def _tokenize_text(text):
    return [tok for tok in (_normalize_token(t) for t in str(text).split()) if tok]

def _tokenize_rater_text(text):
    # The annotation export uses spaces as token separators and occasionally
    # has double spaces; keeping the empty token preserves rating alignment.
    return [_normalize_token(t) for t in str(text).split(" ")]

def _lcs_index_map(needle, haystack):
    n = len(needle)
    m = len(haystack)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if needle[i] == haystack[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    mapping = {}
    i = 0
    j = 0
    while i < n and j < m:
        if needle[i] == haystack[j]:
            mapping[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return mapping

def load_rater_counts(rater_file, data_root=DATA_ROOT):
    raw = {}
    current_participant = None
    with open(rater_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                current_participant = line
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) != 3:
                raise ValueError(f"Bad rater line {lineno}: expected 3 semicolon-separated fields.")
            file_id, text, ratings_text = parts
            ratings = [int(x) for x in ratings_text.split()]
            if any(r not in (0, 1, 2) for r in ratings):
                raise ValueError(f"Bad rater line {lineno}: ratings must be 0, 1, or 2.")
            text_tokens = _tokenize_rater_text(text)
            if len(text_tokens) != len(ratings):
                raise ValueError(
                    f"Bad rater line {lineno}: token count {len(text_tokens)} != rating count {len(ratings)}."
                )
            raw.setdefault(file_id, []).append((current_participant, text_tokens, ratings))

    counts_by_file = {}
    csv_paths = sorted(glob.glob(os.path.join(data_root, "spk*", "*.csv")))
    stems = {os.path.splitext(os.path.basename(p))[0]: p for p in csv_paths}
    missing = sorted(set(stems) - set(raw))
    extra = sorted(set(raw) - set(stems))
    if missing:
        raise ValueError(f"Rater file is missing {len(missing)} CSV file ids; first: {missing[:5]}")
    if extra:
        raise ValueError(f"Rater file has {len(extra)} ids not found under {data_root}; first: {extra[:5]}")

    for file_id, csv_path in stems.items():
        csv_words = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    csv_words.append(row[2])
        csv_tokens = [_normalize_token(w) for w in csv_words]
        counts = np.zeros((len(csv_tokens), 3), dtype=np.float32)

        for _, text_tokens, ratings in raw[file_id]:
            mapping = _lcs_index_map(csv_tokens, text_tokens)
            if len(mapping) != len(csv_tokens):
                missing_words = [csv_words[i] for i in range(len(csv_tokens)) if i not in mapping]
                raise ValueError(
                    f"Could not align all CSV words for {file_id}; missing examples: {missing_words[:8]}"
                )
            for csv_idx, text_idx in mapping.items():
                counts[csv_idx, ratings[text_idx]] += 1.0

        counts_by_file[file_id] = counts

    total_words = sum(v.shape[0] for v in counts_by_file.values())
    rater_counts = [int(v.sum(axis=1).min()) for v in counts_by_file.values()]
    print(
        f"Loaded per-rater labels: {len(counts_by_file)} files, {total_words} words, "
        f"min raters/word={min(rater_counts)}, max raters/word={max(rater_counts)}"
    )
    return counts_by_file

def _target_to_num_classes(target_mode):
    if target_mode == "binary":
        return 2
    if target_mode == "ternary":
        return 3
    return 1

def _target_to_output_dim(target_mode, class_head="softmax"):
    num_classes = _target_to_num_classes(target_mode)
    if target_mode == "regression":
        return 1
    if class_head == "ordinal":
        return num_classes - 1
    return num_classes

def _classification_score(y_true, y_pred, objective):
    if objective == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if objective == "balanced_acc":
        return balanced_accuracy_score(y_true, y_pred)
    raise ValueError(f"Unknown threshold objective: {objective}")

def _candidate_thresholds(values, min_count_per_class=1):
    values = np.sort(np.unique(np.asarray(values, dtype=np.float32)))
    if len(values) < 2:
        return []
    mids = ((values[:-1] + values[1:]) / 2.0).astype(np.float32)
    return [float(x) for x in mids]

def _optimize_mean_thresholds(labels, target_mode, objective="macro_f1", min_count_per_class=1):
    labels = np.asarray(labels, dtype=np.float32)
    labels = labels[np.isfinite(labels)]
    candidates = _candidate_thresholds(labels, min_count_per_class=min_count_per_class)
    if not candidates:
        raise ValueError("Need at least two distinct labels to optimize thresholds.")

    if target_mode == "binary":
        reference = np.rint(labels).clip(0, 1).astype(np.int64)
        best = None
        for t in candidates:
            pred = np.digitize(labels, [t]).astype(np.int64)
            counts = np.bincount(pred, minlength=2)
            if np.any(counts < min_count_per_class):
                continue
            score = _classification_score(reference, pred, objective)
            item = (score, -abs(t - 0.5), [float(t)])
            if best is None or item > best:
                best = item
        if best is None:
            raise ValueError("No valid binary threshold found with the requested minimum class count.")
        return best[2]

    if target_mode == "ternary":
        best = None
        reference = np.rint(labels).clip(0, 2).astype(np.int64)
        for i, t1 in enumerate(candidates):
            for t2 in candidates[i + 1:]:
                pred = np.digitize(labels, [t1, t2]).astype(np.int64)
                counts = np.bincount(pred, minlength=3)
                if np.any(counts < min_count_per_class):
                    continue
                score = _classification_score(reference, pred, objective)
                centered = -(abs(t1 - 0.5) + abs(t2 - 1.5))
                item = (score, centered, [float(t1), float(t2)])
                if best is None or item > best:
                    best = item
        if best is None:
            raise ValueError("No valid ternary thresholds found with the requested minimum class count.")
        return best[2]

    raise ValueError(f"Unsupported target mode for optimized thresholds: {target_mode}")

def _soft_probs_to_reference_classes(probs, target_mode):
    if target_mode == "binary":
        return np.argmax(probs, axis=1).astype(np.int64)
    if target_mode == "ternary":
        return np.argmax(probs, axis=1).astype(np.int64)
    raise ValueError(f"Unsupported target mode: {target_mode}")

def _collect_soft_probs(data_list, rater_counts_by_file, target_mode):
    all_probs = []
    for _, _, labs, _, csv_path in data_list:
        file_id = os.path.splitext(os.path.basename(csv_path))[0]
        counts = rater_counts_by_file[file_id]
        if len(counts) != len(labs):
            raise ValueError(f"Rater count length mismatch for {file_id}: {len(counts)} != {len(labs)}")
        if target_mode == "binary":
            class_counts = np.stack([counts[:, 0], counts[:, 1] + counts[:, 2]], axis=1)
        else:
            class_counts = counts
        denom = np.maximum(class_counts.sum(axis=1, keepdims=True), 1.0)
        all_probs.append((class_counts / denom).astype(np.float32))
    return np.concatenate(all_probs, axis=0)

def optimize_soft_threshold_config(data_list, target_mode, rater_counts_by_file, objective="macro_f1", min_count_per_class=1):
    probs = _collect_soft_probs(data_list, rater_counts_by_file, target_mode)
    reference = _soft_probs_to_reference_classes(probs, target_mode)
    n_classes = _target_to_num_classes(target_mode)
    base_cfg = rater_class_config(target_mode, "rater_soft")

    if target_mode == "binary":
        candidates = sorted(set(float(x) for x in probs[:, 1]))
        candidates = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] + candidates
        candidates = sorted(set(round(x, 6) for x in candidates if 0.0 < x < 1.0))
        best = None
        for t in candidates:
            cfg = dict(base_cfg)
            cfg["prom_threshold"] = float(t)
            pred = probs_to_classes(probs, target_mode, cfg)
            counts = np.bincount(pred, minlength=n_classes)
            if np.any(counts < min_count_per_class):
                continue
            score = _classification_score(reference, pred, objective)
            item = (score, -abs(t - SOFT_BINARY_PROM_THRESHOLD), float(t), cfg)
            if best is None or item > best:
                best = item
        if best is None:
            raise ValueError("No valid binary soft threshold found with the requested minimum class count.")
        cfg = best[3]
    elif target_mode == "ternary":
        prom_scores = probs[:, 1] + probs[:, 2]
        strong_scores = probs[:, 2]
        prom_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        strong_candidates = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
        prom_candidates += [float(x) for x in np.unique(prom_scores)]
        strong_candidates += [float(x) for x in np.unique(strong_scores)]
        prom_candidates = sorted(set(round(x, 6) for x in prom_candidates if 0.0 < x < 1.0))
        strong_candidates = sorted(set(round(x, 6) for x in strong_candidates if 0.0 < x < 1.0))
        best = None
        for prom_t in prom_candidates:
            for strong_t in strong_candidates:
                cfg = dict(base_cfg)
                cfg["prom_threshold"] = float(prom_t)
                cfg["strong_threshold"] = float(strong_t)
                pred = probs_to_classes(probs, target_mode, cfg)
                counts = np.bincount(pred, minlength=n_classes)
                if np.any(counts < min_count_per_class):
                    continue
                score = _classification_score(reference, pred, objective)
                centered = -(abs(prom_t - SOFT_TERNARY_PROM_THRESHOLD) + abs(strong_t - SOFT_TERNARY_STRONG_THRESHOLD))
                item = (score, centered, float(prom_t), float(strong_t), cfg)
                if best is None or item > best:
                    best = item
        if best is None:
            raise ValueError("No valid ternary soft thresholds found with the requested minimum class count.")
        cfg = best[4]
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")

    cfg["threshold_method"] = objective
    cfg["threshold_reference"] = "rater_majority"
    return cfg

def fit_class_config(labels, target_mode, method="kmeans", seed=42, objective="macro_f1", min_count_per_class=1):
    labels = np.asarray(labels, dtype=np.float32)
    labels = labels[np.isfinite(labels)]
    n_classes = _target_to_num_classes(target_mode)
    if target_mode == "regression":
        return None
    if len(labels) == 0:
        raise ValueError("Cannot fit class boundaries without labels.")

    threshold_objective = None
    if method == "round":
        thresholds = [0.5] if n_classes == 2 else [0.5, 1.5]
    elif method == "kmeans":
        unique = np.unique(labels)
        if len(unique) < n_classes:
            raise ValueError(
                f"Need at least {n_classes} distinct target values to learn {target_mode} boundaries; got {len(unique)}."
            )
        km = KMeans(n_clusters=n_classes, n_init=20, random_state=seed)
        km.fit(labels.reshape(-1, 1))
        centers = sorted(float(c) for c in km.cluster_centers_.reshape(-1))
        thresholds = [(centers[i] + centers[i + 1]) / 2.0 for i in range(n_classes - 1)]
    elif method == "gmm":
        unique = np.unique(labels)
        if len(unique) < n_classes:
            raise ValueError(
                f"Need at least {n_classes} distinct target values to learn {target_mode} boundaries; got {len(unique)}."
            )
        gmm = GaussianMixture(
            n_components=n_classes,
            covariance_type="full",
            n_init=10,
            random_state=seed,
            reg_covar=1e-6,
        )
        gmm.fit(labels.reshape(-1, 1))
        means = gmm.means_.reshape(-1)
        order = np.argsort(means)
        sorted_means = means[order]
        sorted_weights = gmm.weights_[order]
        sorted_vars = gmm.covariances_.reshape(-1)[order]
        thresholds = []
        for i in range(n_classes - 1):
            m1, m2 = float(sorted_means[i]), float(sorted_means[i + 1])
            w1, w2 = float(sorted_weights[i]), float(sorted_weights[i + 1])
            v1, v2 = float(sorted_vars[i]), float(sorted_vars[i + 1])
            lo, hi = m1, m2
            grid = np.linspace(lo, hi, 1000)
            d1 = np.log(max(w1, 1e-12)) - 0.5 * np.log(max(v1, 1e-12)) - ((grid - m1) ** 2) / (2.0 * max(v1, 1e-12))
            d2 = np.log(max(w2, 1e-12)) - 0.5 * np.log(max(v2, 1e-12)) - ((grid - m2) ** 2) / (2.0 * max(v2, 1e-12))
            thresholds.append(float(grid[int(np.argmin(np.abs(d1 - d2)))]))
    elif method in ("opt_macro_f1", "opt_balanced_acc"):
        opt_objective = "macro_f1" if method == "opt_macro_f1" else "balanced_acc"
        threshold_objective = opt_objective
        thresholds = _optimize_mean_thresholds(
            labels,
            target_mode,
            objective=opt_objective,
            min_count_per_class=min_count_per_class,
        )
    else:
        raise ValueError(f"Unknown class boundary method: {method}")

    class_names = ["no_prominence", "prominent"] if n_classes == 2 else [
        "no_prominence",
        "maybe_prominent",
        "prominent",
    ]
    return {
        "target_mode": target_mode,
        "num_classes": n_classes,
        "boundary_method": method,
        "threshold_objective": threshold_objective,
        "thresholds": [float(t) for t in thresholds],
        "class_names": class_names,
    }

def rater_class_config(target_mode, label_source):
    n_classes = _target_to_num_classes(target_mode)
    class_names = ["no_prominence", "prominent"] if n_classes == 2 else [
        "no_prominence",
        "maybe_prominent",
        "prominent",
    ]
    cfg = {
        "target_mode": target_mode,
        "num_classes": n_classes,
        "boundary_method": label_source,
        "thresholds": None,
        "class_names": class_names,
    }
    if label_source == "rater_soft":
        if target_mode == "binary":
            cfg["decision_method"] = "proportion_threshold"
            cfg["prom_threshold"] = float(SOFT_BINARY_PROM_THRESHOLD)
        elif target_mode == "ternary":
            cfg["decision_method"] = "proportion_threshold"
            cfg["prom_threshold"] = float(SOFT_TERNARY_PROM_THRESHOLD)
            cfg["strong_threshold"] = float(SOFT_TERNARY_STRONG_THRESHOLD)
    return cfg

def probs_to_classes(probs, target_mode, class_config):
    probs = np.asarray(probs, dtype=np.float32)
    if target_mode == "binary":
        prom_prob = probs[:, 1]
        threshold = float(class_config.get("prom_threshold", SOFT_BINARY_PROM_THRESHOLD))
        return (prom_prob >= threshold).astype(np.int64)
    if target_mode == "ternary":
        prom_prob = probs[:, 1] + probs[:, 2]
        strong_prob = probs[:, 2]
        prom_threshold = float(class_config.get("prom_threshold", SOFT_TERNARY_PROM_THRESHOLD))
        strong_threshold = float(class_config.get("strong_threshold", SOFT_TERNARY_STRONG_THRESHOLD))
        out = np.zeros((len(probs),), dtype=np.int64)
        out[prom_prob >= prom_threshold] = 1
        out[strong_prob >= strong_threshold] = 2
        return out
    raise ValueError(f"Unsupported class target mode: {target_mode}")

def ordinal_probs_to_classes(probs, target_mode, class_config=None):
    probs = np.asarray(probs, dtype=np.float32)
    if target_mode == "binary":
        threshold = 0.5
        if class_config and class_config.get("decision_method") == "proportion_threshold":
            threshold = float(class_config.get("prom_threshold", threshold))
        return (probs[:, 0] >= threshold).astype(np.int64)
    if target_mode == "ternary":
        prom_threshold = 0.5
        strong_threshold = 0.5
        if class_config and class_config.get("decision_method") == "proportion_threshold":
            prom_threshold = float(class_config.get("prom_threshold", prom_threshold))
            strong_threshold = float(class_config.get("strong_threshold", strong_threshold))
        out = np.zeros((len(probs),), dtype=np.int64)
        out[probs[:, 0] >= prom_threshold] = 1
        out[probs[:, 1] >= strong_threshold] = 2
        return out
    raise ValueError(f"Unsupported class target mode: {target_mode}")

def class_targets_to_ordinal_targets(y, target_mode):
    if target_mode == "binary":
        return (y > 0).float().unsqueeze(-1)
    if target_mode == "ternary":
        return torch.stack([(y > 0).float(), (y > 1).float()], dim=-1)
    raise ValueError(f"Unsupported class target mode: {target_mode}")

def soft_targets_to_ordinal_targets(y_prob, target_mode):
    if target_mode == "binary":
        return y_prob[:, :, 1:2]
    if target_mode == "ternary":
        return torch.stack([y_prob[:, :, 1] + y_prob[:, :, 2], y_prob[:, :, 2]], dim=-1)
    raise ValueError(f"Unsupported class target mode: {target_mode}")

def compute_class_weights(data_list, label_transform, target_mode):
    num_classes = _target_to_num_classes(target_mode)
    counts = np.zeros((num_classes,), dtype=np.float64)
    for _, _, labs, _, csv_path in data_list:
        y = label_transform(labs, csv_path)
        if y.ndim == 2:
            counts += y.sum(axis=0)
        else:
            counts += np.bincount(y.astype(np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1e-6)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32), counts

def class_weights_to_ordinal_weights(class_weights, target_mode):
    class_weights = np.asarray(class_weights, dtype=np.float32)
    if target_mode == "binary":
        return class_weights.reshape(1, 2)
    if target_mode == "ternary":
        # For the two cumulative heads, balance negative/positive sides:
        # head 0 predicts y>0 (class 0 vs classes 1+2), head 1 predicts y>1 (0+1 vs class 2).
        return np.array([
            [class_weights[0], (class_weights[1] + class_weights[2]) / 2.0],
            [(class_weights[0] + class_weights[1]) / 2.0, class_weights[2]],
        ], dtype=np.float32)
    raise ValueError(f"Unsupported class target mode: {target_mode}")

def _format_class_counts(values, num_classes):
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=num_classes)
    total = max(int(counts.sum()), 1)
    return ", ".join(f"{i}:{int(c)} ({(c / total):.1%})" for i, c in enumerate(counts))

def _format_per_class_f1(targets, preds, num_classes):
    _, _, f1s, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return ", ".join(
        f"{i}:f1={f1s[i]:.3f}/n={int(support[i])}" for i in range(num_classes)
    )

def _print_confusion_matrix(targets, preds, num_classes, title="Confusion matrix"):
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    print(f"{title} (rows=true, cols=pred):")
    header = "true\\pred" + "".join(f"{i:>8}" for i in range(num_classes))
    print(header)
    for i, row in enumerate(cm):
        values = "".join(f"{int(v):>8}" for v in row)
        print(f"{i:>9}{values}")

def _print_row_normalized_confusion_matrix(targets, preds, num_classes, title="Row-normalized confusion matrix"):
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes))).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    print(f"{title} % (rows=true, cols=pred):")
    header = "true\\pred" + "".join(f"{i:>9}" for i in range(num_classes))
    print(header)
    for i, row in enumerate(pct):
        values = "".join(f"{v:>9.2f}" for v in row)
        print(f"{i:>9}{values}")

def labels_to_classes(labels, class_config):
    labels = np.asarray(labels, dtype=np.float32)
    return np.digitize(labels, class_config["thresholds"]).astype(np.int64)

def make_label_transform(target_mode, class_config=None, rater_counts_by_file=None, class_label_source="mean"):
    if target_mode == "regression":
        return None

    def transform(labels, csv_path):
        if class_label_source == "mean":
            return labels_to_classes(labels, class_config).astype(np.int64)

        file_id = os.path.splitext(os.path.basename(csv_path))[0]
        counts = rater_counts_by_file[file_id]
        if len(counts) != len(labels):
            raise ValueError(f"Rater count length mismatch for {file_id}: {len(counts)} != {len(labels)}")

        if target_mode == "binary":
            class_counts = np.stack([counts[:, 0], counts[:, 1] + counts[:, 2]], axis=1)
        else:
            class_counts = counts

        if class_label_source == "rater_majority":
            return np.argmax(class_counts, axis=1).astype(np.int64)
        if class_label_source == "rater_weighted_majority":
            weighted_counts = class_counts.copy()
            if target_mode == "binary":
                weighted_counts[:, 1] = counts[:, 1] + (2.0 * counts[:, 2])
            else:
                weighted_counts[:, 2] *= 2.0
            return np.argmax(weighted_counts, axis=1).astype(np.int64)
        if class_label_source == "rater_soft":
            denom = np.maximum(class_counts.sum(axis=1, keepdims=True), 1.0)
            return (class_counts / denom).astype(np.float32)

        raise ValueError(f"Unknown class label source: {class_label_source}")

    return transform

# ==========================================
# 6. VIZ
# ==========================================
def visualize_file_prominence(wav_path, csv_path, predicted_ratings, save_path):
    try:
        df = pd.read_csv(csv_path, header=None, names=['start', 'end', 'word', 'rating'])
        duration = librosa.get_duration(path=wav_path)
        t_centers = ((df['start'] + df['end']) / 2).values
        plt.figure(figsize=(12, 5))
        plt.plot(t_centers, df['rating'].values, 'b--', alpha=0.6, label='Human', marker='.')
        plt.plot(t_centers, predicted_ratings, 'r-', alpha=0.8, label='Model', marker='.')
        for _, row in df.iterrows():
            wc = (row['start'] + row['end']) / 2
            plt.text(wc, -0.2, row['word'], rotation=45, ha='center', fontsize=8)
        plt.ylim(-0.5, 2.5)
        plt.xlim(0, duration)
        plt.title(os.path.basename(wav_path))
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except: pass

# ==========================================
# 7. TRAINING
# ==========================================
def run_seed_experiment(seed, data_map):
    print(f"\n--- Starting Run for Seed {seed} ---")
    set_seed(seed)
    speakers = sorted(list(data_map.keys()))
    overall_correlations = []
    mse_scores = []
    acc_scores = []
    f1_scores = []
    seed_class_targets = []
    seed_class_preds = []

    for test_spk in speakers:
        train_data = []
        test_data = data_map[test_spk]
        for spk in speakers:
            if spk != test_spk: train_data.extend(data_map[spk])

        class_config = None
        label_transform = None
        if TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "mean":
            class_config = fit_class_config(
                _collect_labels(train_data),
                TARGET_MODE,
                method=CLASS_BOUNDARY_METHOD,
                seed=seed,
                min_count_per_class=THRESHOLD_MIN_COUNT_PER_CLASS,
            )
            print(f"  {test_spk}: learned thresholds={class_config['thresholds']}")
        elif TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "rater_soft" and SOFT_THRESHOLD_METHOD != "fixed":
            objective = "macro_f1" if SOFT_THRESHOLD_METHOD == "opt_macro_f1" else "balanced_acc"
            class_config = optimize_soft_threshold_config(
                train_data,
                TARGET_MODE,
                RATER_COUNTS_BY_FILE,
                objective=objective,
                min_count_per_class=THRESHOLD_MIN_COUNT_PER_CLASS,
            )
            if TARGET_MODE == "binary":
                print(f"  {test_spk}: optimized soft threshold prom={class_config['prom_threshold']:.4f}")
            else:
                print(
                    f"  {test_spk}: optimized soft thresholds "
                    f"prom={class_config['prom_threshold']:.4f}, strong={class_config['strong_threshold']:.4f}"
                )
        elif TARGET_MODE != "regression":
            class_config = rater_class_config(TARGET_MODE, CLASS_LABEL_SOURCE)
        if TARGET_MODE != "regression":
            label_transform = make_label_transform(
                TARGET_MODE,
                class_config=class_config,
                rater_counts_by_file=RATER_COUNTS_BY_FILE,
                class_label_source=CLASS_LABEL_SOURCE,
            )
            
        train_ds = ProminenceDataset(train_data, training=True, label_transform=label_transform)
        test_ds = ProminenceDataset(test_data, 
                                    scalar_scaler=train_ds.scalar_scaler,
                                    frame_scaler=train_ds.frame_scaler, 
                                    training=False,
                                    label_transform=label_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate)
        
        output_dim = _target_to_output_dim(TARGET_MODE, CLASS_HEAD)
        class_weights = None
        ordinal_weights = None
        if TARGET_MODE != "regression" and CLASS_LOSS_WEIGHTING == "balanced":
            class_weights, class_counts = compute_class_weights(train_data, label_transform, TARGET_MODE)
            print(
                f"  {test_spk}: class weights={np.round(class_weights, 3).tolist()} "
                f"from counts={np.round(class_counts, 1).tolist()}"
            )
            ordinal_weights = class_weights_to_ordinal_weights(class_weights, TARGET_MODE)
        model = ProminencePredictor(
            FRAME_DIM,
            SCALAR_DIM,
            HIDDEN_DIM,
            output_dim=output_dim,
            target_mode=TARGET_MODE,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        mse_crit = nn.MSELoss(reduction='none')
        ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE) if class_weights is not None else None
        ce_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, weight=ce_weight)
        ordinal_weight_t = torch.tensor(ordinal_weights, dtype=torch.float32, device=DEVICE) if ordinal_weights is not None else None
        
        model.train()
        for epoch in range(EPOCHS):
            for frames, scalars, y, lens, _, _ in train_loader:
                frames, scalars, y = frames.to(DEVICE), scalars.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(frames, scalars, lens)
                mask = (y != -1) if y.ndim == 2 else (y[:, :, 0] != -1)

                if TARGET_MODE == "regression":
                    y_float = y.float()
                    mask_float = mask.float()
                    if USE_WEIGHTED_LOSS:
                        weights = 1.0 + (torch.clamp(y_float, min=0) * 2.0)
                        loss = (((outputs - y_float)**2) * weights * mask_float).sum() / mask_float.sum()
                    else:
                        loss = (mse_crit(outputs, y_float) * mask_float).sum() / mask_float.sum()
                else:
                    if CLASS_HEAD == "ordinal":
                        if CLASS_LABEL_SOURCE == "rater_soft":
                            y_ord = soft_targets_to_ordinal_targets(y.float(), TARGET_MODE)
                        else:
                            y_ord = class_targets_to_ordinal_targets(y.long(), TARGET_MODE)
                        per_head_loss = F.binary_cross_entropy_with_logits(outputs, y_ord, reduction="none")
                        if ordinal_weight_t is not None:
                            neg_w = ordinal_weight_t[:, 0].view(1, 1, -1)
                            pos_w = ordinal_weight_t[:, 1].view(1, 1, -1)
                            head_weights = (y_ord * pos_w) + ((1.0 - y_ord) * neg_w)
                            per_head_loss = per_head_loss * head_weights
                        per_token_loss = per_head_loss.mean(dim=-1)
                    else:
                        if CLASS_LABEL_SOURCE == "rater_soft":
                            y_prob = y.float()
                            log_probs = F.log_softmax(outputs, dim=-1)
                            if ce_weight is not None:
                                per_token_loss = -(y_prob * ce_weight.view(1, 1, -1) * log_probs).sum(dim=-1)
                            else:
                                per_token_loss = -(y_prob * log_probs).sum(dim=-1)
                        else:
                            per_token_loss = ce_crit(outputs.reshape(-1, output_dim), y.long().reshape(-1)).view_as(mask)
                    loss = (per_token_loss * mask.float()).sum() / mask.sum().float()
                
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []
        os.makedirs("plots", exist_ok=True)
        with torch.no_grad():
            for idx, (frames, scalars, y, lens, wavs, csvs) in enumerate(test_loader):
                frames, scalars = frames.to(DEVICE), scalars.to(DEVICE)
                outputs = model(frames, scalars, lens)
                curr_len = lens[0].item()
                if TARGET_MODE == "regression":
                    p_np = outputs[0, :curr_len].cpu().numpy()
                else:
                    if CLASS_HEAD == "ordinal":
                        probs_np = torch.sigmoid(outputs[0, :curr_len]).cpu().numpy()
                        p_np = ordinal_probs_to_classes(probs_np, TARGET_MODE, class_config)
                    else:
                        probs_np = torch.softmax(outputs[0, :curr_len], dim=-1).cpu().numpy()
                        if CLASS_LABEL_SOURCE == "rater_soft":
                            p_np = probs_to_classes(probs_np, TARGET_MODE, class_config)
                        else:
                            p_np = np.argmax(probs_np, axis=-1)
                if TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "rater_soft":
                    t_np = probs_to_classes(y[0, :curr_len].cpu().numpy(), TARGET_MODE, class_config)
                else:
                    t_np = y[0, :curr_len].cpu().numpy()
                all_preds.extend(p_np)
                all_targets.extend(t_np)
                if TARGET_MODE == "regression" and seed == SEEDS_TO_TEST[0] and idx == 1:
                    visualize_file_prominence(wavs[0], csvs[0], p_np, f"plots/{test_spk}_viz.png")
                #if seed == SEEDS_TO_TEST[0]:
                #    visualize_file_prominence(wavs[0], csvs[0], p_np, f"plots/{test_spk}_viz_{idx}.png")

        if TARGET_MODE == "regression":
            corr, _ = pearsonr(all_targets, all_preds)
            mse = np.mean((np.array(all_targets) - np.array(all_preds))**2)
            print(f"  {test_spk}: r={corr:.3f}, mse={mse:.4f}")
            overall_correlations.append(corr)
            mse_scores.append(mse)
        else:
            acc = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(all_targets, all_preds)
            print(f"  {test_spk}: acc={acc:.3f}, macro_f1={f1:.4f}, balanced_acc={bal_acc:.4f}")
            print(f"    target_dist: {_format_class_counts(all_targets, output_dim)}")
            print(f"    pred_dist:   {_format_class_counts(all_preds, output_dim)}")
            print(f"    per_class:   {_format_per_class_f1(all_targets, all_preds, output_dim)}")
            seed_class_targets.extend(all_targets)
            seed_class_preds.extend(all_preds)
            acc_scores.append(acc)
            f1_scores.append(f1)

    if TARGET_MODE == "regression":
        avg_corr = np.mean(overall_correlations)
        avg_mse = np.mean(mse_scores)
        print(f"Seed {seed} -> Avg r: {avg_corr:.4f} | Avg MSE: {avg_mse:.4f}")
        return avg_corr, avg_mse

    avg_acc = np.mean(acc_scores)
    avg_f1 = np.mean(f1_scores)
    print(f"Seed {seed} -> Avg acc: {avg_acc:.4f} | Avg macro F1: {avg_f1:.4f}")
    _print_confusion_matrix(
        seed_class_targets,
        seed_class_preds,
        _target_to_num_classes(TARGET_MODE),
        title=f"Seed {seed} aggregate confusion matrix",
    )
    _print_row_normalized_confusion_matrix(
        seed_class_targets,
        seed_class_preds,
        _target_to_num_classes(TARGET_MODE),
        title=f"Seed {seed} aggregate row-normalized confusion matrix",
    )
    print(
        f"Seed {seed} balanced accuracy: "
        f"{balanced_accuracy_score(seed_class_targets, seed_class_preds):.4f}"
    )
    return avg_acc, avg_f1, seed_class_targets, seed_class_preds

def train_full_model(seed, data_map, save_dir="models"):
    print(f"\n--- Training FULL model on ALL speakers (seed={seed}) ---")
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # combine all speakers
    all_data = []
    for spk in sorted(data_map.keys()):
        all_data.extend(data_map[spk])

    class_config = None
    label_transform = None
    if TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "mean":
        class_config = fit_class_config(
            _collect_labels(all_data),
            TARGET_MODE,
            method=CLASS_BOUNDARY_METHOD,
            seed=seed,
            min_count_per_class=THRESHOLD_MIN_COUNT_PER_CLASS,
        )
        print(f"Learned full-model thresholds: {class_config['thresholds']}")
    elif TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "rater_soft" and SOFT_THRESHOLD_METHOD != "fixed":
        objective = "macro_f1" if SOFT_THRESHOLD_METHOD == "opt_macro_f1" else "balanced_acc"
        class_config = optimize_soft_threshold_config(
            all_data,
            TARGET_MODE,
            RATER_COUNTS_BY_FILE,
            objective=objective,
            min_count_per_class=THRESHOLD_MIN_COUNT_PER_CLASS,
        )
        if TARGET_MODE == "binary":
            print(f"Optimized full-model soft threshold prom={class_config['prom_threshold']:.4f}")
        else:
            print(
                f"Optimized full-model soft thresholds "
                f"prom={class_config['prom_threshold']:.4f}, strong={class_config['strong_threshold']:.4f}"
            )
    elif TARGET_MODE != "regression":
        class_config = rater_class_config(TARGET_MODE, CLASS_LABEL_SOURCE)
    if TARGET_MODE != "regression":
        label_transform = make_label_transform(
            TARGET_MODE,
            class_config=class_config,
            rater_counts_by_file=RATER_COUNTS_BY_FILE,
            class_label_source=CLASS_LABEL_SOURCE,
        )

    # fit scalers on all data
    train_ds = ProminenceDataset(all_data, training=True, label_transform=label_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    output_dim = _target_to_output_dim(TARGET_MODE, CLASS_HEAD)
    class_weights = None
    ordinal_weights = None
    if TARGET_MODE != "regression" and CLASS_LOSS_WEIGHTING == "balanced":
        class_weights, class_counts = compute_class_weights(all_data, label_transform, TARGET_MODE)
        print(
            f"Full-model class weights={np.round(class_weights, 3).tolist()} "
            f"from counts={np.round(class_counts, 1).tolist()}"
        )
        ordinal_weights = class_weights_to_ordinal_weights(class_weights, TARGET_MODE)
    model = ProminencePredictor(
        FRAME_DIM,
        SCALAR_DIM,
        HIDDEN_DIM,
        output_dim=output_dim,
        target_mode=TARGET_MODE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_crit = nn.MSELoss(reduction='none')
    ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE) if class_weights is not None else None
    ce_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, weight=ce_weight)
    ordinal_weight_t = torch.tensor(ordinal_weights, dtype=torch.float32, device=DEVICE) if ordinal_weights is not None else None

    model.train()
    for epoch in range(EPOCHS):
        for frames, scalars, y, lens, _, _ in train_loader:
            frames, scalars, y = frames.to(DEVICE), scalars.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(frames, scalars, lens)
            mask = (y != -1) if y.ndim == 2 else (y[:, :, 0] != -1)

            if TARGET_MODE == "regression":
                y_float = y.float()
                mask_float = mask.float()
                if USE_WEIGHTED_LOSS:
                    weights = 1.0 + (torch.clamp(y_float, min=0) * 2.0)
                    loss = (((outputs - y_float)**2) * weights * mask_float).sum() / mask_float.sum()
                else:
                    loss = (mse_crit(outputs, y_float) * mask_float).sum() / mask_float.sum()
            else:
                if CLASS_HEAD == "ordinal":
                    if CLASS_LABEL_SOURCE == "rater_soft":
                        y_ord = soft_targets_to_ordinal_targets(y.float(), TARGET_MODE)
                    else:
                        y_ord = class_targets_to_ordinal_targets(y.long(), TARGET_MODE)
                    per_head_loss = F.binary_cross_entropy_with_logits(outputs, y_ord, reduction="none")
                    if ordinal_weight_t is not None:
                        neg_w = ordinal_weight_t[:, 0].view(1, 1, -1)
                        pos_w = ordinal_weight_t[:, 1].view(1, 1, -1)
                        head_weights = (y_ord * pos_w) + ((1.0 - y_ord) * neg_w)
                        per_head_loss = per_head_loss * head_weights
                    per_token_loss = per_head_loss.mean(dim=-1)
                else:
                    if CLASS_LABEL_SOURCE == "rater_soft":
                        y_prob = y.float()
                        log_probs = F.log_softmax(outputs, dim=-1)
                        if ce_weight is not None:
                            per_token_loss = -(y_prob * ce_weight.view(1, 1, -1) * log_probs).sum(dim=-1)
                        else:
                            per_token_loss = -(y_prob * log_probs).sum(dim=-1)
                    else:
                        per_token_loss = ce_crit(outputs.reshape(-1, output_dim), y.long().reshape(-1)).view_as(mask)
                loss = (per_token_loss * mask.float()).sum() / mask.sum().float()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} done")

    # save deployable checkpoint (model + scalers + config)
    extra_cfg = {
        "W2V_MODEL_NAME": W2V_MODEL_NAME,
        "USE_SSL": USE_SSL,
        "USE_RAW_PITCH": USE_RAW_PITCH,
        "USE_SCALARS": USE_SCALARS,
        "USE_PITCH_SHAPE": USE_PITCH_SHAPE,
        "USE_ATTENTION": USE_ATTENTION,
        "USE_MAX_POOLING": USE_MAX_POOLING,
        "USE_WEIGHTED_LOSS": USE_WEIGHTED_LOSS,
        "MAX_FRAMES_PER_WORD": MAX_FRAMES_PER_WORD,
        "FRAME_DIM": FRAME_DIM,
        "SCALAR_DIM": SCALAR_DIM,
        "HIDDEN_DIM": HIDDEN_DIM,
        "NUM_LAYERS": NUM_LAYERS,
        "DROPOUT": DROPOUT,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "TARGET_MODE": TARGET_MODE,
        "OUTPUT_DIM": output_dim,
        "CLASS_CONFIG": class_config,
        "CLASS_LABEL_SOURCE": CLASS_LABEL_SOURCE,
        "CLASS_HEAD": CLASS_HEAD,
        "CLASS_LOSS_WEIGHTING": CLASS_LOSS_WEIGHTING,
        "CLASS_WEIGHTS": None if class_weights is None else class_weights.tolist(),
        "SOFT_THRESHOLD_METHOD": SOFT_THRESHOLD_METHOD,
        "THRESHOLD_MIN_COUNT_PER_CLASS": THRESHOLD_MIN_COUNT_PER_CLASS,
        "seed": seed,
    }

    if TARGET_MODE == "regression":
        suffix = ""
    elif CLASS_LABEL_SOURCE == "mean":
        suffix = f"_{TARGET_MODE}_{CLASS_BOUNDARY_METHOD}"
    else:
        suffix = f"_{TARGET_MODE}_{CLASS_LABEL_SOURCE}"
    if TARGET_MODE != "regression" and CLASS_LABEL_SOURCE == "rater_soft" and SOFT_THRESHOLD_METHOD != "fixed":
        suffix += f"_{SOFT_THRESHOLD_METHOD}"
    if TARGET_MODE != "regression" and CLASS_HEAD != "softmax":
        suffix += f"_{CLASS_HEAD}"
    if TARGET_MODE != "regression" and CLASS_LOSS_WEIGHTING != "none":
        suffix += f"_{CLASS_LOSS_WEIGHTING}"
    out_path = os.path.join(save_dir, f"prom_model_full{suffix}_seed{seed}.pt")
    save_checkpoint(out_path, model, train_ds.scalar_scaler, train_ds.frame_scaler, extra_cfg)

    return out_path

def save_checkpoint(path, model, scalar_scaler, frame_scaler, extra_cfg: dict):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "scalar_scaler": scalar_scaler,
        "frame_scaler": frame_scaler,
        "config": extra_cfg,
    }
    torch.save(ckpt, path)
    print(f"[Saved] {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["loso", "all"], default="loso",
                        help="loso = leave-one-speaker-out eval, all = train final model on all speakers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional single seed (otherwise uses SEEDS_TO_TEST)")
    parser.add_argument("--no_ssl", action="store_true",
                        help="Train a PiSh-only scalar model without Wav2Vec/VoxRex frame embeddings")
    parser.add_argument("--target", choices=["regression", "binary", "ternary"], default="regression",
                        help="Prediction target: continuous regression, 0/1 prominence, or 0/1/2 prominence classes")
    parser.add_argument("--class_boundary_method", choices=["kmeans", "gmm", "round", "opt_macro_f1", "opt_balanced_acc"], default="kmeans",
                        help="How to convert continuous training ratings to classes for binary/ternary modes")
    parser.add_argument("--class_label_source", choices=["mean", "rater_majority", "rater_weighted_majority", "rater_soft"], default="mean",
                        help="Class labels from mean ratings, per-rater majority vote, 2-weighted majority vote, or per-rater soft distributions")
    parser.add_argument("--class_head", choices=["softmax", "ordinal"], default="softmax",
                        help="Classifier head: flat softmax classes or ordinal cumulative binary heads")
    parser.add_argument("--class_loss_weighting", choices=["none", "balanced"], default="none",
                        help="Use fold-local balanced class weights for categorical losses")
    parser.add_argument("--rater_file", default=None,
                        help="Optional per-rater label file, required for rater_majority or rater_soft class labels")
    parser.add_argument("--soft_binary_prom_threshold", type=float, default=0.25,
                        help="For rater_soft binary diagnostics/inference: class 1 if P(rating>0) is at least this value")
    parser.add_argument("--soft_ternary_prom_threshold", type=float, default=0.25,
                        help="For rater_soft ternary diagnostics/inference: class 1 if P(rating>0) is at least this value")
    parser.add_argument("--soft_ternary_strong_threshold", type=float, default=0.20,
                        help="For rater_soft ternary diagnostics/inference: class 2 if P(rating=2) is at least this value")
    parser.add_argument("--soft_threshold_method", choices=["fixed", "opt_macro_f1", "opt_balanced_acc"], default="fixed",
                        help="For rater_soft: use fixed proportion thresholds or optimize them on each training fold")
    parser.add_argument("--threshold_min_count_per_class", type=int, default=1,
                        help="Minimum train-fold examples required in each derived class when optimizing thresholds")
    args = parser.parse_args()

    USE_SSL = not args.no_ssl
    if not USE_SSL:
        USE_RAW_PITCH = False
    configure_feature_dims()

    TARGET_MODE = args.target
    CLASS_BOUNDARY_METHOD = args.class_boundary_method
    CLASS_LABEL_SOURCE = args.class_label_source
    CLASS_HEAD = args.class_head
    CLASS_LOSS_WEIGHTING = args.class_loss_weighting
    SOFT_BINARY_PROM_THRESHOLD = args.soft_binary_prom_threshold
    SOFT_TERNARY_PROM_THRESHOLD = args.soft_ternary_prom_threshold
    SOFT_TERNARY_STRONG_THRESHOLD = args.soft_ternary_strong_threshold
    SOFT_THRESHOLD_METHOD = args.soft_threshold_method
    THRESHOLD_MIN_COUNT_PER_CLASS = args.threshold_min_count_per_class

    if TARGET_MODE == "regression" and CLASS_LABEL_SOURCE != "mean":
        raise SystemExit("--class_label_source is only used with --target binary or --target ternary.")
    if TARGET_MODE == "regression" and CLASS_HEAD != "softmax":
        raise SystemExit("--class_head is only used with --target binary or --target ternary.")
    if TARGET_MODE == "regression" and CLASS_LOSS_WEIGHTING != "none":
        raise SystemExit("--class_loss_weighting is only used with --target binary or --target ternary.")
    if CLASS_LABEL_SOURCE in ("rater_majority", "rater_weighted_majority", "rater_soft") and not args.rater_file:
        raise SystemExit("--rater_file is required when using per-rater class labels.")
    if CLASS_LABEL_SOURCE == "mean" and args.rater_file:
        print("[Info] --rater_file was provided but --class_label_source=mean, so per-rater labels are not used.")

    if args.rater_file:
        RATER_COUNTS_BY_FILE = load_rater_counts(args.rater_file, DATA_ROOT)

    print("\nEffective configuration")
    print("-" * 30)
    print(f"Model: {W2V_MODEL_NAME if USE_SSL else 'PiSh-only (no SSL backbone)'}")
    print(f"FEATS: SSL={USE_SSL}, Shape={USE_PITCH_SHAPE}, RawPitch={USE_RAW_PITCH}")
    print(f"ARCH:  Attn={USE_ATTENTION if USE_SSL else False}, Max={USE_MAX_POOLING if USE_SSL else False}, Weighted={USE_WEIGHTED_LOSS}")
    print("-" * 30)

    data_map = precompute_data(DATA_ROOT)

    if not data_map:
        raise SystemExit("No data found.")

    seeds = [args.seed] if args.seed is not None else SEEDS_TO_TEST

    if args.mode == "loso":
        primary_scores = []
        secondary_scores = []
        final_class_targets = []
        final_class_preds = []
        for s in seeds:
            result = run_seed_experiment(s, data_map)
            if TARGET_MODE == "regression":
                primary, secondary = result
            else:
                primary, secondary, class_targets, class_preds = result
                final_class_targets.extend(class_targets)
                final_class_preds.extend(class_preds)
            primary_scores.append(primary)
            secondary_scores.append(secondary)

        print("\n" + "="*30)
        print(f"FINAL RESULTS ({EPOCHS} Epochs)")
        print(f"Feature Fusion={USE_PITCH_SHAPE}, Enhanced Head & Optimization={USE_WEIGHTED_LOSS}")
        if TARGET_MODE == "regression":
            print(f"Correlation: {np.mean(primary_scores):.4f} (std {np.std(primary_scores):.4f})")
            print(f"MSE:         {np.mean(secondary_scores):.4f} (std {np.std(secondary_scores):.4f})")
        else:
            if CLASS_LABEL_SOURCE == "mean":
                label_desc = f"{CLASS_BOUNDARY_METHOD} boundaries"
            else:
                label_desc = CLASS_LABEL_SOURCE
                if CLASS_LABEL_SOURCE == "rater_soft" and SOFT_THRESHOLD_METHOD != "fixed":
                    label_desc += f", {SOFT_THRESHOLD_METHOD} thresholds"
            if CLASS_HEAD != "softmax":
                label_desc += f", {CLASS_HEAD} head"
            if CLASS_LOSS_WEIGHTING != "none":
                label_desc += f", {CLASS_LOSS_WEIGHTING} loss"
            print(f"Target:      {TARGET_MODE} ({label_desc})")
            print(f"Accuracy:    {np.mean(primary_scores):.4f} (std {np.std(primary_scores):.4f})")
            print(f"Macro F1:    {np.mean(secondary_scores):.4f} (std {np.std(secondary_scores):.4f})")
            print(f"Balanced Acc:{balanced_accuracy_score(final_class_targets, final_class_preds):.4f}")
            _print_confusion_matrix(
                final_class_targets,
                final_class_preds,
                _target_to_num_classes(TARGET_MODE),
                title="FINAL aggregate confusion matrix",
            )
            _print_row_normalized_confusion_matrix(
                final_class_targets,
                final_class_preds,
                _target_to_num_classes(TARGET_MODE),
                title="FINAL aggregate row-normalized confusion matrix",
            )
        print("="*30)

    elif args.mode == "all":
        # Train one deployable model (or several if multiple seeds)
        for s in seeds:
            train_full_model(s, data_map, save_dir="models")
