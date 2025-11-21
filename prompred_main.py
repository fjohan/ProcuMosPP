import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
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

print(f"Running on: {DEVICE}")
print(f"Model: {W2V_MODEL_NAME}")
print("-" * 30)
print(f"FEATS: Shape={USE_PITCH_SHAPE}, RawPitch={USE_RAW_PITCH}")
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
    if w2v_model is None:
        print(f"Loading Wav2Vec2 Model...")
        processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
        w2v_model = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME).to(DEVICE)
        w2v_model.eval()

from transformers import AutoConfig
config = AutoConfig.from_pretrained(W2V_MODEL_NAME)
W2V_DIM = config.hidden_size
FRAME_DIM = W2V_DIM + 1 if USE_RAW_PITCH else W2V_DIM

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
    inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = w2v_model(**inputs)
    w2v_frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    
    # Acoustic Extraction
    HOP = 320 
    f0_raw, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, frame_length=1024, hop_length=HOP)
    
    # Note: We keep NaNs in f0_raw for Shape interpolation logic
    f0_frames = np.nan_to_num(f0_raw).reshape(-1, 1) 
    
    rms_raw = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    cent_raw = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=HOP)[0]
    
    min_len = min(len(w2v_frames), len(f0_frames), len(rms_raw), len(cent_raw))
    
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
        curr_w2v = w2v_frames[start_idx:end_idx][:min_len]
        if USE_RAW_PITCH:
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
    return f"feats_{safe_model}_shape{USE_PITCH_SHAPE}_raw{USE_RAW_PITCH}_scalars{USE_SCALARS}.pt"

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
    def __init__(self, data_list, scalar_scaler=None, frame_scaler=None, training=True):
        self.data = data_list 
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
    def __init__(self, frame_dim, scalar_dim, hidden_dim=64):
        super().__init__()
        self.pooling = ConfigurablePooling(frame_dim)
        pooling_multiplier = 1 + (1 if USE_MAX_POOLING else 0)
        lstm_input_dim = (frame_dim * pooling_multiplier) + scalar_dim
        
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, 
                            num_layers=NUM_LAYERS, 
                            batch_first=True, 
                            bidirectional=True,
                            dropout=DROPOUT if NUM_LAYERS > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.activation = nn.Sigmoid() 

    def forward(self, frames, scalars, lengths):
        batch_size, seq_len, num_frames, feat_dim = frames.shape
        
        flat_frames = frames.view(-1, num_frames, feat_dim)
        pooled_feats = self.pooling(flat_frames) 
        word_embeddings = pooled_feats.view(batch_size, seq_len, -1)
        
        lstm_input = torch.cat([word_embeddings, scalars], dim=2) 
        packed_x = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        return (self.activation(self.fc(out)) * 3.0).squeeze(-1)

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

    for test_spk in speakers:
        train_data = []
        test_data = data_map[test_spk]
        for spk in speakers:
            if spk != test_spk: train_data.extend(data_map[spk])
            
        train_ds = ProminenceDataset(train_data, training=True)
        test_ds = ProminenceDataset(test_data, 
                                    scalar_scaler=train_ds.scalar_scaler,
                                    frame_scaler=train_ds.frame_scaler, 
                                    training=False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate)
        
        model = ProminencePredictor(FRAME_DIM, SCALAR_DIM, HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        mse_crit = nn.MSELoss(reduction='none')
        
        model.train()
        for epoch in range(EPOCHS):
            for frames, scalars, y, lens, _, _ in train_loader:
                frames, scalars, y = frames.to(DEVICE), scalars.to(DEVICE), y.to(DEVICE).float()
                optimizer.zero_grad()
                preds = model(frames, scalars, lens)
                mask = (y != -1).float()
                
                if USE_WEIGHTED_LOSS:
                    weights = 1.0 + (torch.clamp(y, min=0) * 2.0)
                    loss = (((preds - y)**2) * weights * mask).sum() / mask.sum()
                else:
                    loss = (mse_crit(preds, y) * mask).sum() / mask.sum()
                
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []
        os.makedirs("plots", exist_ok=True)
        with torch.no_grad():
            for idx, (frames, scalars, y, lens, wavs, csvs) in enumerate(test_loader):
                frames, scalars = frames.to(DEVICE), scalars.to(DEVICE)
                preds = model(frames, scalars, lens)
                curr_len = lens[0].item()
                p_np = preds[0, :curr_len].cpu().numpy()
                t_np = y[0, :curr_len].cpu().numpy()
                all_preds.extend(p_np)
                all_targets.extend(t_np)
                if seed == SEEDS_TO_TEST[0] and idx == 1:
                    visualize_file_prominence(wavs[0], csvs[0], p_np, f"plots/{test_spk}_viz.png")

        corr, _ = pearsonr(all_targets, all_preds)
        mse = np.mean((np.array(all_targets) - np.array(all_preds))**2)
        print(f"  {test_spk}: r={corr:.3f}, mse={mse:.4f}")
        overall_correlations.append(corr)
        mse_scores.append(mse)

    avg_corr = np.mean(overall_correlations)
    avg_mse = np.mean(mse_scores)
    print(f"Seed {seed} -> Avg r: {avg_corr:.4f} | Avg MSE: {avg_mse:.4f}")
    return avg_corr, avg_mse

if __name__ == '__main__':
    data_map = precompute_data(DATA_ROOT)
    corrs = []
    mses = []
    if data_map:
        for s in SEEDS_TO_TEST:
            c, m = run_seed_experiment(s, data_map)
            corrs.append(c)
            mses.append(m)
        print("\n" + "="*30)
        print(f"FINAL RESULTS ({EPOCHS} Epochs)")
        print(f"Feature Fusion={USE_PITCH_SHAPE}, Enhanced Head & Optimization={USE_WEIGHTED_LOSS}")
        print(f"Correlation: {np.mean(corrs):.4f} (std {np.std(corrs):.4f})")
        print(f"MSE:         {np.mean(mses):.4f} (std {np.std(mses):.4f})")
        print("="*30)

