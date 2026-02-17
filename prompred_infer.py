import os
import argparse
import math
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoConfig

# ------------------------------------------------------------
# Model definitions (PARAMETERIZED, not using globals)
# ------------------------------------------------------------
class ConfigurablePooling(nn.Module):
    def __init__(self, input_dim, use_attn=True, use_max=True):
        super().__init__()
        self.use_attn = use_attn
        self.use_max = use_max
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
    def __init__(
        self,
        frame_dim,
        scalar_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3,
        use_attn=True,
        use_max=True,
        output_scale=3.0,
    ):
        super().__init__()
        self.pooling = ConfigurablePooling(frame_dim, use_attn=use_attn, use_max=use_max)
        pooling_multiplier = 1 + (1 if use_max else 0)
        lstm_input_dim = (frame_dim * pooling_multiplier) + scalar_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.activation = nn.Sigmoid()
        self.output_scale = float(output_scale)

    def forward(self, frames, scalars, lengths):
        """
        frames:  [B, SeqLen, MaxFrames, FrameDim]
        scalars: [B, SeqLen, ScalarDim]
        lengths: [B] (number of tokens/windows)
        """
        B, S, T, F = frames.shape
        flat_frames = frames.view(-1, T, F)              # [B*S, T, F]
        pooled = self.pooling(flat_frames)               # [B*S, pooled_dim]
        word_emb = pooled.view(B, S, -1)                 # [B, S, pooled_dim]
        x = torch.cat([word_emb, scalars], dim=2)        # [B, S, pooled_dim+scalar_dim]

        # Pack/pad (optional). For inference, simplest: just run as-is if you want.
        # But we keep consistent with training, using pack.
        lengths_cpu = lengths.detach().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        y = self.activation(self.fc(out)) * self.output_scale
        return y.squeeze(-1)  # [B, S]

# ------------------------------------------------------------
# Pitch shape feature
# ------------------------------------------------------------
def get_pitch_shape_coeffs(f0_seq):
    valid_mask = ~np.isnan(f0_seq)
    if np.sum(valid_mask) < 3:
        return [0.0, 0.0, 0.0, 0.0]

    idx = np.arange(len(f0_seq))
    valid_idx = idx[valid_mask]
    valid_vals = f0_seq[valid_mask]

    interp_func = interp1d(valid_idx, valid_vals, kind="linear", fill_value="extrapolate")
    f0_interp = interp_func(idx)

    t_norm = np.linspace(-1, 1, len(f0_interp))
    coeffs, residuals, _, _, _ = np.polyfit(t_norm, f0_interp, 2, full=True)
    mse_error = (residuals[0] / len(f0_interp)) if len(residuals) > 0 else 0.0

    return [float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(mse_error)]

# ------------------------------------------------------------
# Praat writers
# ------------------------------------------------------------
def write_praat_sound(frame_values, dx, sound_path):
    nx = len(frame_values)
    xmin = 0.0
    xmax = nx * dx
    x1 = dx / 2.0

    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "Sound 2"')
    lines.append("")
    lines.append(f"xmin = {xmin}")
    lines.append(f"xmax = {xmax}")
    lines.append(f"nx = {nx}")
    lines.append(f"dx = {dx}")
    lines.append(f"x1 = {x1}")
    lines.append("ymin = 1")
    lines.append("ymax = 1")
    lines.append("ny = 1")
    lines.append("dy = 1")
    lines.append("y1 = 1")
    lines.append("z [] []:")
    lines.append("    z [1]:")
    for i, v in enumerate(frame_values, start=1):
        lines.append(f"        z [1] [{i}] = {float(v)}")

    with open(sound_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_textgrid_interval_tier(intervals, tg_path, xmax, tier_name="labels"):
    """
    intervals: list of (start,end,text), must be non-overlapping. We will fill gaps with empty intervals.
    """
    intervals = sorted(intervals, key=lambda x: x[0])
    filled = []
    cur = 0.0
    for s, e, txt in intervals:
        s = float(s); e = float(e)
        if s > cur:
            filled.append((cur, s, ""))
        filled.append((s, e, str(txt).replace('"', '""')))
        cur = e
    if cur < xmax:
        filled.append((cur, xmax, ""))

    n = len(filled)
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append("")
    lines.append("xmin = 0")
    lines.append(f"xmax = {xmax}")
    lines.append("tiers? <exists>")
    lines.append("size = 1")
    lines.append("item []:")
    lines.append("    item [1]:")
    lines.append('        class = "IntervalTier"')
    lines.append(f'        name = "{tier_name}"')
    lines.append("        xmin = 0")
    lines.append(f"        xmax = {xmax}")
    lines.append(f"        intervals: size = {n}")

    for i, (s, e, txt) in enumerate(filled, start=1):
        lines.append(f"        intervals [{i}]:")
        lines.append(f"            xmin = {s}")
        lines.append(f"            xmax = {e}")
        lines.append(f'            text = "{txt}"')

    with open(tg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_textgrid_point_tier(points, tg_path, xmax, tier_name="segments"):
    """
    points: list of (time, label)
    Useful when windows overlap (IntervalTier can't represent overlaps).
    """
    points = sorted(points, key=lambda x: x[0])
    n = len(points)
    lines = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append("")
    lines.append("xmin = 0")
    lines.append(f"xmax = {xmax}")
    lines.append("tiers? <exists>")
    lines.append("size = 1")
    lines.append("item []:")
    lines.append("    item [1]:")
    lines.append('        class = "TextTier"')
    lines.append(f'        name = "{tier_name}"')
    lines.append("        xmin = 0")
    lines.append(f"        xmax = {xmax}")
    lines.append(f"        points: size = {n}")

    for i, (t, lab) in enumerate(points, start=1):
        lab = str(lab).replace('"', '""')
        lines.append(f"        points [{i}]:")
        lines.append(f"            number = {float(t)}")
        lines.append(f'            mark = "{lab}"')

    with open(tg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ------------------------------------------------------------
# Feature extraction for arbitrary segments
# ------------------------------------------------------------
def extract_segment_features(
    wav_path,
    segments,
    processor,
    w2v_model,
    use_raw_pitch,
    use_pitch_shape,
    use_scalars,
    max_frames_per_word,
    device,
):
    """
    segments: list of (start,end,label,obs_rating_or_None)
    returns:
      frames_arr:  [N, max_frames_per_word, frame_dim]
      scalars_arr: [N, scalar_dim]
      labels:      list of labels (strings)
      starts, ends lists
      obs_ratings  list (float or np.nan)
      T_frames: total w2v frames in file
    """

    y, sr = librosa.load(wav_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # W2V frames
    inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = w2v_model(**inputs)
    w2v_frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, D]
    T_total = w2v_frames.shape[0]

    # Assume ~50 fps for wav2vec2 (20ms). Keep consistent with your training.
    MODEL_FPS = 50.0
    FRAME_DUR = 1.0 / MODEL_FPS

    # Acoustic features aligned to same hop (HOP=320 @16kHz => 50 fps)
    HOP = 320
    f0_raw, _, _ = librosa.pyin(
        y, fmin=50, fmax=600, sr=sr, frame_length=1024, hop_length=HOP
    )
    rms_raw = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    cent_raw = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=HOP)[0]

    min_len = min(len(w2v_frames), len(f0_raw), len(rms_raw), len(cent_raw))
    w2v_frames = w2v_frames[:min_len]
    f0_raw = f0_raw[:min_len]
    rms_raw = rms_raw[:min_len]
    cent_raw = cent_raw[:min_len]

    frame_dim = w2v_frames.shape[1] + (1 if use_raw_pitch else 0)

    frames_out = []
    scalars_out = []
    starts_out, ends_out = [], []
    labels_out = []
    obs_out = []

    for (start, end, label, obs) in segments:
        start = max(0.0, float(start))
        end = min(duration, float(end))
        if end <= start:
            continue

        si = int(start * MODEL_FPS)
        ei = int(end * MODEL_FPS)

        if si >= min_len:
            continue
        if ei > min_len:
            ei = min_len
        if si >= ei:
            si = min(si, min_len - 1)
            ei = si + 1

        # Frames slice
        curr_w2v = w2v_frames[si:ei]
        if use_raw_pitch:
            f0_seg = f0_raw[si:ei]
            f0_num = np.nan_to_num(f0_seg).reshape(-1, 1)
            curr_frames = np.concatenate([curr_w2v, f0_num], axis=1)
        else:
            curr_frames = curr_w2v

        # Pad/crop to fixed length
        if len(curr_frames) > max_frames_per_word:
            s = (len(curr_frames) - max_frames_per_word) // 2
            curr_frames = curr_frames[s : s + max_frames_per_word]
        else:
            pad = max_frames_per_word - len(curr_frames)
            curr_frames = np.pad(curr_frames, ((0, pad), (0, 0)), mode="constant")

        # Scalars
        dur = end - start
        log_dur = math.log(dur + 1e-6)

        rms_seg = rms_raw[si:ei]
        cent_seg = cent_raw[si:ei]
        rms_mean = float(np.mean(rms_seg)) if len(rms_seg) else 0.0
        rms_max = float(np.max(rms_seg)) if len(rms_seg) else 0.0
        cent_mean = float(np.mean(cent_seg)) if len(cent_seg) else 0.0

        # Pitch features
        f0_seg = f0_raw[si:ei]
        if use_pitch_shape:
            pitch_feats = get_pitch_shape_coeffs(f0_seg)
        else:
            if len(f0_seg) == 0 or np.all(np.isnan(f0_seg)):
                pitch_feats = [0.0, 0.0, 0.0]
            else:
                pitch_feats = [
                    float(np.nanmean(f0_seg)),
                    float(np.nanstd(f0_seg)),
                    float(np.nanmax(f0_seg) - np.nanmin(f0_seg)),
                ]

        if use_scalars:
            scalar_vec = [log_dur, rms_mean, rms_max, cent_mean] + pitch_feats
        else:
            scalar_vec = [log_dur]  # minimal

        frames_out.append(curr_frames.astype(np.float32))
        scalars_out.append(np.array(scalar_vec, dtype=np.float32))
        starts_out.append(start)
        ends_out.append(end)
        labels_out.append(str(label))
        obs_out.append(np.nan if obs is None else float(obs))

    return (
        np.array(frames_out, dtype=np.float32),
        np.array(scalars_out, dtype=np.float32),
        labels_out,
        np.array(starts_out, dtype=np.float32),
        np.array(ends_out, dtype=np.float32),
        np.array(obs_out, dtype=np.float32),
        min_len,
        FRAME_DUR,
    )

# ------------------------------------------------------------
# Build segments
# ------------------------------------------------------------
def segments_from_csv(csv_path):
    """
    Robust CSV loader.

    Accepts:
      - With header:
          start,end,word,rating
          start_time,end_time,word
          start_time,end_time,word,rating
      - Without header:
          start,end,word,rating
          start,end,word
    """

    # First try reading with header detection
    df = pd.read_csv(csv_path)

    # Normalize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Case 1: header exists and uses start_time / end_time
    if "start_time" in df.columns and "end_time" in df.columns:
        df = df.rename(columns={
            "start_time": "start",
            "end_time": "end"
        })

    # Case 2: header exists and uses start / end already
    if "start" in df.columns and "end" in df.columns and "word" in df.columns:
        # rating may or may not exist
        if "rating" not in df.columns:
            df["rating"] = np.nan
        segs = []
        for _, r in df.iterrows():
            segs.append((
                float(r["start"]),
                float(r["end"]),
                str(r["word"]),
                None if pd.isna(r["rating"]) else float(r["rating"])
            ))
        return segs, df

    # If we reach here, likely no header → reload without header
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] == 4:
        df.columns = ["start", "end", "word", "rating"]
    elif df.shape[1] == 3:
        df.columns = ["start", "end", "word"]
        df["rating"] = np.nan
    else:
        raise ValueError("CSV format not recognized. Expect 3 or 4 columns.")

    segs = []
    for _, r in df.iterrows():
        segs.append((
            float(r["start"]),
            float(r["end"]),
            str(r["word"]),
            None if pd.isna(r["rating"]) else float(r["rating"])
        ))

    return segs, df

def _segments_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["start", "end", "word", "rating"])
    segs = []
    for _, r in df.iterrows():
        segs.append((float(r["start"]), float(r["end"]), str(r["word"]), float(r["rating"])))
    return segs, df

def segments_from_windows(duration, interval=0.4, overlap=0.1):
    step = max(1e-6, interval - overlap)
    segs = []
    t = 0.0
    k = 1
    while t < duration:
        s = t
        e = min(duration, t + interval)
        segs.append((s, e, str(k), None))
        k += 1
        t += step
        if e >= duration:
            break
    return segs

def _find_matching_csv_for_wav(wav_path):
    base_no_ext = os.path.splitext(wav_path)[0]
    csv_path = base_no_ext + ".csv"
    if os.path.exists(csv_path):
        return csv_path

    # Fallback for case variants like .CSV
    folder = os.path.dirname(wav_path) or "."
    target = os.path.basename(base_no_ext).lower()
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if stem.lower() == target and ext.lower() == ".csv":
            return os.path.join(folder, name)
    return None

def discover_wav_csv_pairs(input_dir):
    pairs = []
    missing_csv = []

    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(root, name)
            csv_path = _find_matching_csv_for_wav(wav_path)
            if csv_path is None:
                missing_csv.append(wav_path)
            else:
                pairs.append((wav_path, csv_path))

    pairs.sort()
    missing_csv.sort()
    return pairs, missing_csv

def run_inference_for_file(
    wav_path,
    csv_path,
    args,
    processor,
    w2v_model,
    model,
    scalar_scaler,
    frame_scaler,
    use_raw_pitch,
    use_pitch_shape,
    use_scalars,
    max_frames_per_word,
    scalar_dim,
    device,
    out_csv_override=None,
    inplace=False,
    batch_mode=False,
):
    # ----- Build segments -----
    if csv_path:
        segs, _ = segments_from_csv(csv_path)
        mode = "csv"
    else:
        duration = float(librosa.get_duration(path=wav_path))
        segs = segments_from_windows(duration, args.interval, args.overlap)
        mode = "windows"

    # ----- Extract features for segments -----
    frames_arr, scalars_arr, labels, starts, ends, obs, T_total, frame_dur = extract_segment_features(
        wav_path,
        segs,
        processor,
        w2v_model,
        use_raw_pitch,
        use_pitch_shape,
        use_scalars,
        max_frames_per_word,
        device,
    )

    if len(labels) == 0:
        raise RuntimeError(f"No valid segments extracted: {wav_path}")

    # ----- Normalize (same as training) -----
    scalars_arr = scalars_arr[:, :scalar_dim]
    scalars_norm = scalar_scaler.transform(scalars_arr).astype(np.float32)

    frames_norm = frames_arr.copy()
    if use_raw_pitch and frame_scaler is not None:
        n_items, n_frames, _ = frames_norm.shape
        frames_norm[:, :, -1] = frame_scaler.transform(
            frames_norm[:, :, -1].reshape(-1, 1)
        ).reshape(n_items, n_frames)

    # ----- Build tensors for model -----
    frames_t = torch.tensor(frames_norm, dtype=torch.float32).unsqueeze(0).to(device)
    scalars_t = torch.tensor(scalars_norm, dtype=torch.float32).unsqueeze(0).to(device)
    lengths_t = torch.tensor([frames_norm.shape[0]], dtype=torch.long).to(device)

    # ----- Predict -----
    with torch.no_grad():
        preds = model(frames_t, scalars_t, lengths_t).squeeze(0).cpu().numpy()

    # ----- Output CSV -----
    if inplace:
        if mode != "csv":
            raise RuntimeError("--inplace requires CSV-based inference.")
        out_csv = csv_path
        out_df = pd.DataFrame({
            "start": starts,
            "end": ends,
            "word": labels,
            "predicted_rating": preds,
        })
    else:
        out_csv = out_csv_override
        if out_csv is None:
            out_csv = f"{os.path.splitext(wav_path)[0]}_pred.csv"
        out_df = pd.DataFrame({
            "start": starts,
            "end": ends,
            "label": labels,
            "pred": preds,
        })
        if mode == "csv":
            out_df["obs"] = obs

    out_df.to_csv(out_csv, index=False, header=not args.no_header)
    print(f"[Wrote] {out_csv}")

    # ----- Build full frame curve (for Praat Sound) -----
    acc = np.zeros((T_total,), dtype=np.float32)
    cnt = np.zeros((T_total,), dtype=np.float32)

    model_fps = 50.0
    for s, e, p in zip(starts, ends, preds):
        i0 = int(float(s) * model_fps)
        i1 = int(float(e) * model_fps)
        i0 = max(0, min(i0, T_total))
        i1 = max(0, min(i1, T_total))
        if i1 <= i0:
            i1 = min(T_total, i0 + 1)
        acc[i0:i1] += float(p)
        cnt[i0:i1] += 1.0

    curve = np.where(cnt > 0, acc / cnt, 0.0)

    # ----- Praat output -----
    if args.praat:
        wavroot = os.path.splitext(wav_path)[0]
        if args.out_prefix:
            prefix = args.out_prefix
            if batch_mode:
                prefix = f"{args.out_prefix}_{os.path.basename(wavroot)}"
        else:
            prefix = wavroot
        xmax = float(T_total) * frame_dur

        tg_path = f"{prefix}_pred.TextGrid"
        snd_path = f"{prefix}_prom.Sound"

        if mode == "csv":
            intervals = [(s, e, lab) for s, e, lab in zip(starts, ends, labels)]
            write_textgrid_interval_tier(intervals, tg_path, xmax, tier_name="words")
        else:
            points = [((float(s) + float(e)) / 2.0, lab) for s, e, lab in zip(starts, ends, labels)]
            write_textgrid_point_tier(points, tg_path, xmax, tier_name="segments")

        write_praat_sound(curve.tolist(), frame_dur, snd_path)
        print(f"[Wrote] {tg_path}")
        print(f"[Wrote] {snd_path}")

# ------------------------------------------------------------
# Main inference
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Saved .pt checkpoint from training (--mode all)")
    ap.add_argument("--wav", default=None, help="Single wav file")
    ap.add_argument("--input_dir", default=None, help="Recursively process wav/csv pairs under this directory")
    ap.add_argument("--csv", default=None, help="Optional CSV: start,end,word,rating (no header)")
    ap.add_argument("--interval", type=float, default=0.4, help="Window length if no CSV")
    ap.add_argument("--overlap", type=float, default=0.1, help="Window overlap if no CSV")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (default: <wavbase>_pred.csv)")
    ap.add_argument("--inplace", action="store_true", help="Directory mode only: overwrite input CSVs with start,end,word,predicted_rating")
    ap.add_argument("--no_header", action="store_true", help="Write output CSV without a header row")
    ap.add_argument("--praat", action="store_true", help="Write Praat TextGrid + prominence Sound")
    ap.add_argument("--out_prefix", default=None, help="Output prefix for Praat files (default: <wavbase>)")
    args = ap.parse_args()

    if bool(args.wav) == bool(args.input_dir):
        raise SystemExit("Specify exactly one of --wav or --input_dir.")
    if args.inplace and not args.input_dir:
        raise SystemExit("--inplace is only supported with --input_dir.")
    if args.input_dir and args.csv:
        raise SystemExit("--csv is only supported with --wav.")
    if args.input_dir and args.out_csv:
        raise SystemExit("--out_csv is only supported with --wav.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load checkpoint -----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    W2V_MODEL_NAME = cfg.get("W2V_MODEL_NAME", "facebook/wav2vec2-base-960h")
    USE_RAW_PITCH = bool(cfg.get("USE_RAW_PITCH", False))
    USE_PITCH_SHAPE = bool(cfg.get("USE_PITCH_SHAPE", True))
    USE_SCALARS = bool(cfg.get("USE_SCALARS", True))
    USE_ATTENTION = bool(cfg.get("USE_ATTENTION", True))
    USE_MAX_POOLING = bool(cfg.get("USE_MAX_POOLING", True))

    MAX_FRAMES_PER_WORD = int(cfg.get("MAX_FRAMES_PER_WORD", 50))
    FRAME_DIM = int(cfg.get("FRAME_DIM", AutoConfig.from_pretrained(W2V_MODEL_NAME).hidden_size + (1 if USE_RAW_PITCH else 0)))
    SCALAR_DIM = int(cfg.get("SCALAR_DIM", 8))
    HIDDEN_DIM = int(cfg.get("HIDDEN_DIM", 64))
    NUM_LAYERS = int(cfg.get("NUM_LAYERS", 1))
    DROPOUT = float(cfg.get("DROPOUT", 0.3))
    OUTPUT_SCALE = float(cfg.get("OUTPUT_SCALE", cfg.get("output_scale", 3.0)))  # default matches your current training

    scalar_scaler = ckpt["scalar_scaler"]
    frame_scaler = ckpt.get("frame_scaler", None)

    # ----- Load wav2vec2 -----
    print(f"[Info] Loading W2V: {W2V_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME).to(device)
    w2v_model.eval()

    # ----- Build model and load weights -----
    model = ProminencePredictor(
        frame_dim=FRAME_DIM,
        scalar_dim=SCALAR_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attn=USE_ATTENTION,
        use_max=USE_MAX_POOLING,
        output_scale=OUTPUT_SCALE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.input_dir:
        pairs, missing_csv = discover_wav_csv_pairs(args.input_dir)
        if len(pairs) == 0:
            raise SystemExit(f"No wav/csv pairs found under: {args.input_dir}")

        print(f"[Info] Found {len(pairs)} wav/csv pairs under {args.input_dir}")
        if missing_csv:
            print(f"[Info] Skipping {len(missing_csv)} wav file(s) without matching CSV.")
            for path in missing_csv:
                print(f"  [Skip] {path}")

        if args.inplace:
            print("[Info] --inplace enabled: matched CSV files will be overwritten.")

        ok = 0
        failed = 0
        for wav_path, csv_path in pairs:
            try:
                run_inference_for_file(
                    wav_path=wav_path,
                    csv_path=csv_path,
                    args=args,
                    processor=processor,
                    w2v_model=w2v_model,
                    model=model,
                    scalar_scaler=scalar_scaler,
                    frame_scaler=frame_scaler,
                    use_raw_pitch=USE_RAW_PITCH,
                    use_pitch_shape=USE_PITCH_SHAPE,
                    use_scalars=USE_SCALARS,
                    max_frames_per_word=MAX_FRAMES_PER_WORD,
                    scalar_dim=SCALAR_DIM,
                    device=device,
                    out_csv_override=None,
                    inplace=args.inplace,
                    batch_mode=True,
                )
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"[Error] {wav_path}: {exc}")

        print(f"[Summary] processed={ok}, failed={failed}, skipped_no_csv={len(missing_csv)}")
    else:
        run_inference_for_file(
            wav_path=args.wav,
            csv_path=args.csv,
            args=args,
            processor=processor,
            w2v_model=w2v_model,
            model=model,
            scalar_scaler=scalar_scaler,
            frame_scaler=frame_scaler,
            use_raw_pitch=USE_RAW_PITCH,
            use_pitch_shape=USE_PITCH_SHAPE,
            use_scalars=USE_SCALARS,
            max_frames_per_word=MAX_FRAMES_PER_WORD,
            scalar_dim=SCALAR_DIM,
            device=device,
            out_csv_override=args.out_csv,
            inplace=False,
            batch_mode=False,
        )

if __name__ == "__main__":
    main()
