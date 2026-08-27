import time
import pydicom
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from main_train_acep_v3_aug import get_model

torch.set_num_threads(2)  # Limit CPU threads to mimic edge device constraints
 
# mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, resnet50, densenet121, convnext_tiny
MODEL_NAME  = "mobilenet_v3_large"
NUM_CLASSES = 2
PRETRAINED  = True
 
# mobilenetv2, convnext-tiny, ResNet-50, MNv3-small, MNv3-large, EfficientNet-B0, densenet
MODEL_DIR        = Path("results/XxBenchmarkedModelsxX/xXModels_PreTrainedXx/MNv3-large")
FOLD_WEIGHT_GLOB = "best_fold_*.pth"

CENTRES = None
DVT_FOLDER_NAMES = {"DVT", "NON_DVT", "NONDVT", "NON-DVT"}
DICOM_SUFFIXES = {".dcm", ".DCM"}
INCLUDE_EXTENSIONLESS = False
 
# Source of the DICOMs to sweep (with file paths from .csv)
RESTRICT_TO_CSV   = True
METADATA_CSV      = Path("experiment1_dataset/external_test_metadata.csv")
DICOM_PATH_COLUMN = "dicom_path"
DICOM_ROOT        = Path("Dataset[Original]")
DICOM_GLOB        = "**/*.dcm"
 
MAX_DICOMS   = None
WARMUP_ITERS = 25               # GPU warmup cycle per fold before benchmarking
IMAGE_SIZE   = 224
 
OUT_PER_RUN_CSV = Path("results/wall_clock_per_dicom.csv")   # one row per (fold, dicom)
OUT_SUMMARY_CSV = Path("results/wall_clock_summary.csv")     # mean/std/min/max, edge cases
 
 
 
def build_transform():
    """preprocessing (resize 224, imageNet normalisation)"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
 
def normalise_dvt_status(folder_name):
    return "DVT" if folder_name.upper() == "DVT" else "nonDVT"
 
def is_dicom_file(p):
    if not p.is_file():
        return False
    if p.suffix in DICOM_SUFFIXES:
        return True
    return INCLUDE_EXTENSIONLESS and p.suffix == ""
 
def discover_fold_weights():
    """Return [(fold_id, path), ...], one entry per file"""
    paths = sorted(MODEL_DIR.glob(FOLD_WEIGHT_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No fold weights found @ {MODEL_DIR / FOLD_WEIGHT_GLOB}"
        )
    folds = []
    for p in paths:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        fold_id = int(digits) if digits else len(folds) + 1
        folds.append((fold_id, p))
    return folds
 
def discover_dicoms():
    """
    Index Dataset root, return list of dicts {path, stem, centre, dvt_status}
 
    Walks the two known levels explicitly so centre and DVT status come straight 
    from the folder structure and anything outside that layout is reported
    """
    if not DICOM_ROOT.exists():
        raise FileNotFoundError(
            f"DICOM root not found: {DICOM_ROOT.resolve()}  (cwd={Path.cwd()})"
        )
 
    records, skipped_dirs = [], []
    valid_status_names = {n.upper() for n in DVT_FOLDER_NAMES}
 
    for centre_dir in sorted(d for d in DICOM_ROOT.iterdir() if d.is_dir()):
        centre = centre_dir.name
        if CENTRES is not None and centre not in CENTRES:
            continue
 
        for status_dir in sorted(d for d in centre_dir.iterdir() if d.is_dir()):
            if status_dir.name.upper() not in valid_status_names:
                skipped_dirs.append(str(status_dir))
                continue
            dvt_status = normalise_dvt_status(status_dir.name)
 
            for f in sorted(status_dir.rglob("*")):
                if is_dicom_file(f):
                    records.append({
                        "path": f,
                        "stem": f.stem,
                        "centre": centre,
                        "dvt_status": dvt_status,
                    })
 
    if skipped_dirs:
        print(f"[warning] ignored {len(skipped_dirs)} dir that are not DVT/NON_DVT, "
              f"e.g. {skipped_dirs[:3]}")
 
    if not records:
        raise FileNotFoundError(
            f"No DICOMs found under {DICOM_ROOT.resolve()}.\n"
            f"  - expected <CENTRE>/<DVT|NON_DVT>/*.dcm\n"
            f"  - if files have no extension, set INCLUDE_EXTENSIONLESS = True\n"
            f"  - top-level entries seen: {[d.name for d in DICOM_ROOT.iterdir()][:10]}"
        )
    
    seen, deduped = set(), []
    for r in records:
        key = str(r["path"].resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    records = deduped
    
    if RESTRICT_TO_CSV:
        if not METADATA_CSV.exists():
            print(f"[warning] RESTRICT_TO_CSV set but {METADATA_CSV} missing; using all DICOMs")
        else:
            df = pd.read_csv(METADATA_CSV)
            if DICOM_PATH_COLUMN not in df.columns:
                raise KeyError(f"'{DICOM_PATH_COLUMN}' not in {METADATA_CSV} "
                               f"(columns: {list(df.columns)})")
            wanted = {Path(str(s)).stem for s in df[DICOM_PATH_COLUMN].dropna()}
            before = len(records)
            records = [r for r in records if r["stem"] in wanted]
            matched = {r["stem"] for r in records}
            print(f"[info] CSV references {len(wanted)} unique DICOM(s); "
                  f"matched {len(matched)} on disk ({before} -> {len(records)})")
            unmatched = wanted - matched
            if unmatched:
                print(f"[warn] {len(unmatched)} CSV DICOM(s) not found on disk, "
                      f"e.g. {sorted(unmatched)[:3]}")
            if not records:
                raise FileNotFoundError("CSV restriction removed every DICOM; "
                                        "check filename conventions.")
 
    if MAX_DICOMS is not None:
        records = records[:MAX_DICOMS]
 
    by_centre = pd.Series([r["centre"] for r in records]).value_counts().to_dict()
    print(f"[info] {len(records)} DICOM(s) found. Per centre: {by_centre}")
    return records
 
def run_single_dicom(dicom_path, model, device, transform):
    """
    Wall-clock timing of the edge deployed pipeline for single DICOM:
    read DICOM to memory -> per-frame preprocess -> evaluate with n per fold weights
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
 
    ds = pydicom.dcmread(str(dicom_path))
    pixel_array = ds.pixel_array
 
    if pixel_array.ndim == 3:        # T x H x W (grayscale multi-frame)
        frames = pixel_array
    elif pixel_array.ndim == 4:      # T x H x W x 3 (RGB multi-frame)
        frames = pixel_array
    elif pixel_array.ndim == 2:      # H x W (single-frame still)
        frames = pixel_array[None, ...]
    else:
        raise ValueError(f"Unexpected DICOM shape: {pixel_array.shape}")
 
    num_frames = len(frames)
 
    with torch.no_grad():
        for frame in frames:
            frame_f32 = frame.astype(np.float32)
            min_val, max_val = frame_f32.min(), frame_f32.max()
            if max_val > min_val:
                frame_norm = (frame_f32 - min_val) / (max_val - min_val)
            else:
                frame_norm = np.zeros_like(frame_f32)
            frame_uint8 = (frame_norm * 255).astype(np.uint8)
 
            if frame_uint8.ndim == 2:
                img = Image.fromarray(frame_uint8, mode="L").convert("RGB")
            else:
                img = Image.fromarray(frame_uint8, mode="RGB")
 
            input_tensor = transform(img).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            _ = torch.argmax(probs, dim=1).item()
 
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
 
    total_time = end_time - start_time
    return {
        "num_frames": num_frames,
        "total_time_sec": total_time,
        "fps": num_frames / total_time,
        "ms_per_frame": (total_time / num_frames) * 1000.0,
    }
 
 
def warmup(model, device):
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
 
 
def summarise(df):
    """
    Build summary table contianing: 
        mean/std/min/max,
        edge cases of each min and max,
        total elapsed time, fps, ms_per_frame, num_frames
    """
    metrics = ["total_time_sec", "fps", "ms_per_frame", "num_frames"]
    ok = df[df["status"] == "ok"].copy()
    rows = []
 
    def block(scope, sub):
        for m in metrics:
            vals = sub[m].to_numpy(dtype=float)
            n = len(vals)
            if n == 0:
                continue
            imin, imax = sub[m].idxmin(), sub[m].idxmax()
            rows.append({
                "scope": scope,
                "metric": m,
                "n_runs": n,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "min_case": f"fold{sub.loc[imin,'fold']}:{sub.loc[imin,'dicom']}",
                "max_case": f"fold{sub.loc[imax,'fold']}:{sub.loc[imax,'dicom']}",
            })
 
    block("overall", ok)
    for fold in sorted(ok["fold"].unique()):
        block(f"fold_{fold}", ok[ok["fold"] == fold])
    if "centre" in ok.columns:
        for centre in sorted(ok["centre"].dropna().unique()):
            block(f"centre_{centre}", ok[ok["centre"] == centre])
    if "dvt_status" in ok.columns:
        for st in sorted(ok["dvt_status"].dropna().unique()):
            block(f"status_{st}", ok[ok["dvt_status"] == st])
 
    return pd.DataFrame(rows)
 
 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
 
    fold_weights = discover_fold_weights()
    dicoms = discover_dicoms()
    transform = build_transform()
 
    print(f"Model: {MODEL_NAME} | folds: {len(fold_weights)} | "
          f"DICOMs: {len(dicoms)} | total runs: {len(fold_weights) * len(dicoms)}")
 
    model = get_model(model_name=MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    model.to(device)
 
    records = []
    for fold_id, weights_path in fold_weights:
        print(f"\n=== Fold {fold_id} :: {weights_path.name} ===")

        model.load_state_dict(
            torch.load(str(weights_path), map_location=device, weights_only=True)
        )
        model.eval()
        warmup(model, device)
 
        for i, d in enumerate(dicoms, start=1):
            dicom_path = d["path"]
            name = d["stem"]
            base = {
                "fold": fold_id,
                "dicom": name,
                "centre": d["centre"],
                "dvt_status": d["dvt_status"],
                "dicom_path": str(dicom_path),
            }
            try:
                r = run_single_dicom(dicom_path, model, device, transform)
                records.append({
                    **base,
                    "num_frames": r["num_frames"],
                    "total_time_sec": r["total_time_sec"],
                    "fps": r["fps"],
                    "ms_per_frame": r["ms_per_frame"],
                    "status": "ok",
                })
                print(f"  [{i}/{len(dicoms)}] {d['centre']}/{d['dvt_status']}/{name}: "
                      f"{r['num_frames']} frames | {r['total_time_sec']:.3f}s | "
                      f"{r['fps']:.2f} fps | {r['ms_per_frame']:.2f} ms/frame")
            except Exception as e:
                records.append({
                    **base,
                    "num_frames": np.nan, "total_time_sec": np.nan,
                    "fps": np.nan, "ms_per_frame": np.nan,
                    "status": f"error: {type(e).__name__}: {e}",
                })
                print(f"  [{i}/{len(dicoms)}] {name}: SKIPPED ({e})")
 
    per_run = pd.DataFrame(records)
    OUT_PER_RUN_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(OUT_PER_RUN_CSV, index=False)
 
    summary = summarise(per_run)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
 
    n_ok = int((per_run["status"] == "ok").sum())
    n_err = len(per_run) - n_ok
    print(f"\nCompleted: {n_ok} successful runs, {n_err} skipped.")
    print(f"Per-run rows  -> {OUT_PER_RUN_CSV}")
    print(f"Summary stats -> {OUT_SUMMARY_CSV}\n")
 
    overall = summary[summary["scope"] == "overall"].set_index("metric")
    for m in ["total_time_sec", "fps", "ms_per_frame"]:
        if m in overall.index:
            row = overall.loc[m]
            print(f"{m:>15}: {row['mean']:.3f} ± {row['std']:.3f} "
                  f"(min {row['min']:.3f}, max {row['max']:.3f})")
 
 
if __name__ == "__main__":
    main()