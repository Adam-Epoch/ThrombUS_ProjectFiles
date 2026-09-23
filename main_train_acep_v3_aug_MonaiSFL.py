import json
import time
import re

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from monai.losses import FocalLoss as MonaiFocalLoss

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


#%%
TRAIN_CSV = Path("experiment1_dataset/train_metadata_augmented.csv")
EXTERNAL_TEST_CSV = Path("experiment1_dataset/external_test_metadata.csv")

#  mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, efficientnet_b0, resnet50
# densenet121, convnext_tiny
MODEL_NAMES = [
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "convnext_tiny",
]
PRETRAINED = False


#%% check echonous hyperparameters 

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7

FOCAL_ALPHA     = 0.25
FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1

FOCAL_USE_SOFTMAX = True

#%% no change 

IMAGE_SIZE = 224
NUM_CLASSES = 5
N_SPLITS = 5


#%%

def extract_centre_and_dvt_from_dicom_path(dicom_path):
    parts = Path(dicom_path).parts

    for i, part in enumerate(parts):
        if part.upper() == "DVT":
            return parts[i - 1], "DVT"
        if part.upper() in ["NONDVT", "NON-DVT", "NON_DVT"]:
            return parts[i - 1], "nonDVT"

    return None, None

#%%

def extract_patient_code_from_image_path(image_path):
    filename = Path(image_path).name

    match = re.search(r"annonymized_([^_]+)", filename)
    if match:
        return match.group(1)

    match = re.search(r"anonymized_([^_]+)", filename)
    if match:
        return match.group(1)

    return None

#%%

def add_patient_group_columns(df):
    centres = []
    dvt_statuses = []
    patient_codes = []
    patient_group_ids = []

    for _, row in df.iterrows():
        centre, dvt_status = extract_centre_and_dvt_from_dicom_path(row["dicom_path"])
        patient_code = extract_patient_code_from_image_path(row["image_path"])

        centres.append(centre)
        dvt_statuses.append(dvt_status)
        patient_codes.append(patient_code)

        if centre and dvt_status and patient_code:
            patient_group_ids.append(f"{centre}_{dvt_status}_{patient_code}")
        else:
            patient_group_ids.append(None)

    df = df.copy()
    df["centre"] = centres
    df["dvt_status"] = dvt_statuses
    df["patient_code"] = patient_codes
    df["patient_group_id"] = patient_group_ids

    return df


#%% dataset

class ACEPDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["acep_grade"]) - 1

        if self.transform:
            img = self.transform(img)
            
        metadata = {
            "image_path": row["image_path"],
            "dicom_path": row.get("dicom_path", ""),
            "patient_group_id": row.get("patient_group_id", ""),
            "centre": row.get("centre", ""),
            "dvt_status": row.get("dvt_status", "")
            }

        return img, label, metadata


#%% models

def get_model(model_name, pretrained=True, num_classes=5):
    model_name = model_name.lower()

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


#%% loss

class FocalLoss(nn.Module):
    """
    MONAI's FocalLoss w/ label smoothing

    use_softmax=False produces the torchvision sigmoid fl behaviour 
    use_softmax=True switches to softmax focal loss
    label_smoothing=0: each grade is treated as as rigid with 0 overlap 

    Under the softmax branch, MONAI interprets a scalar `alpha` as a 
    foreground/background weight (class 0 is weighted: 1 - alpha, the rest alpha), giving class 0 privilage.
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=5,
                 label_smoothing=0.0, use_softmax=False, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal = MonaiFocalLoss(
            gamma=gamma,
            alpha=alpha,
            use_softmax=use_softmax,
            reduction=reduction,
            to_onehot_y=False,      # we hand it a (smoothed) soft one-hot target
            include_background=True,
        )

    def forward(self, logits, target):
        # (B,) int64 -> (B, C) float one-hot, same device/dtype as logits
        target_1h = F.one_hot(target, num_classes=self.num_classes).to(logits.dtype)
        if self.label_smoothing > 0.0:
            target_1h = (
                target_1h * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )
        return self.focal(logits, target_1h)


#%% training

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels, _ in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

#%% evaluation

def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_metadata = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            batch_size = images.size(0)
            for i in range(batch_size):
                all_metadata.append({
                    "image_path": metadata["image_path"][i],
                    "dicom_path": metadata["dicom_path"][i],
                    "patient_group_id": metadata["patient_group_id"][i],
                    "centre": metadata["centre"][i],
                    "dvt_status": metadata["dvt_status"][i]
                    })

    avg_loss = total_loss / len(loader.dataset)

    if return_predictions:
        return (avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs), all_metadata)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1

#%% inference time 

def benchmark_inference_time(model, loader, device, warmup_batches=3):
    model.eval()
    image_times = []

    with torch.no_grad():
        for i, (images, _, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for images, _, _ in tqdm(loader, desc="Benchmarking inference time"):
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            per_image_time = (end - start) / images.size(0)
            image_times.extend([per_image_time] * images.size(0))

    image_times = np.array(image_times)

    return {
        "mean_inference_time_per_image_sec": float(image_times.mean()),
        "mean_inference_time_per_image_ms": float(image_times.mean() * 1000),
        "std_inference_time_per_image_ms": float(image_times.std() * 1000),
        "median_inference_time_per_image_ms": float(np.median(image_times) * 1000),
        "fps": float(1.0 / image_times.mean()),
    }

#%% final evaluation 

def final_evaluation(model, loader, criterion, device, output_dir, split_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss, y_true, y_pred, y_probs, metadata = evaluate(
        model,
        loader,
        criterion,
        device,
        return_predictions=True
    )

    y_true_acep = y_true + 1
    y_pred_acep = y_pred + 1

    metrics = {
        "split": split_name,
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_absolute_grade_error": float(np.mean(np.abs(y_true_acep - y_pred_acep))),
        "within_1_grade_accuracy": float(np.mean(np.abs(y_true_acep - y_pred_acep) <= 1)),
    }

    time_metrics = benchmark_inference_time(model, loader, device)
    metrics.update(time_metrics)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_true_acep, y_pred_acep, labels=[1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in [1, 2, 3, 4, 5]],
        columns=[f"pred_{i}" for i in [1, 2, 3, 4, 5]]
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    report = classification_report(
        y_true_acep,
        y_pred_acep,
        labels=[1, 2, 3, 4, 5],
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    pred_df = pd.DataFrame(metadata)
    
    pred_df["true_grade"] = y_true_acep
    pred_df["pred_grade"] = y_pred_acep
    pred_df["absolute_grade_error"] = np.abs(y_true_acep - y_pred_acep)
    pred_df["within_1_grade"] = pred_df["absolute_grade_error"] <= 1

    for i in range(5):
        pred_df[f"prob_grade_{i+1}"] = y_probs[:, i]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    return metrics


#%% main

def main(model_name):
    train_df = pd.read_csv(TRAIN_CSV)
    external_df = pd.read_csv(EXTERNAL_TEST_CSV)

    train_df = train_df.dropna(subset=["image_path", "acep_grade", "dicom_path"]).reset_index(drop=True)
    external_df = external_df.dropna(subset=["image_path", "acep_grade", "dicom_path"]).reset_index(drop=True)

    train_df = add_patient_group_columns(train_df)
    external_df = add_patient_group_columns(external_df)

    train_df = train_df.dropna(subset=["patient_group_id"]).reset_index(drop=True)
    external_df = external_df.dropna(subset=["patient_group_id"]).reset_index(drop=True)

    print("Training samples:", len(train_df))
    print("External test samples:", len(external_df))
    print("Unique training patients:", train_df["patient_group_id"].nunique())
    print("Unique external patients:", external_df["patient_group_id"].nunique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    external_ds = ACEPDataset(external_df, val_tfms)
    external_loader = DataLoader(
        external_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    base_results_dir = Path("results") / f"{model_name}_pretrained_{PRETRAINED}"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    all_val_metrics = []
    all_external_metrics = []

    groups = train_df["patient_group_id"]

    for fold, (fold_train_idx, fold_val_idx) in enumerate(
        sgkf.split(X=train_df, y=train_df["acep_grade"], groups=groups),
        start=1
    ):
        print(f"\n Fold {fold}/{N_SPLITS}")

        fold_train_df = train_df.iloc[fold_train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[fold_val_idx].reset_index(drop=True)
        
        print("Train ACEP Distribution:")
        print(fold_train_df["acep_grade"].value_counts(normalize=True).sort_index())
        
        print("Val ACEP Distribution")
        print(fold_val_df["acep_grade"].value_counts(normalize=True).sort_index())

        train_patients = set(fold_train_df["patient_group_id"])
        val_patients = set(fold_val_df["patient_group_id"])
        overlap = train_patients & val_patients
        print("Patient overlap:", len(overlap))

        fold_train_ds = ACEPDataset(fold_train_df, train_tfms)
        fold_val_ds = ACEPDataset(fold_val_df, val_tfms)

        fold_train_loader = DataLoader(
            fold_train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        fold_val_loader = DataLoader(
            fold_val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        model = get_model(
            model_name=model_name,
            pretrained=PRETRAINED,
            num_classes=NUM_CLASSES
        ).to(device)

        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            num_classes=NUM_CLASSES,
            label_smoothing=LABEL_SMOOTHING,
            use_softmax=FOCAL_USE_SOFTMAX,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LR,
            weight_decay=WEIGHT_DECAY)

        fold_save_path = base_results_dir / f"best_fold_{fold}.pth"
        
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        history = []

        for epoch in range(EPOCHS):
            print(f"\nFold {fold} | Epoch {epoch + 1}/{EPOCHS}")

            train_loss = train_one_epoch(
                model,
                fold_train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, val_acc, val_f1 = evaluate(
                model,
                fold_val_loader,
                criterion,
                device
            )
            
            history.append({
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1
                })

            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss:   {val_loss:.4f}")
            print(f"Val acc:    {val_acc:.4f}")
            print(f"Val F1:     {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), fold_save_path)
                print(f"Saved best model for fold {fold}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")
                
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        history_df = pd.DataFrame(history)
        history_df.to_csv(base_results_dir/ f"fold_{fold}_training_history.csv", index=False)

        model.load_state_dict(torch.load(fold_save_path, map_location=device))

        val_results_dir = base_results_dir / f"fold_{fold}" / "validation"
        external_results_dir = base_results_dir / f"fold_{fold}" / "external_test"

        val_metrics = final_evaluation(
            model=model,
            loader=fold_val_loader,
            criterion=criterion,
            device=device,
            output_dir=val_results_dir,
            split_name="validation"
        )
        val_metrics["fold"] = fold
        val_metrics["best_val_f1_during_training"] = float(best_val_f1)
        all_val_metrics.append(val_metrics)

        external_metrics = final_evaluation(
            model=model,
            loader=external_loader,
            criterion=criterion,
            device=device,
            output_dir=external_results_dir,
            split_name="external_test"
        )
        external_metrics["fold"] = fold
        all_external_metrics.append(external_metrics)

    val_summary_df = pd.DataFrame(all_val_metrics)
    external_summary_df = pd.DataFrame(all_external_metrics)

    val_summary_df.to_csv(base_results_dir / "validation_fold_metrics.csv", index=False)
    external_summary_df.to_csv(base_results_dir / "external_test_fold_metrics.csv", index=False)

    val_stats = val_summary_df.drop(columns=["split"]).agg(["mean", "std"]).transpose()
    external_stats = external_summary_df.drop(columns=["split"]).agg(["mean", "std"]).transpose()

    val_stats.to_csv(base_results_dir / "validation_mean_std.csv")
    external_stats.to_csv(base_results_dir / "external_test_mean_std.csv")

    print("\nValidation mean ± std:")
    print(val_stats)

    print("\nExternal test mean ± std:")
    print(external_stats)

#%%

if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        print(f"training model: {model_name}")
        main(model_name)
