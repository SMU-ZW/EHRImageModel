import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision


# ====== Config ======
data_path = Path('../MIMIC/physionet.org/content/mimic-cxr-jpg/get-zip/2.1.0/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0')
f24_path = Path('cxr_filtered_f24h.csv.gz')
structured_path = Path('all_structured_data.csv')
unstructured_path = Path('all_unstructured_data.csv')

def list_files_in_directory(directory, level=0, max_depth=2):
    if level >= max_depth:
        return
    
    for item in sorted(directory.iterdir()):
        indent = '    ' * level
        if item.is_dir():
            if level == max_depth - 1:
                count = sum(1 for _ in item.iterdir())
                print(f"{indent}[DIR] {item.name}: {count} items")
            else:
                print(f"{indent}[DIR] {item.name}")
                list_files_in_directory(item, level + 1, max_depth)
        else:
            print(f"{indent}{item.name}")

def format_time(t):
    try:
        t = float(t)
        int_part = int(t)
        decimal_part = f"{t:.6f}".split(".")[1]  
        int_str = str(int_part).zfill(6)        
        return f"{int_str}.{decimal_part}"
    except:
        return "000000.000000"

def select_key_dicom_ids(df):
    if "StudyDateTime" not in df.columns:
        raise ValueError("❗️Missing 'StudyDateTime' column. Please construct it first.")

    df = df[df["StudyDateTime"].notna()].copy()  
    results = []

    for subject_id, group in df.groupby("subject_id"):
        group = group.sort_values("StudyDateTime").reset_index(drop=True)

        if len(group) == 1:
            indices = [0, 0, 0]
        elif len(group) == 2:
            indices = [0, 1, 1]
        else:
            indices = [0, len(group) // 2, len(group) - 1]

        selected_dicom_ids = group.iloc[indices]["dicom_id"].tolist()
        study_datetime = group.iloc[indices[1]]["StudyDateTime"]  

        results.append({
            "subject_id": subject_id,
            "StudyDateTime": study_datetime,
            "files": selected_dicom_ids
        })

    return pd.DataFrame(results)

def preprocess_images(df_actual_files, base_path, df_original, max_display=10):
    processed_images = []
    dicom_to_date = df_original.set_index("dicom_id")["StudyDateTime"].astype(str).to_dict()

    shown_count = 0  

    for _, row in tqdm(df_actual_files.iterrows(), total=len(df_actual_files), desc="Processing Images"):
        subject_id = row["subject_id"]
        dicom_list = row["files"]

        fig, axes = None, None
        if shown_count < max_display:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, dicom_id in enumerate(dicom_list):
            file_name = f"{dicom_id}.jpg"
            study_date = dicom_to_date.get(dicom_id, "Unknown")

            subject_prefix = f"p{str(subject_id)[:2]}"
            subject_dir = f"p{subject_id}"
            found = False

            for study_path in (base_path / subject_prefix / subject_dir).glob("s*"):
                file_path = study_path / file_name
                if file_path.exists():
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = img.astype(np.float32) / 255.0
                        processed_images.append(img)

                        if shown_count < max_display:
                            axes[i].imshow(img)
                            axes[i].set_title(file_name)
                            axes[i].text(0.5, -0.1, f"StudyDateTime: {study_date}",
                                         transform=axes[i].transAxes,
                                         ha='center', va='top', fontsize=10)
                            axes[i].axis("off")
                        found = True
                        break

            if not found and shown_count < max_display:
                axes[i].set_title("Not Found")
                axes[i].text(0.5, -0.1, f"StudyDateTime: {study_date}",
                             transform=axes[i].transAxes,
                             ha='center', va='top', fontsize=10)
                axes[i].axis("off")

        if shown_count < max_display:
            plt.suptitle(f"Subject ID: {subject_id}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            shown_count += 1  

    return processed_images

def save_processed_images_to_hdf5(processed_images, df_actual_files, df_original, hdf5_path="processed_images.h5"):
    dicom_to_date = df_original.set_index("dicom_id")["StudyDateTime"].astype(str).to_dict()
    grouped = df_actual_files.groupby("subject_id")["files"].apply(list).to_dict()
    label_dict = df_actual_files.set_index("subject_id")["mortality_icu"].to_dict()

    feature_index = 0
    with h5py.File(hdf5_path, "w") as hf:
        for subject_id, dicom_ids in grouped.items():
            subject_imgs = []
            subject_dates = []

            for dicom_list in dicom_ids:
                for dicom_id in dicom_list:
                    if feature_index < len(processed_images):
                        subject_imgs.append(processed_images[feature_index])
                        feature_index += 1
                    else:
                        subject_imgs.append(np.zeros((224, 224, 3)))
                    study_date = dicom_to_date.get(dicom_id, "Unknown")
                    subject_dates.append(study_date)

            while len(subject_imgs) < 3:
                subject_imgs.append(subject_imgs[-1] if subject_imgs else np.zeros((224, 224, 3)))
                subject_dates.append(subject_dates[-1] if subject_dates else "Unknown")

            subject_imgs = np.array(subject_imgs[:3])
            subject_dates = subject_dates[:3]

            grp = hf.create_group(str(subject_id))
            grp.create_dataset("images", data=subject_imgs, compression="gzip")
            for i, date in enumerate(subject_dates):
                grp.create_dataset(f"studydate_{i}", data=date.encode("utf-8"))

            if subject_id in label_dict:
                grp.create_dataset("mortality_icu", data=int(label_dict[subject_id]))

    print(f"✅ Process image saved to {hdf5_path}")


def main():
    if data_path.exists():
        list_files_in_directory(data_path, max_depth=2)
    else:
        print(f"Directory '{data_path}' does not exist.")
        
    df = pd.read_csv(f24_path, compression="gzip" if f24_path.suffix == ".gz" else None)
    df = df.drop_duplicates()
    df["StudyDateStr"] = df["StudyDate"].astype(str)
    df["StudyDateStr"] = df["StudyDate"].astype(str).str.zfill(8)
    df["StudyTimeStr"] = df["StudyTime"].apply(format_time)

    df["StudyDateTime"] = pd.to_datetime(
        df["StudyDateStr"] + " " + df["StudyTimeStr"],
        format="%Y%m%d %H%M%S.%f",
        errors="coerce"
    )
    df_sorted = df.sort_values(by=["subject_id", "StudyDateTime","ViewPosition"])
    df_sorted["order"] = df_sorted.groupby("subject_id").cumcount() + 1
    # df_sorted = df_sorted[df_sorted["ViewPosition"] == "AP"]
    df_final = df_sorted[["subject_id", "study_id", "StudyDateTime" ,"dicom_id", "ViewPosition" ,"order"]]
    df_actual_files = select_key_dicom_ids(df_final)
    df_labels = pd.read_csv(unstructured_path)
    df_labels = df_labels[['subject_id', 'mortality_icu']]
    df_merged = pd.merge(
        df_actual_files,
        df_labels,
        on='subject_id',
        how='left',
        indicator=True
    )

    df_unmatched = df_merged[df_merged['_merge'] == 'left_only']
    df_actual_files = df_merged[df_merged['_merge'] != 'left_only'].drop(columns=['_merge'])
    processed_images = preprocess_images(df_actual_files, data_path / "files",  df_final, max_display=0)
    save_processed_images_to_hdf5(processed_images, df_actual_files, df_final, hdf5_path="processed_images_mortality.h5")

# ====== Run ======
if __name__ == "__main__":
    main()
