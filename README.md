# MIMIC-IV CXR mortality & readmission model

MIMIC-IV CXR mortality & readmission model

## About

## Structure

### Steps

00_cxr_csv_check.ipynb: Check cxr info and filter records 24h/48h before or within icu.

Run to create cxr_filtered_f24h.csv.gz for moratlity training and cxr_filtered_all.csv.gz for readmission training 


01a_cxr_mortality_preprocess.ipynb: Preprocess image files h5 for mortality before training. 

Run to create processed_images_mortality.h5 for training. Required all_unstructured_data.csv from MIMIC_IV EHR data pipeline.

01a_cxr_mortality_preprocess.py: Runable version of 01a.


01b_cxr_readmission_preprocess.ipynb: Preprocess image files h5 for readmission before training. 

Run to create processed_images_readmission.h5 for training. Required all_unstructured_data.csv from MIMIC_IV EHR data pipeline.

01b_cxr_readmission_preprocess.py: Runable version of 01b.


02a_training_mortality.ipynb: Training process of mortality.


02b_training_readmission.ipynb: Training process of readmission.
