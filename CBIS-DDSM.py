import os
import pandas as pd

path = "cbis-ddsm"

for files in os.listdir(path):
    files_dir = os.path.join(path, files)
    
#     print(os.path.join(path, files))
#     print(f"file {files} has {len(os.listdir(files_dir))} files")

    if files == 'jpeg':   # to pass 6774 files 
        pass
    else:
        for file in os.listdir(files_dir):
            print(file)
            
            
dicom_df = pd.read_csv(path + "/csv/dicom_info.csv")

#print(dicom_df.head())

#print(dicom_df.describe().T)

#print(dicom_df.info())

#print(dicom_df.SeriesDescription.unique())

#print(dicom_df.SeriesDescription.value_counts())


cropped_images = dicom_df[dicom_df.SeriesDescription=="cropped images"].image_path

#print(cropped_images.head())

full_mammogram = dicom_df[dicom_df.SeriesDescription=="full mammogram images"].image_path

#print(full_mammogram.head())


roi_mask = dicom_df[dicom_df.SeriesDescription=="ROI mask images"].image_path

print(roi_mask.head())
