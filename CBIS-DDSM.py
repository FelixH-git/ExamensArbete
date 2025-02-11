import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import cv2
import numpy as np





path = "CBIS-DDSM"

for files in os.listdir(path):
    files_dir = os.path.join(path, files)
    
#     print(os.path.join(path, files))
#     print(f"file {files} has {len(os.listdir(files_dir))} files")

    if files == 'jpeg':   # to pass 6774 files 
        pass
    else:
        for file in os.listdir(files_dir):
            pass
            #print(file)
            
            
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


mass_train = pd.read_csv("CBIS-DDSM/csv/mass_case_description_train_set.csv")
mass_test = pd.read_csv("CBIS-DDSM/csv/mass_case_description_test_set.csv")


print(mass_train.iloc[:, 11].head())

mass_train = mass_train.rename(columns={"left or right breast": "left_or_right_breast", 
                                        "image view": "image_view",
                                        "abnormality id": "abnormality_id",
                                        "abnormality type": "abnormality_type",
                                        "mass shape": "mass_shape",
                                        "mass margins": "mass_margins",
                                        "image file path": "image_file_path",
                                        "cropped image file path": "cropped_image_file_path",
                                        "ROI mask file path": "ROI_mask_file_path"})

mass_test = mass_test.rename(columns={"left or right breast": "left_or_right_breast", 
                                        "image view": "image_view",
                                        "abnormality id": "abnormality_id",
                                        "abnormality type": "abnormality_type",
                                        "mass shape": "mass_shape",
                                        "mass margins": "mass_margins",
                                        "image file path": "image_file_path",
                                        "cropped image file path": "cropped_image_file_path",
                                        "ROI mask file path": "ROI_mask_file_path"})


print(f"shape of mass train {mass_train.shape}")
print(f"shape of mass test {mass_test.shape}")
#print(mass_train.pathology.unique())

#print(mass_train.head())
#print(mass_train.info())

def plot_samples(sample, row=15, col=15):
    plt.figure(figsize=(row,col))
    for i, file in enumerate(sample[0:5]):
        cropped_images_show = PIL.Image.open(file)
        gray_img = cropped_images_show.convert("L")
        plt.subplot(1, 5, i+1)
        plt.imshow(gray_img, cmap="gray")
        plt.axis("off")
    plt.show()
    


def display_images(dataset, column, number):
    """Displays images in dataset, handling missing files and converting formats."""
    
    # create figure and axes
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    
    # Loop through rows and display images
    for index, (i, row) in enumerate(dataset.head(number).iterrows()):
        image_path = row[column]
        
       # Check if image_path is valid (not None) and exists
        if image_path is None or not os.path.exists(image_path):
            # print(f"File not found or invalid path: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        
        # Handle case when image can't be read
        if image is None:
            # print(f"Error reading image: {image_path}")
            continue
        
        # Convert BGR to RGB if needed (for correct color display)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax = axes[index]
        ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
        print(np.array(image).shape)
    
    plt.tight_layout()
    plt.show()
    
    
#display_images(mass_train, 'image_file_path', 5)
# print("Cropped images paths:")
# print(cropped_images.iloc[0])

#plot_samples(full_mammogram, 15, 15)

