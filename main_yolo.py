from datasets import load_dataset
from datasets import DatasetDict
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from parquet_to_yolo import split_data
import matplotlib.pyplot as plt
import PIL
###DETTA ÄR ETT CLASSIFICATION DATASET, VI TRÄNAR YOLOV8m-cls
dataset = load_dataset("emre570/breastcancer-ultrasound-images")

test_num = len(dataset["test"])
train_num = len(dataset["train"])


val_size = test_num / train_num

train_val_split = dataset["train"].train_test_split(test_size=val_size)


#print(train_val_split)

dataset = DatasetDict(
    {"train": train_val_split["train"], "validation": train_val_split["test"], "test": dataset["test"]}
)

train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

# print(train_ds['label'], "Train")
# print(val_ds['label'], "Validation")

how_many_labels_per_class_val = {"benign": 0, "malignant": 0, "normal": 0}


for i in range(len(val_ds["label"])):
    if train_ds["label"][i] == 0:
        how_many_labels_per_class_val["benign"] += 1
    if train_ds["label"][i] == 1:
        how_many_labels_per_class_val["malignant"] += 1
    if train_ds["label"][i] == 2:
        how_many_labels_per_class_val["normal"] += 1
        
         
print(how_many_labels_per_class_val)

plt.bar(how_many_labels_per_class_val.keys(), how_many_labels_per_class_val.values(), width=0.8, color="r")
plt.show()



### Split data set into yolov8 format look at parquet to yolo
# split_data(train_ds, "train")
# split_data(val_ds, "val")
# split_data(test_ds, "test")

# model = YOLO("yolov8m-cls.pt")


# model.train(
#     data="data.yaml",
#     epochs=100,
#     imgsz=640,
#     device= 0 if torch.cuda.is_available() else "cpu",
#     save=True
# )





 

    
    
   