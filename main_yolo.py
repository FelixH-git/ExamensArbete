from datasets import load_dataset
from datasets import DatasetDict
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from parquet_to_yolo import split_data
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


print(train_ds)
print(val_ds)
print(torch.cuda.is_available())

shown_labels = set()
#f.write(f"{data["label"][i]}")


#for i in range(len(train_ds)):
split_data(train_ds, "train")
split_data(val_ds, "val")
split_data(test_ds, "test")

# model = YOLO("yolov8m-cls.pt")


# model.train(
#     data="data.yaml",
#     epochs=100,
#     imgsz=640,
#     device= 0 if torch.cuda.is_available() else "cpu",
#     save=True
# )





# plt.figure(figsize=(10, 10))

    

#train_ds["image"][0].save(fr"breastcancer-ultrasound-images\data\train\test.png")

#print(train_ds["label"][0])

# with open("test.txt", "w") as f:
#     f.write(train_ds["label"][0])
 

    
    
    
        
#plt.show()