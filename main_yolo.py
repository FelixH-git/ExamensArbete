from datasets import load_dataset
from datasets import DatasetDict
import matplotlib.pyplot as plt
###DETTA ÄR ETT CLASSIFICATION DATASET, VI TRÄNAR YOLOV8m-cls
dataset = load_dataset("emre570/breastcancer-ultrasound-images")

test_num = len(dataset["test"])
train_num = len(dataset["train"])


val_size = test_num / train_num

train_val_split = dataset["train"].train_test_split(test_size=val_size)


print(train_val_split)

dataset = DatasetDict(
    {"train": train_val_split["train"], "validation": train_val_split["test"], "test": dataset["test"]}
)

train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]


shown_labels = set()

plt.figure(figsize=(10, 10))

for i, sample in enumerate(train_ds):
    label = train_ds.features["label"].names[sample["label"]]
    if label not in shown_labels:
        plt.subplot(1, len(train_ds.features["label"].names), len(shown_labels) + 1)
        plt.imshow(sample["image"])
        plt.title(label)
        plt.axis("off")
        shown_labels.add(label)
        if len(shown_labels) == len(train_ds.features["label"].names):
            break

plt.show()