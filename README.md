pip install -r requirements.txt, please uninstall torch and install it via cuda at https://pytorch.org/get-started/locally/.



In order to train, go to the datasets folder and run 

```

yolo task=classify mode=train data=dataset model=yolov8m-cls.pt epochs=256

```