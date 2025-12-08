import torch
from ultralytics import YOLO

model = YOLO("yolo11s-obb.pt")  # will auto-download

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,                 # use GPU assuming it's available
    project="results", 
    name="y11s_obb_local_train_50epochs"
)
