from ultralytics import YOLO

model = YOLO("yolo11s-obb.pt")  # will auto-download

results = model.train(
    data="data.yaml",
    epochs=3,
    imgsz=640,
    batch=2,
    #device=0,                 # use GPU assuming it's available
    project="results", 
    name="y11s_obb_custom"
)
