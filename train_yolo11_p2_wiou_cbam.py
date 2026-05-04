import os
os.environ["YOLO_WIOU"] = "true"

from ultralytics import YOLO

if __name__ == '__main__':
    # P2 + WIoU + CBAM
    model = YOLO("yolo11-p2-wiou-cbam.yaml")
    model.load("yolo11n.pt")

    results = model.train(
        data="VisDrone.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        workers=4,
        cos_lr=True,
        patience=20,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        project="VisDrone_Thesis",
        name="yolo11n_p2_wiou_cbam",
    )
