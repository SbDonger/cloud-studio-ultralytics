from ultralytics import YOLO

if __name__ == '__main__':
    # 消融实验：P2 + CBAM（不含 WIoU，使用默认 CIoU）
    model = YOLO("ultralytics/cfg/models/11/yolo11-p2-cbam.yaml")
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
        name="yolo11n_p2_cbam",
    )
