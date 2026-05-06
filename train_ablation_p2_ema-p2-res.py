from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-p2-ema.yaml")
    model.load("yolo11n.pt")

    results = model.train(
        data="VisDrone.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        amp=True,
        workers=8,
        patience=0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,

        seed=0,
        deterministic=True,

        project="VisDrone_Thesis_150e",
        name="yolo11n_p2_ema_res_150e",
    )
