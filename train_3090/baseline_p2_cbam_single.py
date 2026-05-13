from ultralytics import YOLO


if __name__ == "__main__":
    # Single-scale P2 + CBAM, train from scratch
    model = YOLO("../ultralytics/cfg/models/11/yolo11-p2-cbam-single.yaml")

    results = model.train(
        data="VisDrone.yaml",
        epochs=200,
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
        cache=False,

        seed=0,
        deterministic=True,

        project="VisDrone_Thesis_200e",
        name="yolo11n_p2_cbam_single_scratch_200e",
    )