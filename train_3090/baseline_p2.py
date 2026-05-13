from ultralytics import YOLO

if __name__ == "__main__":
    # 不使用预训练权重，从零训练
    model = YOLO("yolo11-p2.yaml")

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
        name="yolo11n_baseline_p2_scratch_200e",
    )
