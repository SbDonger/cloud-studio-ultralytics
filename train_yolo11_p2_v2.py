from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11-p2.yaml")
    model.load("yolo11n.pt")

    results = model.train(
        data="VisDrone.yaml",
        epochs=300,
        imgsz=1280,           # 关键：高分辨率让 P2 层真正发挥作用
        batch=4,              # A10 24GB 的安全批次
        device=0,
        amp=True,
        workers=4,
        cos_lr=True,
        patience=50,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        tal_topk=13,          # 4 层检测多分正样本
        scale=0.9,            # 增强小目标缩放多样性
        project="VisDrone_Thesis",
        name="yolo11n_p2_1280",
    )
