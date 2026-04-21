from ultralytics import YOLO

if __name__ == '__main__':
    # 基准实验：官方 YOLO11n（原始 3 尺度，无 P2）
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="VisDrone.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,

        # --- 显存与速度优化 ---
        amp=True,
        workers=4,

        # --- 学习率与收敛优化 ---
        cos_lr=True,
        patience=20,
        close_mosaic=10,

        # --- 数据增强（与 P2 实验保持一致） ---
        mixup=0.0,
        copy_paste=0.0,

        # --- 实验管理 ---
        project="VisDrone_Thesis",
        name="yolo11n_baseline",

        # --- 可复现性（与 P2 实验保持一致，如需严格复现请取消注释） ---
        # seed=42,
        # deterministic=True,
    )
